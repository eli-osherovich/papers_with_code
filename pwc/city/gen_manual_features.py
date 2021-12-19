#!/usr/bin/env python

import pathlib

from absl import app
import pandas as pd
from prefect import Flow
from prefect import task
from prefect.utilities import logging

from pwc.datasets import HomeCreditDefaultRisk as Dataset
from pwc.city import task_lib

_logger = logging.get_logger()
_id = "SK_ID_CURR"

set_nans = task(task_lib.set_nans)


@task
def load_dataset() -> dict[str, pd.DataFrame]:
  ds = Dataset()
  data = {}
  for d in [
    "bureau_balance", "bureau", "credit_card_balance", "installments_payments",
    "pos_cash_balance", "previous_application", "train"
  ]:
    data[d] = ds.as_dataframe(d)
  return data


@task
def gen_group_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
  df = data["train"]
  res_df = df[[_id]].copy()
  for group in [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "CNT_CHILDREN", "NAME_INCOME_TYPE",
    "NAME_FAMILY_STATUS", "OCCUPATION_TYPE", "CNT_FAM_MEMBERS",
    "ORGANIZATION_TYPE"
  ]:
    new_feat_col = f"manual__INCOME_BY_{group}_MEDIAN"
    new_rel_col = f"manual__INCOME_REL_TO_{group}"
    inc_by_feat = df[[
      "AMT_INCOME_TOTAL",
      group,
    ]].groupby(group).median()["AMT_INCOME_TOTAL"]
    res_df[new_feat_col] = df[group].map(inc_by_feat)
    res_df[new_rel_col] = df["AMT_INCOME_TOTAL"] / res_df[new_feat_col]
  return res_df


@task
def gen_docs_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
  df = data["train"]
  res_df = df[[_id]].copy()
  cols = [f for f in df.columns if "FLAG_DOC" in f]
  for func in ["min", "max", "mean", "median", "std", "kurtosis"]:
    res_df[f"manual__ALL_DOCS_{func}"] = df[cols].agg(func, axis=1)

  return res_df


@task
def gen_flag_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
  df = data["train"]
  res_df = df[[_id]].copy()
  cols = [
    f for f in df.columns
    if ("FLAG_" in f) & ("FLAG_DOC" not in f) & ("_FLAG_" not in f)
  ]
  for func in ["min", "max", "mean", "median", "std", "kurtosis"]:
    res_df[f"manual__OTHER_FLAGS_{func}"] = df[cols].agg(func, axis=1)

  return res_df


@task
def gen_ext_sources_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
  df = data["train"]
  res = df[[_id]].copy()
  cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
  for func in ["min", "max", "mean", "median", "std", "kurtosis"]:
    res[f"manual__EXT_SOURCES_{func}"] = df[cols].agg(func, axis=1)
  res["manual__EXT_SOURCES_mul"] = (
    df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
  )

  return res


@task
def gen_bureau_balance_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
  df = data["bureau_balance"]
  df = pd.get_dummies(df, dummy_na=True)
  res = df.groupby("SK_ID_BUREAU").agg(["min", "max", "mean", "size"])
  res.columns = task_lib.flatten_col_names(res, prefix="manual__BUREAU_BALANCE")
  return res


@task
def gen_bureau_features(
  data: dict[str, pd.DataFrame], bb_df: pd.DataFrame
) -> pd.DataFrame:
  df = data["bureau"]

  df = df.join(bb_df, on="SK_ID_BUREAU")
  df = df.drop(["SK_ID_BUREAU"], axis=1)
  dummies = pd.get_dummies(df, dummy_na=True)

  res = dummies.groupby(_id).agg(["min", "max", "mean", "size"])
  res.columns = task_lib.flatten_col_names(res, prefix="manual__BUREAU")

  for ca in df["CREDIT_ACTIVE"].unique():
    # Aggregations for different credit active statuses.
    agg = df[df["CREDIT_ACTIVE"] == ca].groupby(_id).agg([
      "min", "max", "mean", "size"
    ])
    agg.columns = task_lib.flatten_col_names(
      agg, prefix=f"manual_BUREAU_CREDIT_ACTIVE_{ca}"
    )
    res = res.join(agg, on=_id)
  return res


@task
def gen_previous_application_features(
  data: dict[str, pd.DataFrame]
) -> pd.DataFrame:
  df = data["previous_application"].copy()
  df["APP_CREDIT_PERC"] = df["AMT_APPLICATION"] / df["AMT_CREDIT"]

  dummies = pd.get_dummies(df, dummy_na=True)
  dummies = dummies.drop("SK_ID_PREV", axis=1)
  res = dummies.groupby(_id).agg(["mean"])
  res.columns = task_lib.flatten_col_names(res, prefix="manual__PREVAPP")

  # Number of previous applications
  num_prev_app = df[["SK_ID_PREV", _id]].groupby(_id).nunique()
  num_prev_app.columns = ["manual__NPREV_COUNT"]
  res = res.join(num_prev_app, on=_id)
  df = df.drop("SK_ID_PREV", axis=1)

  # Calculate aggregations of the original numeric columns
  agg = df.select_dtypes("number").groupby(_id).agg([
    "min", "max", "var", "sum"
  ])
  agg.columns = task_lib.flatten_col_names(agg, prefix="manual__PREVAPP")
  res = res.join(agg, on=_id)
  return res

  #   for status in df["NAME_CONTRACT_STATUS"].unique():
  #     # Aggregations for different credit active statuses.
  #     agg = df[df["NAME_CONTRACT_STATUS"] == status].groupby(_id
  #                                                           ).agg(aggregations)
  #     agg.columns = task_lib.flatten_col_names(
  #       agg, prefix="manual__PREVAPP_NAME_CONTRACT"
  #     )
  #     res = res.join(agg, on=_id)

  return res


@task
def gen_pos_cash_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
  df = data["pos_cash_balance"]
  res = pd.get_dummies(df)
  aggregations = ["min", "max", "mean", "size"]
  res = res.groupby(_id).agg(aggregations)
  res.columns = task_lib.flatten_col_names(res, prefix="manual__POS")
  # Count pos cash accounts
  res["manual__POS_COUNT"] = res.groupby(_id).size()
  return res


@task
def gen_credit_card_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
  df = data["credit_card_balance"]
  df = df.drop("SK_ID_PREV", axis=1)
  res = df.groupby(_id).agg(["min", "max", "mean", "sum", "var"])
  res.columns = task_lib.flatten_col_names(res, prefix="manual__CC")
  # Count credit card lines
  res["manual__CC_COUNT"] = df.groupby(_id).size()
  return res


@task
def gen_installments_payments_features(
  data: dict[str, pd.DataFrame]
) -> pd.DataFrame:
  df = data["installments_payments"]
  # Percentage and difference paid in each installment (amount paid and installment value)
  df["manual__PAYMENT_PERC"] = df["AMT_PAYMENT"] / df["AMT_INSTALMENT"]
  df["manual__PAYMENT_DIFF"] = df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]
  # Days past due and days before due (no negative values)
  df["manual__DPD"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
  df["manual__DBD"] = df["DAYS_INSTALMENT"] - df["DAYS_ENTRY_PAYMENT"]
  df["manual__DPD"] = df["manual__DPD"].apply(lambda x: x if x > 0 else 0)
  df["manual__DBD"] = df["manual__DBD"].apply(lambda x: x if x > 0 else 0)
  # Features: Perform aggregations
  aggregations = ["min", "max", "mean", "sum", "var"]
  res = df.groupby(_id).agg(aggregations)
  nunique = df[[_id, "NUM_INSTALMENT_VERSION",
                "SK_ID_PREV"]].groupby(_id).agg(["nunique"])
  res = res.join(nunique, on=_id)
  res.columns = task_lib.flatten_col_names(res, prefix="manual__INSTPAY")

  # Count installments accounts
  res["manual__INSTAL_COUNT"] = df.groupby(_id).size()
  return res


@task
def merge_df(dataframes: list[pd.DataFrame], *, logger=None) -> None:
  res = dataframes[0]
  for df in dataframes[1:]:
    res = res.merge(df, how="left", on=_id)
    if logger:
      logger.info(f"New shape: {res.shape}")

  fpath = pathlib.Path(__file__).parent / "data" / "manual.feather"

  res.to_feather(fpath)


@task
def get_id(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
  return data["train"][[_id]]


def main(argv):
  del argv  # not used
  with Flow("Manual features") as flow:
    data = load_dataset()
    data = set_nans(data)
    id_df = get_id(data)
    group_feats = gen_group_features(data)
    ext_sources_feats = gen_ext_sources_features(data)
    docs_features = gen_docs_features(data)
    flags_features = gen_flag_features(data)
    bureau_balance_features = gen_bureau_balance_features(data)
    bureau_features = gen_bureau_features(data, bureau_balance_features)
    prev_app_features = gen_previous_application_features(data)
    pos_cash_features = gen_pos_cash_features(data)
    credit_card_features = gen_credit_card_features(data)
    installments_payments_features = gen_installments_payments_features(data)

    merge_df([
      id_df,
      group_feats,
      ext_sources_feats,
      docs_features,
      flags_features,
      bureau_features,
      prev_app_features,
      pos_cash_features,
      credit_card_features,
      installments_payments_features,
    ],
             logger=_logger)

  flow.run()


if __name__ == "__main__":
  app.run(main)
