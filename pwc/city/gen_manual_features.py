#!/usr/bin/env python

import pathlib

from absl import app
import pandas as pd
from prefect import Flow
from prefect import task
from prefect.utilities import logging

from pwc.city import task_lib
from pwc.datasets import HomeCreditDefaultRisk as Dataset
from pwc.datasets import utils

_logger = logging.get_logger()
_id = "SK_ID_CURR"

set_nans = task(task_lib.set_nans)


def get_age_label(days_birth):
  """ Return the age group label (int). """
  age_years = -days_birth / 365
  if age_years < 27: return 1
  elif age_years < 40: return 2
  elif age_years < 50: return 3
  elif age_years < 65: return 4
  elif age_years < 99: return 5
  else: return 0


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
def gen_ratio_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
  df = data["train"]
  res = df[[_id]].copy()

  def ratio(target: str, num: str, denom: str):
    if 0 in df[denom]:
      ratio = df[num] / (1 + df[denom])
    else:
      ratio = df[num] / df[denom]
    res[target] = ratio

  ratio("manual__CREDIT_TO_ANNUITY_RATIO", "AMT_CREDIT", "AMT_ANNUITY")
  ratio("manual__CREDIT_TO_GOODS_RATIO", "AMT_CREDIT", "AMT_GOODS_PRICE")
  ratio("manual__CREDIT_TO_INCOME_RATIO", "AMT_CREDIT", "AMT_INCOME_TOTAL")

  ratio("manual__ANNUITY_TO_INCOME_RATIO", "AMT_ANNUITY", "AMT_INCOME_TOTAL")
  ratio("manual__ANNUITY_TO_GOODS_RATIO", "AMT_ANNUITY", "AMT_GOODS_PRICE")

  ratio("manual__INCOME_TO_EMPLOYED_RATIO", "AMT_INCOME_TOTAL", "DAYS_EMPLOYED")
  ratio("manual__INCOME_TO_BIRTH_RATIO", "AMT_INCOME_TOTAL", "DAYS_BIRTH")
  ratio("manual__EMPLOYED_TO_BIRTH_RATIO", "DAYS_EMPLOYED", "DAYS_BIRTH")
  ratio("manual__ID_TO_BIRTH_RATIO", "DAYS_ID_PUBLISH", "DAYS_BIRTH")
  ratio("manual__CAR_TO_BIRTH_RATIO", "OWN_CAR_AGE", "DAYS_BIRTH")
  ratio("manual__CAR_TO_EMPLOYED_RATIO", "OWN_CAR_AGE", "DAYS_EMPLOYED")
  ratio("manual__PHONE_TO_BIRTH_RATIO", "DAYS_LAST_PHONE_CHANGE", "DAYS_BIRTH")
  return utils.pandas_downcast(res)


@task
def gen_group_features(
  data: dict[str, pd.DataFrame], ext_source_df: pd.DataFrame,
  ratio_df: pd.DataFrame
) -> pd.DataFrame:
  df = pd.merge(data["train"], ext_source_df, on=_id)
  df = pd.merge(df, ratio_df, on=_id)
  df["AGE_GROUP"] = df.DAYS_BIRTH.apply(get_age_label)
  res = df[[_id]].copy()

  groups = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "CNT_CHILDREN", "NAME_INCOME_TYPE",
    "NAME_FAMILY_STATUS", "OCCUPATION_TYPE", "CNT_FAM_MEMBERS",
    "ORGANIZATION_TYPE", "NAME_EDUCATION_TYPE", "AGE_GROUP"
  ]
  features = [
    "manual__EXT_SOURCES_min", "manual__EXT_SOURCES_max",
    "manual__EXT_SOURCES_mean", "manual__EXT_SOURCES_median",
    "manual__CREDIT_TO_ANNUITY_RATIO", "manual__ANNUITY_TO_GOODS_RATIO",
    "AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_CREDIT", "AMT_GOODS_PRICE"
  ]

  for group in groups:
    for feat in features:
      for func in ["median", "std"]:
        new_func_col = f"manual__{feat}_{group}_{func}"
        group_func = df[[feat, group]].groupby(group).agg(func)[feat]
        res[new_func_col] = df[group].map(group_func)
      new_rel_col = f"manual__{feat}_REL_TO_{group}_median"
      res[new_rel_col] = df[feat] / res[f"manual__{feat}_{group}_median"]
  return utils.pandas_downcast(res)


@task
def gen_docs_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
  df = data["train"]
  res = df[[_id]].copy()
  cols = [f for f in df.columns if "FLAG_DOC" in f]
  for func in ["min", "max", "mean", "median", "std", "kurtosis"]:
    res[f"manual__ALL_DOCS_{func}"] = df[cols].agg(func, axis=1)

  return utils.pandas_downcast(res)


@task
def gen_flag_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
  df = data["train"]
  res = df[[_id]].copy()
  cols = [
    f for f in df.columns
    if ("FLAG_" in f) & ("FLAG_DOC" not in f) & ("_FLAG_" not in f)
  ]
  for func in ["min", "max", "mean", "median", "std", "kurtosis"]:
    res[f"manual__OTHER_FLAGS_{func}"] = df[cols].agg(func, axis=1)

  return utils.pandas_downcast(res)


@task
def gen_ext_sources_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
  df = data["train"]
  res = df[[_id]].copy()
  cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
  for func in ["min", "max", "mean", "median", "std"]:
    res[f"manual__EXT_SOURCES_{func}"] = df[cols].agg(func, axis=1)

  res["manual__EXT_SOURCES_mul"] = (
    df.EXT_SOURCE_1.fillna(1) * df.EXT_SOURCE_2.fillna(1) *
    df.EXT_SOURCE_3.fillna(1)
  )

  res["manual__EXT_SOURCES_weighted_sum"] = (
    df.EXT_SOURCE_1.fillna(1) * 2 + df.EXT_SOURCE_2.fillna(1) * 1 +
    df.EXT_SOURCE_3.fillna(1) * 3
  )

  res["manual__EXT_SOURCES_count"] = df[cols].count(axis=1)

  return utils.pandas_downcast(res)


@task
def gen_bureau_balance_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
  df = data["bureau_balance"]
  df = pd.get_dummies(df, dummy_na=True)
  gb = df.groupby("SK_ID_BUREAU")
  res = gb.agg(["min", "max", "mean"])
  res.columns = task_lib.flatten_col_names(res, prefix="manual__BUREAU_BALANCE")
  res["manual__BUREAU_BALANCE_size"] = gb.size()
  return utils.pandas_downcast(res)


@task
def gen_bureau_features(
  data: dict[str, pd.DataFrame], bb_df: pd.DataFrame
) -> pd.DataFrame:
  df = data["bureau"].copy()
  df["manual__CREDIT_DURATION"] = df["DAYS_CREDIT_ENDDATE"] - df["DAYS_CREDIT"]
  df["manual__ENDDATE_DIF"] = (
    df["DAYS_CREDIT_ENDDATE"] - df["DAYS_ENDDATE_FACT"]
  )
  # Credit to debt ratio and difference
  df["manual__DEBT_PERCENTAGE"] = (
    df["AMT_CREDIT_SUM"] / df["AMT_CREDIT_SUM_DEBT"]
  )
  df["manual__DEBT_CREDIT_DIFF"] = (
    df["AMT_CREDIT_SUM"] - df["AMT_CREDIT_SUM_DEBT"]
  )
  df["manual__CREDIT_TO_ANNUITY_RATIO"] = (
    df["AMT_CREDIT_SUM"] / df["AMT_ANNUITY"]
  )

  dummies = pd.get_dummies(df, dummy_na=True)
  dummies = dummies.join(bb_df, on="SK_ID_BUREAU")
  dummies = dummies.drop(["SK_ID_BUREAU"], axis=1)

  gb = dummies.groupby(_id)
  res = gb.agg(["min", "max", "mean"])
  res.columns = task_lib.flatten_col_names(res, prefix="manual__BUREAU")
  res["manual__BUREAU_size"] = gb.size()

  for ca in df["CREDIT_ACTIVE"].unique():
    # Aggregations for different credit active statuses.
    gb = df[df["CREDIT_ACTIVE"] == ca].groupby(_id)
    agg = gb.agg(["min", "max", "mean"])
    agg.columns = task_lib.flatten_col_names(
      agg, prefix=f"manual__BUREAU_CREDIT_ACTIVE_{ca}"
    )
    agg[f"manual__BUREAU_CREDIT_ACTIVE_{ca}_size"] = gb.size()
    res = res.join(agg, on=_id)
  return utils.pandas_downcast(res)


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
  agg = df.select_dtypes("number").groupby(_id).agg(["min", "max", "var"])
  agg.columns = task_lib.flatten_col_names(agg, prefix="manual__PREVAPP")
  res = res.join(agg, on=_id)

  for status in df["NAME_CONTRACT_STATUS"].unique():
    # Aggregations for different credit active statuses.
    agg = df[df["NAME_CONTRACT_STATUS"] == status].groupby(_id).agg([
      "min", "max", "mean", "var"
    ])
    agg.columns = task_lib.flatten_col_names(
      agg, prefix=f"manual__PREVAPP_NAME_CONTRACT_{status}"
    )
    res = res.join(agg, on=_id)

  return utils.pandas_downcast(res)


@task
def gen_pos_cash_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
  df = data["pos_cash_balance"]
  res = pd.get_dummies(df)
  aggregations = ["min", "max", "mean", "var"]
  gb = res.groupby(_id)
  res = gb.agg(aggregations)
  res.columns = task_lib.flatten_col_names(res, prefix="manual__POS")
  # Count pos cash accounts
  res["manual__POS_COUNT"] = gb.size()
  return utils.pandas_downcast(res)


@task
def gen_credit_card_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
  df = data["credit_card_balance"]
  df = df.drop("SK_ID_PREV", axis=1)
  gb = df.groupby(_id)
  res = gb.agg(["min", "max", "mean", "var"])
  res.columns = task_lib.flatten_col_names(res, prefix="manual__CC")
  # Count credit card lines
  res["manual__CC_COUNT"] = gb.size()
  return utils.pandas_downcast(res)


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
  return utils.pandas_downcast(res)


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
    ext_sources_feats = gen_ext_sources_features(data)
    ratio_feats = gen_ratio_features(data)
    group_feats = gen_group_features(data, ext_sources_feats, ratio_feats)
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
      ext_sources_feats,
      ratio_feats,
      group_feats,
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
