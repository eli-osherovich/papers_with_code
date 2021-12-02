import random
from typing import Any

import lightautoml
from lightautoml.automl.base import AutoML
from lightautoml.dataset.roles import CategoryRole
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.dataset.roles import TargetRole
from lightautoml.ml_algo.boost_cb import BoostCB
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.linear_sklearn import LinearLBFGS
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.features.lgb_pipeline import LGBAdvancedPipeline, LGBSimpleFeatures
from lightautoml.pipelines.features.linear_pipeline import LinearFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.reader.base import PandasToPandasReader
import numpy as np
import pandas as pd
import prefect
from prefect import Flow
from prefect import task
from prefect.engine.results import LocalResult
import psutil
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import torch

from pwc.datasets import HomeCreditDefaultRisk as DataSet

from absl import flags

FLAGS = flags.FLAGS


@task
def load_data() -> tuple[dict[str, pd.DataFrame], dict]:
  ds = DataSet()
  data = {}
  data["train"] = ds.as_dataframe("train")
  data["previous_application"] = ds.as_dataframe("previous_application")
  data["installments_payments"] = ds.as_dataframe("installments_payments")
  data["bureau"] = ds.as_dataframe("bureau")
  data["bureau_balance"] = ds.as_dataframe("bureau_balance")
  roles = {"target": "TARGET", "drop": "SK_ID_CURR"}
  return data, roles


@task
def feature_engineering(
  data: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, dict]:

  def set_nans(df):
    # Remove the rows without gender
    if "CODE_GENDER" in df.columns:
      df = df[df['CODE_GENDER'] != "XNA"]

    # Replace NaN substitutes with proper NaNs
    df.replace(["XNA", 365243], np.nan, inplace=True)

  def previous_app_counts():
    prev = data["previous_application"]

    # Count (statuses) of previous applications
    return prev.groupby(
      ["SK_ID_CURR"]
    ).NAME_CONTRACT_STATUS.value_counts().unstack(fill_value=0)

  def previous_app_agg():
    prev = data["previous_application"]
    # Add feature: value ask / value received percentage
    prev["APP_CREDIT_PERC"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]
    num_aggregations = {
      "AMT_ANNUITY": ["max", "mean"],
      "AMT_APPLICATION": ["max", "mean"],
      "AMT_CREDIT": ["max", "mean"],
      "APP_CREDIT_PERC": ["max", "mean"],
      "AMT_DOWN_PAYMENT": ["max", "mean"],
      "AMT_GOODS_PRICE": ["max", "mean"],
      "HOUR_APPR_PROCESS_START": ["max", "mean"],
      "RATE_DOWN_PAYMENT": ["max", "mean"],
      "DAYS_DECISION": ["max", "mean"],
      "CNT_PAYMENT": ["mean", "sum"],
    }
    res = prev.groupby("SK_ID_CURR").agg(num_aggregations)
    res.columns = [f"{c[0]}_{c[1]}" for c in res.columns]
    return res

  def gen_population_features(df):
    for feat in [
      "NAME_CONTRACT_TYPE", "CODE_GENDER", "CNT_CHILDREN", "NAME_INCOME_TYPE",
      "NAME_FAMILY_STATUS", "OCCUPATION_TYPE", "CNT_FAM_MEMBERS",
      "ORGANIZATION_TYPE"
    ]:
      new_feat_col = f"NEW_INC_BY_{feat}_MEDIAN"
      new_rel_col = f"NEW_INC_REL_TO_{feat}"
      inc_by_feat = df[["AMT_INCOME_TOTAL",
                        feat]].groupby(feat).median()["AMT_INCOME_TOTAL"]
      df[new_feat_col] = df[feat].map(inc_by_feat)
      df[new_rel_col] = df["AMT_INCOME_TOTAL"] / df[new_feat_col]
    return df

  def gen_new_features(df):
    df["NEW_CREADIT_TO_INCOME_RATIO"
      ] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["NEW_CREDIT_TO_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["NEW_CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["NEW_ANNUITY_TO_INCOME_RATIO"
      ] = df["AMT_ANNUITY"] / (1 + df["AMT_INCOME_TOTAL"])

    df["NEW_INC_PER_CHLD"] = df["AMT_INCOME_TOTAL"] / (1 + df["CNT_CHILDREN"])
    df["NEW_EMPLOY_TO_BIRTH_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]

    df["NEW_SOURCES_PROD"
      ] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["NEW_EXT_SOURCES_MEAN"] = df[[
      "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"
    ]].mean(axis=1)
    df["NEW_SCORES_STD"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2",
                               "EXT_SOURCE_3"]].std(axis=1)
    df["NEW_SCORES_STD"] = df["NEW_SCORES_STD"].fillna(
      df["NEW_SCORES_STD"].mean()
    )

    df["NEW_CAR_TO_BIRTH_RATIO"] = df["OWN_CAR_AGE"] / df["DAYS_BIRTH"]
    df["NEW_CAR_TO_EMPLOY_RATIO"] = df["OWN_CAR_AGE"] / df["DAYS_EMPLOYED"]
    df["NEW_PHONE_TO_BIRTH_RATIO"
      ] = df["DAYS_LAST_PHONE_CHANGE"] / df["DAYS_BIRTH"]
    df["NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER"
      ] = df["DAYS_LAST_PHONE_CHANGE"] / df["DAYS_EMPLOYED"]
    df["NEW_CREDIT_TO_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    return df

  def gen_installment_features():
    ins = data["installments_payments"]

    # Percentage and difference paid in each installment (amount paid and installment value)
    ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]

    # Days past due and days before due (no negative values)
    ins["DPD"] = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
    ins["DBD"] = ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]
    ins["DPD"] = ins["DPD"].apply(lambda x: x if x > 0 else 0)
    ins["DBD"] = ins["DBD"].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
      "NUM_INSTALMENT_VERSION": ["nunique"],
      "DPD": ["max", "mean", "sum", "min", "std"],
      "DBD": ["max", "mean", "sum", "min", "std"],
      "PAYMENT_PERC": ["max", "mean", "var", "min", "std"],
      "PAYMENT_DIFF": ["max", "mean", "var", "min", "std"],
      "AMT_INSTALMENT": ["max", "mean", "sum", "min", "std"],
      "AMT_PAYMENT": ["min", "max", "mean", "sum", "std"],
      "DAYS_ENTRY_PAYMENT": ["max", "mean", "sum", "std"]
    }

    ins_agg = ins.groupby("SK_ID_CURR").agg(aggregations)
    ins_agg.columns = [
      "INSTAL_" + e[0] + "_" + e[1].upper() for e in ins_agg.columns
    ]
    # Count installments accounts
    ins_agg["INSTAL_COUNT"] = ins.groupby("SK_ID_CURR").size()
    return ins_agg

  def gen_bureau_features():
    bureau = data["bureau"]
    bb = data["bureau_balance"]

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {"MONTHS_BALANCE": ["min", "max", "size"]}
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(bb_aggregations)
    bb_agg.columns = [e[0] + "_" + e[1].upper() for e in bb_agg.columns]
    bureau = bureau.join(bb_agg, how="left", on="SK_ID_BUREAU")
    bureau.drop(["SK_ID_BUREAU"], axis=1, inplace=True)

    # Bureau and bureau_balance numeric features
    num_aggregations = {
      "DAYS_CREDIT": ["mean", "var"],
      "DAYS_CREDIT_ENDDATE": ["mean"],
      "DAYS_CREDIT_UPDATE": ["mean"],
      "CREDIT_DAY_OVERDUE": ["mean"],
      "AMT_CREDIT_MAX_OVERDUE": ["mean"],
      "AMT_CREDIT_SUM": ["mean", "sum"],
      "AMT_CREDIT_SUM_DEBT": ["mean", "sum"],
      "AMT_CREDIT_SUM_OVERDUE": ["mean"],
      "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
      "AMT_ANNUITY": ["max", "mean"],
      "CNT_CREDIT_PROLONG": ["sum"],
      "MONTHS_BALANCE_MIN": ["min"],
      "MONTHS_BALANCE_MAX": ["max"],
      "MONTHS_BALANCE_SIZE": ["mean", "sum"]
    }

    bureau_agg = bureau.groupby("SK_ID_CURR").agg(num_aggregations)
    bureau_agg.columns = [
      "BUREAU_" + e[0] + "_" + e[1].upper() for e in bureau_agg.columns
    ]

    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau["CREDIT_ACTIVE"] == "Active"]
    active_agg = active.groupby("SK_ID_CURR").agg(num_aggregations)
    active_agg.columns = pd.Index([
      "ACTIVE_" + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()
    ])
    bureau_agg = bureau_agg.join(active_agg, how="left", on="SK_ID_CURR")

    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau["CREDIT_ACTIVE"] == "Closed"]
    closed_agg = closed.groupby("SK_ID_CURR").agg(num_aggregations)
    closed_agg.columns = pd.Index([
      "CLOSED_" + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()
    ])
    bureau_agg = bureau_agg.join(closed_agg, how="left", on="SK_ID_CURR")

    return bureau_agg

  for df in data.values():
    set_nans(df)

  df = data["train"]
  df = gen_population_features(df)
  df = gen_new_features(df)
  df = df.join(previous_app_counts(), on="SK_ID_CURR", rsuffix="PREV_")
  df = df.join(previous_app_agg(), on="SK_ID_CURR", rsuffix="PREV_")
  df = df.join(
    gen_installment_features(), on="SK_ID_CURR", rsuffix="INSTALLMENT_"
  )
  df = df.join(gen_bureau_features(), on="SK_ID_CURR", rsuffix="BUREAU_")

  # Roles
  roles = {"target": "TARGET"}

  return df, roles


@task
def init(config: dict[str, Any]) -> None:
  logger = prefect.context.get("logger")

  np.random.seed(config["random_state"])
  tf.random.set_seed(config["random_state"])
  torch.manual_seed(config["random_state"])
  random.seed(config["random_state"])
  torch.set_num_threads(config["n_threads"])

  logger.info("Initialized RNG seeds")


@task
def load_config() -> dict[str, Any]:
  logger = prefect.context.get("logger")

  config = {
    "target": "TARGET",
    "n_threads": psutil.cpu_count(logical=True),
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "timeout": 3600,
  }
  logger.info(f"Setting target column to {config['target']}")
  logger.info(f"Setting number of threads to {config['n_threads']}")
  logger.info(f"Setting number of CV folds to {config['cv_folds']}")
  return config


@task
def split_data(df: pd.DataFrame,
               config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
  logger = prefect.context.get("logger")
  train_df, test_df = train_test_split(
    df,
    test_size=config["test_size"],
    random_state=config["random_state"],
    stratify=df[config["target"]]
  )
  logger.info(
    f"Splitted data train/test: {1-config['test_size']}/{config['test_size']}"
  )
  return train_df, test_df


@task
def first_level_pipeline(config) -> list[MLPipeline]:
  lgbm_model0 = BoostLGBM(
    default_params={
      "boosting": "gbdt",
      "early_stopping_rounds": FLAGS.patience,
      "learning_rate": 0.025,
      "num_leaves": FLAGS.num_leaves,
      "num_threads": config["n_threads"],
      "num_trees": FLAGS.max_trees,
      "random_state": 0,
    }
  )

  lgbm_model1 = BoostLGBM(
    default_params={
      "boosting": "dart",
      # "early_stopping_rounds": FLAGS.patience, Not supported in DART
      "learning_rate": 0.025,
      "num_leaves": FLAGS.num_leaves,
      "num_threads": config["n_threads"],
      "num_trees": FLAGS.max_trees,
      "random_state": 1,
    }
  )

  lgbm_model2 = BoostLGBM(
    default_params={
      "boosting": "rf",
      "early_stopping_rounds": FLAGS.patience,
      "learning_rate": 0.025,
      "num_leaves": FLAGS.num_leaves,
      "num_threads": config["n_threads"],
      "num_trees": FLAGS.max_trees,
      "random_state": 2,
    }
  )

  lgbm_model3 = BoostLGBM(
    default_params={
      "boosting": "goss",
      "bagging_freq": 0,  # GOSS does not support bagging.
      "early_stopping_rounds": FLAGS.patience,
      "learning_rate": 0.025,
      "num_leaves": FLAGS.num_leaves,
      "num_threads": config["n_threads"],
      "num_trees": FLAGS.max_trees,
      "random_state": 3,
    }
  )

  cb_model0 = BoostCB(
    default_params={
      "learning_rate": 0.025,
      "num_trees": FLAGS.max_trees,
      "random_state": 4,
      "thread_count": config["n_threads"],
    }
  )

  gbt_pipeline = MLPipeline(
    [
      cb_model0,
      lgbm_model0,
      # (
      #   cb_model0,
      #   OptunaTuner(
      #     n_trials=2, timeout=config["timeout"], fit_on_holdout=False
      #   ),
      # ),
      # (
      #   lgbm_model0,
      #   OptunaTuner(
      #     n_trials=2, timeout=config["timeout"], fit_on_holdout=False
      #   ),
      # ),
      # (
      #   lgbm_model1,
      #   OptunaTuner(
      #     n_trials=2, timeout=config["timeout"], fit_on_holdout=False
      #   ),
      # ),
      # (
      #   lgbm_model2,
      #   OptunaTuner(
      #     n_trials=2, timeout=config["timeout"], fit_on_holdout=False
      #   ),
      # ),
      # (
      #   lgbm_model3,
      #   OptunaTuner(
      #     n_trials=2, timeout=config["timeout"], fit_on_holdout=False
      #   ),
      # ),
    ],
    features_pipeline=LGBAdvancedPipeline(),
  )

  linear_pipeline = MLPipeline(
    [LinearLBFGS()],
    features_pipeline=LinearFeatures(),
  )
  return [
    gbt_pipeline,
    # linear_pipeline,
  ]


@task
def second_level_pipeline(config) -> list[MLPipeline]:
  del config  # not used
  return [MLPipeline([LinearLBFGS()])]


@task
def get_model(pipelines, config):
  task = lightautoml.tasks.Task("binary")
  reader = PandasToPandasReader(
    task, cv=config["cv_folds"], random_state=config["random_state"]
  )

  return AutoML(reader, pipelines)


@task
def train(model, train_df, test_df, roles, config):
  logger = prefect.context.get("logger")

  oof_pred = model.fit_predict(train_df, roles=roles, verbose=3)
  test_pred = model.predict(test_df)

  logger.info(
    f"OOF score: "
    f"{roc_auc_score(train_df[config['target']].values, oof_pred.data[:, 0])}"
  )

  logger.info(
    f"TEST score: "
    f"{roc_auc_score(test_df[config['target']].values, test_pred.data[:, 0])}"
  )


result = LocalResult(dir="/tmp/flow-results")
with Flow("AutoML", result=result) as flow_automl:
  config = load_config()
  init(config)
  df, roles = load_data()
  df, roles = feature_engineering(df)
  train_df, test_df = split_data(df, config)
  level1 = first_level_pipeline(config)
  level2 = second_level_pipeline(config)
  model = get_model([level1, level2], config)
  train(model, train_df, test_df, roles, config)
