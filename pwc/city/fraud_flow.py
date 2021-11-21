import random
from typing import Any

import lightautoml
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.automl.presets.tabular_presets import TabularUtilizedAutoML
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

from pwc.datasets import SyntheticFraudDetection


@task
def load_data() -> tuple[pd.DataFrame, dict]:
  ds = SyntheticFraudDetection()
  df = ds.as_dataframe("train")
  roles = {"target": "isFraud"}
  return df, roles


@task
def feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
  df["MaxOut"] = df.amount == df.oldbalanceOrg
  df["Overdraft"] = df.amount > df.oldbalanceOrg
  df["TransactionType"] = (
    df.nameOrig.apply(lambda x: x[0]) + df.nameDest.apply(lambda x: x[0])
  )
  df["LuckyNumber"] = df.amount == 10000000
  df["Timestamp"] = df.step.apply(
    lambda s: pd.Timestamp("2021") + pd.Timedelta(f"{s}h")
  )
  df = df.drop([
    "nameOrig", "nameDest", "oldbalanceOrg", "newbalanceOrig", "type", "step"
  ],
               axis=1)

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
    "target": "isFraud",
    "n_threads": psutil.cpu_count(logical=False),
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
def automl(train_df, test_df, roles, config):
  logger = prefect.context.get("logger")

  model = TabularAutoML(
    task=lightautoml.tasks.Task("binary"),
    timeout=config["timeout"],
    general_params={
      "nested_cv":
        False,
      "use_algos": [
        ["lgb_tuned", "cb_tuned"],
        ["linear_l2", "lgb_tuned", "cb_tuned"],
      ]
    },
    reader_params={
      "cv": config["cv_folds"],
      "random_state": config["random_state"]
    },
    tuning_params={
      "max_tuning_iter": 20,
      "max_tuning_time": 30
    },
    lgb_params={
      "default_params": {
        "num_threads": config["n_threads"],
        "early_stopping_rounds": 20,
      }
    },
  )

  oof_pred = model.fit_predict(train_df, roles=roles, verbose=3)
  test_pred = model.predict(test_df)
  logger.info(
    f"OOF score: {roc_auc_score(train_df[config['target']], oof_pred.data[:, 0])}"
  )
  logger.info(
    f"TEST score: {roc_auc_score(test_df[config['target']], test_pred.data[:, 0])}"
  )


result = LocalResult(dir="/tmp/flow-results")
with Flow("AutoML", result=result) as flow_automl:
  config = load_config()
  init(config)
  df, roles = load_data()
  train_df, test_df = split_data(df, config)
  automl(train_df, test_df, roles, config)
flow_automl.run()
