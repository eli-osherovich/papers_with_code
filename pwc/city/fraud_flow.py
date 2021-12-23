#!/usr/bin/env python

import random
from typing import Any

from absl import app
from absl import flags
from lightautoml.dataset.roles import CategoryRole
from lightautoml.dataset.roles import TargetRole
from lightautoml.ml_algo.boost_cb import BoostCB
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.linear_sklearn import LinearLBFGS
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.features.lgb_pipeline import LGBAdvancedPipeline
from lightautoml.pipelines.features.linear_pipeline import LinearFeatures
from lightautoml.pipelines.ml.base import MLPipeline
import numpy as np
import pandas as pd
import prefect
from prefect import Flow
from prefect import task
from prefect.utilities import logging
import psutil
import torch

from pwc.city import task_lib
from pwc.datasets import SyntheticFraudDetection

FLAGS = flags.FLAGS

flags.DEFINE_bool("save", True, "Save results")

flags.DEFINE_float("lr", 0.025, "Learning rate", lower_bound=1e-10)
flags.DEFINE_integer("cv_folds", 5, "Number of tuning trials", lower_bound=2)
flags.DEFINE_integer(
  "max_trees", 5000, "Maximal number of trees for boosters", lower_bound=1
)
flags.DEFINE_integer(
  "num_leaves", 128, "Number of tuning trials", lower_bound=2
)

flags.DEFINE_integer("patience", 50, "Patience during early stopping")
flags.DEFINE_integer(
  "timeout", 3600 * 12, "Trials timeout (seconds)", lower_bound=1
)
flags.DEFINE_integer("trials", 150, "Number of tuning trials", lower_bound=1)
flags.DEFINE_float("test_size", 0.1, "Test size")

_logger = logging.get_logger()
_target = "isFraud"


def task(func):
  return func


split_data = task(task_lib.split_data)
train = task(task_lib.train)
get_automl = task(task_lib.get_automl)


@task
def load_data() -> pd.DataFrame:
  ds = SyntheticFraudDetection()
  return ds.as_dataframe("train")


@task
def feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
  df["WithdrawAll"] = df.amount == df.oldbalanceOrg
  df["OneMillion"] = df.amount == 1_000_000
  df["TransactionType"] = (
    df.nameOrig.apply(lambda x: x[0]) + "->" +
    df.nameDest.apply(lambda x: x[0])
  )
  # df["Timestamp"] = df.step.apply(
  #   lambda s: pd.Timestamp("2021") + pd.Timedelta(f"{s}h")
  # )
  df = df.drop(["nameOrig", "nameDest", "type", "step"], axis=1)
  roles = {
    TargetRole():
      _target,
    # DatetimeRole(seasonality=["wd", "hour"], country="Nigeria"): "Timestamp",
    CategoryRole(str):
      "TransactionType",
    CategoryRole(bool, label_encoded=True): ["WithdrawAll", "OneMillion"]
  }
  return df, roles


@task
def load_config(*, logger=None) -> dict[str, Any]:
  logger = prefect.context.get("logger")

  config = {
    "target": _target,
    "n_threads": psutil.cpu_count(logical=True),
    "test_size": FLAGS.test_size,
    "random_state": 42,
    "cv_folds": FLAGS.cv_folds,
    "timeout": FLAGS.timeout,
  }
  if logger:
    logger.info(f"Setting target column to {config['target']}")
    logger.info(f"Setting number of threads to {config['n_threads']}")
    logger.info(f"Setting number of CV folds to {config['cv_folds']}")
  return config


@task
def init(config: dict[str, Any], *, logger=None) -> None:
  np.random.seed(config["random_state"])
  torch.manual_seed(config["random_state"])
  random.seed(config["random_state"])
  torch.set_num_threads(config["n_threads"])
  if logger:
    logger.info("Initialized RNG seeds")


@task
def first_level_pipeline(config) -> list[MLPipeline]:
  gbt_model = BoostLGBM(
    name="LGBM_GBDT",
    default_params={
      "boosting": "gbdt",
      "early_stopping_rounds": FLAGS.patience,
      "learning_rate": FLAGS.lr,
      "num_leaves": FLAGS.num_leaves,
      "num_threads": config["n_threads"],
      "num_trees": FLAGS.max_trees,
      "random_state": 0,
    }
  )

  dart_model = BoostLGBM(
    name="LGBM_DART",
    default_params={
      "boosting": "dart",
      # "early_stopping_rounds": FLAGS.patience, Not supported in DART
      "learning_rate": FLAGS.lr,
      "num_leaves": FLAGS.num_leaves,
      "num_threads": config["n_threads"],
      "num_trees": FLAGS.max_trees,
      "random_state": 1,
    }
  )

  rf_model = BoostLGBM(
    name="LGBM_RF",
    default_params={
      "boosting": "rf",
      "early_stopping_rounds": FLAGS.patience,
      "learning_rate": FLAGS.lr,
      "num_leaves": FLAGS.num_leaves,
      "num_threads": config["n_threads"],
      "num_trees": FLAGS.max_trees,
      "random_state": 2,
    }
  )

  goss_model = BoostLGBM(
    name="LGBM_GOSS",
    default_params={
      "boosting": "goss",
      "bagging_freq": 0,  # GOSS does not support bagging.
      "early_stopping_rounds": FLAGS.patience,
      "learning_rate": FLAGS.lr,
      "num_leaves": FLAGS.num_leaves,
      "num_threads": config["n_threads"],
      "num_trees": FLAGS.max_trees,
      "random_state": 3,
    }
  )

  cb_model = BoostCB(
    name="CB",
    default_params={
      "learning_rate": FLAGS.lr,
      "num_trees": FLAGS.max_trees,
      "random_state": 4,
      "thread_count": config["n_threads"],
    }
  )

  gbt_pipeline = MLPipeline(
    [
      (
        gbt_model,
        OptunaTuner(
          n_trials=FLAGS.trials,
          timeout=config["timeout"],
          fit_on_holdout=False
        ),
      ),
      (
        cb_model,
        OptunaTuner(
          n_trials=FLAGS.trials,
          timeout=config["timeout"],
          fit_on_holdout=False
        ),
      ),
      # (
      #   dart_model  # Too slow to run tuning
      #   # OptunaTuner(
      #   #   n_trials=FLAGS.trials,
      #   #   timeout=config["timeout"],
      #   #   fit_on_holdout=False
      #   # ),
      # ),
      (
        rf_model,
        OptunaTuner(
          n_trials=FLAGS.trials,
          timeout=config["timeout"],
          fit_on_holdout=False
        ),
      ),
      (
        goss_model,
        OptunaTuner(
          n_trials=FLAGS.trials,
          timeout=config["timeout"],
          fit_on_holdout=False
        ),
      ),
    ],
    features_pipeline=LGBAdvancedPipeline(),
  )

  linear_pipeline = MLPipeline(
    [LinearLBFGS()],
    features_pipeline=LinearFeatures(),
  )
  return [gbt_pipeline, linear_pipeline]


def main(argv):
  del argv  # unused
  with Flow("AutoML") as flow:
    config = load_config()
    init(config)
    df = load_data()
    df, roles = feature_engineering(df)
    train_df, test_df = split_data(df, config)
    level1 = first_level_pipeline(config)
    automl = get_automl([level1], config)
    train(automl, train_df, test_df, roles, config, logger=_logger)
  flow.run()


if __name__ == "__main__":
  app.run(main)
