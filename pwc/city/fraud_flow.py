import random
from typing import Any

from absl import flags
import lightautoml
from lightautoml.automl.base import AutoML
from lightautoml.dataset.roles import CategoryRole
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.dataset.roles import TargetRole
from lightautoml.ml_algo.boost_cb import BoostCB
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.linear_sklearn import LinearLBFGS
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.features.lgb_pipeline import LGBAdvancedPipeline
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
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

from pwc.datasets import SyntheticFraudDetection

FLAGS = flags.FLAGS


@task
def load_data() -> tuple[pd.DataFrame, dict]:
  ds = SyntheticFraudDetection()
  df = ds.as_dataframe("train")
  roles = {"target": "isFraud"}
  return df, roles


@task
def feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
  df["MaxOut"] = df.amount == df.oldbalanceOrg
  df["TransactionType"] = (
    df.nameOrig.apply(lambda x: x[0]) + "->" +
    df.nameDest.apply(lambda x: x[0])
  )
  df["Timestamp"] = df.step.apply(
    lambda s: pd.Timestamp("2021") + pd.Timedelta(f"{s}h")
  )
  df = df.drop(["nameOrig", "nameDest", "type", "step"], axis=1)
  roles = {
    TargetRole(): "isFraud",
    DatetimeRole(seasonality=["wd", "hour"], country="Nigeria"): "Timestamp",
    CategoryRole(str): "TransactionType",
    CategoryRole(np.int32, label_encoded=True): ["MaxOut", "MillionDollar"]
  }
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
    "cv_folds": FLAGS.cv_folds,
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
      "early_stopping_rounds": 25,
      "learning_rate": 0.025,
      "num_leaves": 64,
      "num_threads": config["n_threads"],
      "num_trees": FLAGS.max_trees,
      "random_state": 0,
    }
  )

  lgbm_model1 = BoostLGBM(
    default_params={
      "boosting": "dart",
      # "early_stopping_rounds": 25, Not supported in DART
      "learning_rate": 0.025,
      "num_leaves": 64,
      "num_threads": config["n_threads"],
      "num_trees": FLAGS.max_trees,
      "random_state": 1,
    }
  )

  lgbm_model2 = BoostLGBM(
    default_params={
      "boosting": "rf",
      "early_stopping_rounds": 25,
      "learning_rate": 0.025,
      "num_leaves": 64,
      "num_threads": config["n_threads"],
      "num_trees": FLAGS.max_trees,
      "random_state": 2,
    }
  )

  lgbm_model3 = BoostLGBM(
    default_params={
      "boosting": "goss",
      "bagging_freq": 0,  # GOSS does not support bagging.
      "early_stopping_rounds": 25,
      "learning_rate": 0.025,
      "num_leaves": 64,
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
      (
        cb_model0,
        OptunaTuner(
          n_trials=FLAGS.trials,
          timeout=config["timeout"],
          fit_on_holdout=False
        ),
      ),
      (
        lgbm_model0,
        OptunaTuner(
          n_trials=FLAGS.trials,
          timeout=config["timeout"],
          fit_on_holdout=False
        ),
      ),
      (
        lgbm_model1,
        OptunaTuner(
          n_trials=FLAGS.trials,
          timeout=config["timeout"],
          fit_on_holdout=False
        ),
      ),
      (
        lgbm_model2,
        OptunaTuner(
          n_trials=FLAGS.trials,
          timeout=config["timeout"],
          fit_on_holdout=False
        ),
      ),
      (
        lgbm_model3,
        OptunaTuner(
          n_trials=FLAGS.trials,
          timeout=config["timeout"],
          fit_on_holdout=False
        ),
      ),
    ],
    features_pipeline=LGBSimpleFeatures(),
  )

  linear_pipeline = MLPipeline(
    [LinearLBFGS()],
    features_pipeline=LinearFeatures(),
  )
  return [gbt_pipeline, linear_pipeline]


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
