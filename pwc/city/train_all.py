#!/usr/bin/env python

from datetime import datetime
import pathlib
import random
from typing import Any

from absl import app
from absl import flags
import joblib
from lightautoml.automl.base import AutoML
from lightautoml.automl.blend import WeightedBlender
from lightautoml.ml_algo.boost_cb import BoostCB
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.linear_sklearn import LinearLBFGS
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.features.lgb_pipeline import LGBAdvancedPipeline, LGBSimpleFeatures
from lightautoml.pipelines.features.linear_pipeline import LinearFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.reader.base import PandasToPandasReader
import lightautoml.tasks
import numpy as np
import pandas as pd
from prefect import Flow
from prefect import task
from prefect.utilities import logging
import psutil
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch

from pwc.city import task_lib
from pwc.datasets import HomeCreditDefaultRisk as Dataset

FLAGS = flags.FLAGS

flags.DEFINE_bool(
  "combine_features", False, "Combine with the original application features"
)
flags.DEFINE_bool("save", True, "Save results")

flags.DEFINE_float("corr_thresh", 0.98, "Correlation threshold", lower_bound=0)
flags.DEFINE_float("imp_thresh", 0.005, "Importance threshold", lower_bound=0)
flags.DEFINE_float(
  "imp_thresh_autofeat_ts", 0.05, "Importance threshold", lower_bound=0
)
flags.DEFINE_float("null_thresh", 0.6, "NULLity threshold", lower_bound=0)
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

_logger = logging.get_logger()

_id = "SK_ID_CURR"
_target = "TARGET"


def task(func):
  return func


remove_null_columns = task(task_lib.remove_null_columns)
remove_high_corr = task(task_lib.remove_high_corr)
load_features_data = task(task_lib.load_features_data)
save_model = task(task_lib.save_model)
get_important_features = task(task_lib.get_important_features)
set_nans = task(task_lib.set_nans)


@task
def load_targets() -> pd.DataFrame:
  ds = Dataset()
  df = ds.as_dataframe("train")
  return df[[_id, _target]]


@task
def merge_features(
  cur_df: pd.DataFrame,
  features_data: pd.DataFrame,
  logger=None
) -> pd.DataFrame:

  features_data = features_data.drop(_target, axis=1, errors="ignore")

  # In case of column overlap, use those from cur_df
  cols_to_use = features_data.columns.difference(cur_df.columns).to_list()

  # _id can be either a column or idex (e.g., for aggregated features)
  if _id in features_data:
    cols_to_use.append(_id)
  res = cur_df.merge(
    features_data[cols_to_use],
    how="left",
    on=_id,
    suffixes=("", ""),  # make sure no columns are duplicated.
    validate="one_to_one"
  )

  if logger:
    logger.info(
      f"Merged dataframes: {cur_df.shape} + {features_data.shape} -> {res.shape}"
    )
  return res


@task
def save_feather(df: pd.DataFrame, path: str) -> None:
  df.reset_index().to_feather(path)


@task
def train(model, train_df, test_df, config, *, logger=None):
  roles = {"target": _target}
  oof_pred = model.fit_predict(train_df, roles=roles, verbose=2)
  test_pred = model.predict(test_df)

  if logger:
    logger.info(
      f"OOF score: "
      f"{roc_auc_score(train_df[config['target']].values, oof_pred.data[:, 0])}"
    )
    logger.info(
      f"TEST score: "
      f"{roc_auc_score(test_df[config['target']].values, test_pred.data[:, 0])}"
    )
  timestamp = datetime.now().isoformat(timespec='minutes')
  model_dir = pathlib.Path(__file__).parent / "models"
  joblib.dump(model, model_dir / f"{timestamp}.pkl")


@task
def init(config: dict[str, Any]) -> None:
  np.random.seed(config["random_state"])
  torch.manual_seed(config["random_state"])
  random.seed(config["random_state"])
  torch.set_num_threads(config["n_threads"])


@task
def load_config(*, logger=None) -> dict[str, Any]:
  config = {
    "target": _target,
    "n_threads": psutil.cpu_count(logical=True),
    "test_size": 0.1,
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
def first_level_pipeline(config) -> list[MLPipeline]:
  kaggle_model = BoostLGBM(
    name="KModel",
    default_params={
      "boosting": "gbdt",
      "early_stopping_rounds": 200,
      "num_threads": config["n_threads"],
      "random_state": 0,
      "learning_rate": 0.02,  # 02,
      "num_leaves": 20,
      "colsample_bytree": 0.9497036,
      "subsample": 0.8715623,
      "subsample_freq": 1,
      "max_depth": 8,
      "reg_alpha": 0.041545473,
      "reg_lambda": 0.0735294,
      "min_split_gain": 0.0222415,
      "min_child_weight": 60,  # 39.3259775,
    }
  )
  gbt_model = BoostLGBM(
    name="LGBM",
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
    name="DART",
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
    name="RF",
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
    name="KGOSS",
    default_params={
      "boosting": "goss",
      "n_estimators": 10000,
      "bagging_freq": 0,  # GOSS does not support bagging.
      "num_threads": config["n_threads"],
      "random_state": 737851,
      "learning_rate": 0.005134,
      "num_leaves": 54,
      "max_depth": 10,
      "subsample_for_bin": 240000,
      "reg_alpha": 0.436193,
      "reg_lambda": 0.479169,
      "colsample_bytree": 0.508716,
      "min_split_gain": 0.024766,
      "subsample": 1,
      "is_unbalance": False,
      "early_stopping_rounds": 100,
    }
  )

  cb_model = BoostCB(
    default_params={
      "learning_rate": FLAGS.lr,
      "num_trees": FLAGS.max_trees,
      "random_state": 4,
      "thread_count": config["n_threads"],
    }
  )

  gbt_pipeline = MLPipeline(
    [
      kaggle_model,
      goss_model,
      dart_model,
      gbt_model,
      # (
      #   gbt_model,
      #   OptunaTuner(
      #     n_trials=FLAGS.trials,
      #     timeout=config["timeout"],
      #     fit_on_holdout=False
      #   )
      # ),
      cb_model,
      rf_model,
    ],
    features_pipeline=LGBAdvancedPipeline(),
  )

  linear_pipeline = MLPipeline(
    [LinearLBFGS()],
    features_pipeline=LinearFeatures(),
  )
  return [
    gbt_pipeline,
    linear_pipeline,
  ]


@task
def get_model(pipelines, config):
  task = lightautoml.tasks.Task("binary")
  reader = PandasToPandasReader(
    task, cv=config["cv_folds"], random_state=config["random_state"]
  )

  return AutoML(reader, pipelines, blender=WeightedBlender())


@task
def split_data(df: pd.DataFrame,
               config: dict[str, Any],
               *,
               logger=None) -> tuple[pd.DataFrame, pd.DataFrame]:
  train_df, test_df = train_test_split(
    df,
    test_size=config["test_size"],
    random_state=config["random_state"],
    stratify=df[config["target"]]
  )
  if logger:
    logger.info(
      f"Splitted data train/test: {1-config['test_size']}/{config['test_size']}"
    )
  return train_df, test_df


def main(argv):
  del argv  # unused

  with Flow("main") as flow:
    config = load_config()
    init(config)
    all_feats = [
      ("train", None),
      ("manual", None),
      ("manual_kaggle", None),
      ("train", "autofeat_num"),
      ("train", "autofeat_bool"),
      ("credit_card_balance", "autofeat_ts"),
      ("installments_payments", "autofeat_ts"),
      ("pos_cache_balance", "autofeat_ts"),
    ]
    train_df = load_targets()
    for data_type, features_type in all_feats:
      if features_type == "autofeat_ts":
        imp_thresh = FLAGS.imp_thresh_autofeat_ts
      else:
        imp_thresh = FLAGS.imp_thresh
      imp_features = get_important_features(
        data_type, features_type, imp_thresh, logger=_logger
      )
      features_df = load_features_data(
        data_type, features_type, cols=imp_features, logger=_logger
      )
      features_df = remove_null_columns(
        features_df, FLAGS.null_thresh, logger=_logger
      )
      features_df = remove_high_corr(
        features_df, FLAGS.corr_thresh, logger=_logger
      )
      train_df = merge_features(train_df, features_df, logger=_logger)
    save_feather(train_df, "/tmp/df_before_nans.feather")
    train_df = set_nans(train_df)
    save_feather(train_df, "/tmp/df_after_nans.feather")
    train_df = remove_high_corr(train_df, FLAGS.corr_thresh, logger=_logger)
    save_feather(train_df, "/tmp/df_after_corr.feather")
    train_df, test_df = split_data(train_df, config)
    level1 = first_level_pipeline(config)
    # level2 = second_level_pipeline(config)
    model = get_model([level1], config)
    train(model, train_df, test_df, config, logger=_logger)

  flow.run()


if __name__ == "__main__":
  app.run(main)
