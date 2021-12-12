#!/usr/bin/env python

import json
import pathlib

from absl import app
from absl import flags
import joblib
import lightautoml
from lightautoml.automl.base import AutoML
from lightautoml.automl.blend import WeightedBlender
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
import pandas as pd
import prefect
from prefect import Flow
from prefect import task
import numpy as np

from pwc.datasets import common
from pwc.datasets import HomeCreditDefaultRisk as Dataset

FLAGS = flags.FLAGS

flags.DEFINE_string("data_type", "installments_payments", "Data part")
flags.DEFINE_string("features_type", "autofeat_ts", "Features type")
flags.DEFINE_integer("n", 100, "Number of files to load")
flags.DEFINE_bool(
  "combine_features", False, "Combine with the original application features"
)
flags.DEFINE_float("corr_thresh", 0.98, "Correlation threshold", lower_bound=0)


@task
def load_features_data() -> pd.DataFrame:
  dfs = []
  for i in range(FLAGS.n):
    f_name = pathlib.Path(
      __file__
    ).parent / f"data/{FLAGS.data_type}_{FLAGS.features_type}_{i}.feather"
    df = pd.read_feather(f_name)
    dfs.append(df)
  return pd.concat(dfs, ignore_index=True)





@task
def get_train_data(ts_data: pd.DataFrame) -> pd.DataFrame:
  ds = Dataset()
  application_data = ds.as_dataframe("train")
  if not FLAGS.combine_features:
    application_data = application_data[["TARGET", "SK_ID_CURR"]]

  common.pandas_downcast(application_data, inplace=True)
  res = application_data.merge(ts_data, on="SK_ID_CURR")
  return res.drop("SK_ID_CURR", axis=1)


@task
def train(train_df):
  task = lightautoml.tasks.Task("binary")
  reader = PandasToPandasReader(task, cv=5, random_state=13)
  roles = {"target": "TARGET"}

  model = BoostLGBM(
    default_params={
      "boosting": "gbdt",
      "early_stopping_rounds": 50,
      "num_threads": 8,
      "random_state": 0,
    }
  )
  pipeline = MLPipeline([model], features_pipeline=LGBSimpleFeatures())
  automl = AutoML(reader, [[pipeline]])
  automl.fit_predict(train_df, roles=roles, verbose=2)
  return automl


@task
def save_model(model) -> None:
  model_path = pathlib.Path(
    __file__
  ).parent / f"models/{FLAGS.data_type}_{FLAGS.features_type}_model.pkl"

  features_path = pathlib.Path(
    __file__
  ).parent / f"features/{FLAGS.data_type}_{FLAGS.features_type}_features_importance.pkl"

  joblib.dump(model, model_path)
  joblib.dump(
    model.levels[0][0].ml_algos[0].get_features_score(), features_path
  )


def main(argv):
  del argv  # unused
  with Flow("main") as flow:
    features_df = load_features_data()
    train_df = get_train_data(features_df)
    model = train(train_df)
    save_model(model)
  flow.run()


if __name__ == "__main__":
  app.run(main)
