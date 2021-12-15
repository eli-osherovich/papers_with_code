#!/usr/bin/env python

from absl import app
from absl import flags
import lightautoml.tasks
from lightautoml.automl.base import AutoML
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.reader.base import PandasToPandasReader
import pandas as pd
from prefect import Flow
from prefect import task
from prefect.utilities import logging
from sklearn.metrics import roc_auc_score

from pwc.city import task_lib
from pwc.datasets import HomeCreditDefaultRisk as Dataset

FLAGS = flags.FLAGS

flags.DEFINE_bool(
  "combine_features", False, "Combine with the original application features"
)
flags.DEFINE_bool("save", True, "Save results")

flags.DEFINE_float("corr_thresh", 0.98, "Correlation threshold", lower_bound=0)
flags.DEFINE_float("imp_thresh", 0.02, "Importance threshold", lower_bound=0)
flags.DEFINE_float("null_thresh", 0.6, "NULLity threshold", lower_bound=0)

_logger = logging.get_logger()

remove_null_columns = task(task_lib.remove_null_columns)
remove_high_corr = task(task_lib.remove_high_corr)
load_features_data = task(task_lib.load_features_data)
save_model = task(task_lib.save_model)
get_important_features = task(task_lib.get_important_features)


@task
def load_targets() -> pd.DataFrame:
  ds = Dataset()
  df = ds.as_dataframe("train")
  return df[["SK_ID_CURR", "TARGET"]]


@task
def merge_features(
  cur_df: pd.DataFrame,
  features_data: pd.DataFrame,
  logger=None
) -> pd.DataFrame:

  features_data = features_data.drop("TARGET", axis=1, errors="ignore")
  res = cur_df.merge(features_data, how="left", on="SK_ID_CURR")

  if logger:
    logger.info(f"Merged dataframe shape: {res.shape}")
  return res


@task
def train(train_df: pd.DataFrame, *, logger=None):
  task = lightautoml.tasks.Task("binary")
  reader = PandasToPandasReader(task, cv=5, random_state=13)
  roles = {"target": "TARGET", "drop": "SK_ID_CURR"}

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
  oof_pred = automl.fit_predict(train_df, roles=roles, verbose=2)

  if logger:
    logger.info(
      f"OOF score: {roc_auc_score(train_df['TARGET'].to_numpy(), oof_pred.data[:, 0])}"
    )

  return automl


def main(argv):
  del argv  # unused

  with Flow("main") as flow:
    all_feats = zip(
      [
        "train", "train", "train", "credit_card_balance",
        "installments_payments", "pos_cache_balance"
      ],
      [
        None, "autofeat_num", "autofeat_bool", "autofeat_ts", "autofeat_ts",
        "autofeat_ts"
      ],
    )
    train_df = load_targets()
    for data_type, features_type in all_feats:
      imp_features = get_important_features(
        data_type, features_type, FLAGS.imp_thresh, logger=_logger
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
    model = train(train_df, logger=_logger)
    # if FLAGS.save:
    #   save_model(model, FLAGS.data_type, FLAGS.features_type)
  flow.run()


if __name__ == "__main__":
  app.run(main)
