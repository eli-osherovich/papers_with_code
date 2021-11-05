from absl import flags
import numpy as np
import pandas as pd
import tensorflow as tf

from ...datasets import common as ds_common

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "dataset",
  default="MyDataset",
  help="Dataset to use",
)

flags.DEFINE_string(
  "train_name",
  default="train",
  help="Name of the train config",
)


def get_numpy() -> tuple[np.ndarray, np.ndarray]:
  return ds_common.load_numpy(FLAGS.dataset, FLAGS.train_name)


def get_dataframe() -> tuple[pd.DataFrame, pd.DataFrame]:
  return ds_common.load_dataframe(FLAGS.dataset, FLAGS.train_name)


def get_dataset() -> tf.data.Dataset:
  return ds_common.load_dataset(FLAGS.dataset, FLAGS.train_name)
