"""Common interface to Dataset loading."""
from absl import flags

from ...datasets import common as ds_common
from ...datasets import dataset

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "dataset",
  default="UCICoverType",
  help="Dataset to use",
)

flags.DEFINE_string(
  "train_name",
  default="train",
  help="Name of the train config",
)


def get_numpy() -> dataset.NP_RESULT:
  return ds_common.load_numpy(FLAGS.dataset, FLAGS.train_name)


def get_dataframe() -> dataset.DF_RESULT:
  return ds_common.load_dataframe(FLAGS.dataset, FLAGS.train_name)


def get_dataset() -> dataset.DS_RESULT:
  return ds_common.load_dataset(FLAGS.dataset, FLAGS.train_name)
