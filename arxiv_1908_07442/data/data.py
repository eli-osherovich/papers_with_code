from absl import flags

from ...datasets import common as ds_common

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "dataset",
  default="KddCup99",
  help="Dataset to use",
)

flags.DEFINE_string(
  "train_name",
  default="train",
  help="Name of the train config",
)


def get_numpy():
  return ds_common.load_numpy(FLAGS.dataset, FLAGS.train_name)


def get_dataframe():
  return ds_common.load_dataframe(FLAGS.dataset, FLAGS.train_name)
