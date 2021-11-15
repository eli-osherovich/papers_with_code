"""Common interface to Dataset loading."""
import contextlib

from absl import flags

from ...datasets import common as ds_common

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "dataset",
  default="UCICoverType",
  help="Dataset to use",
)

flags.DEFINE_string(
  "train_name",
  default="train",
  help="Name of the train split",
)

flags.DEFINE_string(
  "val_name",
  default="val",
  help="Name of the validaton split",
)

flags.DEFINE_string(
  "test_name",
  default="test",
  help="Name of the test split",
)


def get_data_numpy():
  res = {}
  res["train"] = ds_common.load_preprocessed_numpy(
    FLAGS.dataset, FLAGS.train_name
  )

  with contextlib.suppress(KeyError):
    res["val"] = ds_common.load_preprocessed_numpy(
      FLAGS.dataset, FLAGS.val_name
    )

  with contextlib.suppress(KeyError):
    res["test"] = ds_common.load_preprocessed_numpy(
      FLAGS.dataset, FLAGS.test_name
    )

  return res
