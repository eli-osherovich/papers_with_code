"""Common interface to model training and tuning.
"""
import enum
import importlib
import sys

from absl import flags

from .. import data
from .. import models  # noqa F401 pylint: disable=unused-import

FLAGS = flags.FLAGS


@enum.unique
class TRAIN_MODE(enum.Enum):
  TRAIN = 1
  TRAIN_CV = 2
  TUNE = 3


flags.DEFINE_enum_class(
  "train_mode", TRAIN_MODE.TRAIN, TRAIN_MODE, "Training mode"
)


def _get_model_training_module():
  module_name = FLAGS.model.name.lower() + "_training"
  training_module = importlib.import_module(
    "." + module_name, sys.modules[__package__].__name__
  )
  return training_module


def training_fn():
  X, y = data.get_dataframe()
  m = models.get_model()
  model_training_module = _get_model_training_module()
  actual_training_fn = getattr(
    model_training_module, FLAGS.train_mode.name.lower()
  )
  return actual_training_fn(X, y)
