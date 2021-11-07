"""Common interface to model training and tuning.
"""
import enum
import sys

from absl import flags

from .. import data
from .. import models  # noqa F401 pylint: disable=unused-import

from . import {{cookiecutter.model_prefix}}{{cookiecutter._training_module_suffix}}  # noqa F401 pylint: disable=unused-import

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
  module_name = FLAGS.model.name.lower() + "{{cookiecutter._training_module_suffix}}"
  full_name = __package__ + "." + module_name
  try:
    return sys.modules[full_name]
  except KeyError:
    raise RuntimeError(
      f"Module {full_name} is not loaded. Did you forget to add `from . import {module_name}`?"
    )


def training_fn():
  x, y = data.get_numpy()
  model_training_module = _get_model_training_module()
  print(model_training_module)
  actual_training_fn = getattr(
    model_training_module, FLAGS.train_mode.name.lower()
  )
  return actual_training_fn(x, y)
