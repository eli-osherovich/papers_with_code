import sys

from absl import flags

from . import {{cookiecutter.model_prefix}}{{cookiecutter.training_module_suffix}}  # noqa F401 pylint: disable=unused-import
from .. import data
from .. import models  # noqa F401 pylint: disable=unused-import


FLAGS = flags.FLAGS

flags.DEFINE_enum(
  "action", "train", ["train", "train_cv", "tune"], "Action to perform"
)


def _get_action():
  if FLAGS.action == "train":
    return "train"
  elif FLAGS.action == "train_cv":
    return "train_cv"
  elif FLAGS.action == "tune":
    return "tune"
  else:
    raise RuntimeError(f"Unknown action: {FLAGS.action}")


def _get_model_training_module():
  module_name = FLAGS.model.name.lower() + "{{cookiecutter.training_module_suffix}}"
  return getattr(sys.modules[__package__], module_name)


def training_fn():
  X, y = data.get_dataframe()
  model_module = _get_model_training_module()
  action = _get_action()
  actual_training_fn = getattr(model_module, action)
  return actual_training_fn(X, y)
