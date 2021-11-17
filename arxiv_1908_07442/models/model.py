"""A common interface to all the implemented models.
"""
import enum
import sys

from absl import flags
import tensorflow as tf

from . import tabnet_model

FLAGS = flags.FLAGS


@enum.unique
class MODEL(enum.Enum):
  TABNET = 1


flags.DEFINE_enum_class("model", MODEL.TABNET, MODEL, "Model to use")
flags.DEFINE_integer("num_classes", None, "Number of classes", lower_bound=1)


def _get_model_module():
  module_name = FLAGS.model.name.lower() + "_model"
  full_name = __package__ + "." + module_name
  try:
    return sys.modules[full_name]
  except KeyError:
    raise RuntimeError(
      f"Module {full_name} is not loaded. Did you forget to add `from . import {module_name}`?"
    )


def get_model(**model_args) -> tf.keras.Model:
  if FLAGS.num_classes is not None:
    model_args["num_classes"] = FLAGS.num_classes
  model_module = _get_model_module()
  return model_module.get_model(**model_args)
