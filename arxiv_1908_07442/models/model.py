"""A common interface to all the implemented models.
"""
import enum
import importlib
import sys

from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS


@enum.unique
class MODEL(enum.Enum):
  TABNET = 1


flags.DEFINE_enum_class("model", MODEL.TABNET, MODEL, "Model to use")


def get_model(**model_args) -> tf.keras.Model:
  module_name = FLAGS.model.name.lower() + "_model"
  model_module = importlib.import_module(
    "." + module_name, sys.modules[__package__].__name__
  )
  return model_module.get_model(**model_args)
