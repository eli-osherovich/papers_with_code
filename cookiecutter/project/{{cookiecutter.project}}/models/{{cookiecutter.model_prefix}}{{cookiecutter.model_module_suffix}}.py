"""{{cookiecutter.model}} implementation."""

from absl import flags
import gin
import tensorflow as tf

from . import tasks

FLAGS = flags.FLAGS


@gin.configurable
class {{cookiecutter.model}}(tf.keras.Model):
  def __init__(self, **kwargs) -> None:
      super().__init__(**kwargs)


@gin.configurable
def get_model(**kwargs) -> tf.keras.Model:
  if FLAGS.task == tasks.TASK.REGRESSION:
    return get_regression_model(**kwargs)
  elif FLAGS.task == tasks.TASK.BINARY:
    return get_binary_model(**kwargs)
  elif FLAGS.task == tasks.TASK.MULTICLASS:
    return get_multiclass_model(**kwargs)
  else:
    raise RuntimeError(f"Wrong problem type: {FLAGS.type}")


def get_regression_model(**kwargs) -> tf.keras.Model:
  raise NotImplementedError()


def get_binary_model(**kwargs) -> tf.keras.Model:
  raise NotImplementedError()


def get_multiclass_model(**kwargs) -> tf.keras.Model:
  raise NotImplementedError()
