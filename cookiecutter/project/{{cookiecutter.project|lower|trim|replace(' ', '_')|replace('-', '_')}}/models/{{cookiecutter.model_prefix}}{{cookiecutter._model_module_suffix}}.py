"""{{cookiecutter.model}} implementation."""

from absl import flags
import gin
import tensorflow as tf

from . import tasks

FLAGS = flags.FLAGS


@gin.configurable
class {{cookiecutter.model}}(tf.keras.Model):

  def __init__(self, n_hidden: int, **kwargs) -> None:
    super().__init__(**kwargs)
    self.n_hidden = n_hidden

  def build(self, input_shape):
    batch_size, *input_dim = input_shape
    self.model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=input_dim, batch_size=batch_size)
    ])
    for _ in range(self.n_hidden):
      self.model.add(tf.keras.layers.Dense(16, activation="relu"))

  def call(self, x, **kwargs):
    return self.model(x, **kwargs)


@gin.configurable
def get_model(**kwargs) -> tf.keras.Model:
  if FLAGS.task == tasks.TASK.REGRESSION:
    return get_regression_model(**kwargs)
  elif FLAGS.task == tasks.TASK.BINARY:
    return get_binary_model(**kwargs)
  elif FLAGS.task == tasks.TASK.MULTICLASS:
    return get_multiclass_model(**kwargs)
  else:
    raise RuntimeError(f"Wrong task type: {FLAGS.type}")

@gin.configurable
def get_regression_model(**kwargs) -> tf.keras.Model:
  # Add a regression head to the main trunk.
  model = tf.keras.Sequential([
    {{cookiecutter.model}}(**kwargs),
    tf.keras.layers.Dense(1),
  ])
  model.compile(loss="mse", optimizer="adam", metrics=["RootMeanSquaredError"])
  return model

@gin.configurable
def get_binary_model(**kwargs) -> tf.keras.Model:
  # Add a binary classification head to the main trunk.
  model = tf.keras.Sequential([
    {{cookiecutter.model}}(**kwargs),
    tf.keras.layers.Dense(1, activation="sigmoid"),
  ])
  model.compile(loss="bce", optimizer="adam", metrics=["accuracy"])
  return model

@gin.configurable
def get_multiclass_model(num_classes: int = {{cookiecutter.num_classes}}, **kwargs) -> tf.keras.Model:
  # Add a binary classification head to the main trunk.
  model = tf.keras.Sequential([
    {{cookiecutter.model}}(**kwargs),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
  ])
  model.compile(
    loss="SparseCategoricalCrossentropy",
    optimizer="adam",
    metrics=["accuracy"]
  )
  return model