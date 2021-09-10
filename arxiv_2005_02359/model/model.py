from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_integer(
  "n_layers",
  default=5,
  help="Number of layers in the model",
  lower_bound=1,
)

flags.DEFINE_integer(
  "layer_dim",
  default=64,
  help="Dimensionality of inner layers in the model",
  lower_bound=1,
)


def get_model(n_classes):
  model = tf.keras.Sequential()
  for _ in range(FLAGS.n_layers):
    model.add(tf.keras.layers.Dense(FLAGS.layer_dim, activation="relu"))
  model.add(tf.keras.layers.Dense(n_classes))
  return model
