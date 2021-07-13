import tensorflow as tf
import tensorflow_hub as hub
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string(
  "model", "https://tfhub.dev/google/bit/s-r50x1/1", "TF HUB's  model handle"
)


def get_model(n_classes):
  backbone = hub.KerasLayer(FLAGS.model, trainable=True)

  model = tf.keras.Sequential([
    backbone,
    tf.keras.layers.Dense(n_classes),
  ])

  model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
  )
  return model
