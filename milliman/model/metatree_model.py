import gin
import tensorflow as tf

from . import metatree


@gin.configurable
def get_model(depth: int, input_dim: int, emb_dim: int) -> tf.keras.Model:
  model = metatree.TreeModel(depth=depth, input_dim=input_dim, emb_dim=emb_dim)
  model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
      tf.keras.metrics.BinaryAccuracy(name='acc'),
    ],
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4))

  return model
