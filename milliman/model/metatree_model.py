import gin
import tensorflow as tf

from . import metatree


@gin.configurable
def get_model(depth: int, input_dim: int, emb_dim: int) -> tf.keras.Model:
  return metatree.TreeModel(depth=depth, input_dim=input_dim, emb_dim=emb_dim)
