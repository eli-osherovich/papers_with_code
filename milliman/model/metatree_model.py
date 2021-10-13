import functools

import gin
import tensorflow as tf

from . import metatree


@gin.configurable
def get_model(depth: int, emb_dim: int, b_limits) -> tf.keras.Model:
  input_dim = len(b_limits[0])

  encoder_fn = functools.partial(
    metatree.gen_input_encoder, input_dim=input_dim, emb_dim=emb_dim
  )

  inner_model_fn = functools.partial(
    metatree.gen_inner_model,
    input_dim=input_dim,
    emb_dim=emb_dim,
    b_limits=b_limits
  )
  leaf_model_fn = functools.partial(
    metatree.gen_leaf_model, input_dim=input_dim, emb_dim=emb_dim
  )

  model = metatree.TreeModel(
    depth=depth,
    encoder_model_fn=encoder_fn,
    inner_model_fn=inner_model_fn,
    leaf_model_fn=leaf_model_fn
  )
  model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
      tf.keras.metrics.BinaryAccuracy(name="acc"),
    ],
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4, amsgrad=True),
  )

  return model
