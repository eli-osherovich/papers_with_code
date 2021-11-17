import functools

import gin
import tensorflow as tf

from . import metatree


def get_keras_class(module, **kwargs):
  cur_module = getattr(tf.keras, module)
  cls_instance = cur_module.get(kwargs.pop("identifier"))
  return type(cls_instance)(**kwargs)


@gin.configurable
def get_optimizer(**kwargs):
  return get_keras_class("optimizers", **kwargs)


@gin.configurable
def get_loss(**kwargs):
  return get_keras_class("losses", **kwargs)


@gin.configurable
def get_metric(**kwargs):
  return get_keras_class("metrics", **kwargs)


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
    loss=get_loss(),
    metrics=[
      get_metric(),
    ],
    optimizer=get_optimizer(),
  )

  return model
