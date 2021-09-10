import functools
from typing import Callable

import tensorflow as tf
import tensorflow_addons as tfa

L2 = 1e-3
L1 = 1e-4


class TreeModel(tf.keras.Model):
  """Class representing an entire tree"""

  def __init__(self, *, depth: int, input_dim: int, emb_dim: int) -> None:
    super().__init__()
    self.encoder = gen_input_encoder(input_dim=input_dim, emb_dim=emb_dim)
    self.tree = build_tree(
      depth=depth,
      inner_model_fn=functools.partial(
        gen_inner_model, input_dim=input_dim, emb_dim=emb_dim
      ),
      leaf_model_fn=functools.partial(gen_leaf_model, emb_dim=emb_dim),
    )

  def call(self, x):
    mask = tf.ones(shape=(x.shape[0], 1), dtype=tf.bool)
    emb = self.encoder(x)
    return self.tree((x, emb), mask=mask)


class LeafNode(tf.keras.layers.Layer):
  """Class represanting leaf nodes"""

  def __init__(self, model_fn: Callable[[], tf.keras.Model], id_: int = 0):
    super().__init__()
    self.id = id_
    self.model = model_fn()

  def call(self, inputs, *, mask=None):
    x, emb = inputs
    del x, mask  # unused
    self.value = self.model(emb)
    return self.value


class InnerNode(tf.keras.layers.Layer):
  """Class representing inner nodes"""

  def __init__(
    self,
    model_fn: Callable[[], tf.keras.Model],
    proba_reg_weight: float,
    id_: int = 0,
  ) -> None:
    super().__init__()
    self.model = model_fn()
    self.proba_reg_weight = proba_reg_weight
    self.id = id_

  def call(self, inputs, *, training=None, mask=None):
    x, emb = inputs
    w, b, beta = self.model(emb)

    logits = beta * (w * x + b)

    w_max_idx = tf.math.argmax(w, axis=1)

    # Use only one feature as we do during inference.
    proba_right = tf.nn.sigmoid(
      tf.gather(logits, w_max_idx, batch_dims=1, axis=1)[..., None]
    )

    if training:
      # We do not use this loss for evaluation (hence, for monitoring either)
      self.add_loss(
        self.proba_reg_weight *
        tf.keras.losses.binary_crossentropy([0.5],
                                            tf.math.reduce_mean(proba_right))
      )
    else:
      # During inference the behavior is different in this aspect:
      # 1. The system uses hard decision trees.

      # Hard decision tree.
      proba_right = tf.where(proba_right >= 0.5, 1.0, 0.0)

      # Bookeeping for tree printing.
      # Split by one feature only: the one that results in the largest logit.
      self.split_feature_idx = w_max_idx
      self.threshold = -tf.squeeze(b) / tf.gather(
        w, self.split_feature_idx, batch_dims=1, axis=1
      )

      # Depending on the sing of beta, the inequality may be either:
      # 1. geq (>=)
      # 2. leq (<=)
      self.geq = tf.squeeze(beta >= 0)

      self.x = tf.gather(x, w_max_idx, batch_dims=1, axis=1)
      self.w = tf.gather(w, w_max_idx, batch_dims=1, axis=1)
      self.ww = w
      self.beta = tf.squeeze(beta)
      self.b = tf.squeeze(b)
      self.proba_right = tf.squeeze(proba_right)

    mask_right = tf.math.logical_and(mask, proba_right >= 0.5)
    mask_left = tf.math.logical_and(mask, proba_right < 0.5)
    emb_right = tf.cast(mask_right, emb.dtype) * emb
    emb_left = tf.cast(mask_left, emb.dtype) * emb
    # tf.print(
    #   self.id,
    #   tf.math.count_nonzero(mask),
    #   tf.math.count_nonzero(mask_left),
    #   tf.math.count_nonzero(mask_right),
    # )
    return (1 - proba_right) * self.left(
      (x, emb_left), mask=mask_left
    ) + proba_right * self.right((x, emb_right), mask=mask_right)


def gen_input_encoder(
  *,
  input_dim: int,
  emb_dim: int,
) -> tf.keras.Model:
  return tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(input_dim,)),
    tf.keras.layers.GaussianNoise(0.25),
    tf.keras.layers.Dense(
      emb_dim,
      activation="relu",
      kernel_regularizer=tf.keras.regularizers.L1L2(L1, L2),
    ),
    tf.keras.layers.Dense(
      emb_dim,
      activation="relu",
      kernel_regularizer=tf.keras.regularizers.L1L2(L1, L2),
    ),
    tf.keras.layers.Dense(
      emb_dim,
      activation="relu",
      kernel_regularizer=tf.keras.regularizers.L1L2(L1, L2),
    ),
    tf.keras.layers.Dense(
      emb_dim,
      activation="relu",
      kernel_regularizer=tf.keras.regularizers.L1L2(L1, L2),
    ),
    tf.keras.layers.Dropout(0.5),
  ])


def gen_inner_model(*, input_dim: int, emb_dim: int) -> tf.keras.Model:
  emb = tf.keras.Input(shape=(emb_dim,))
  h = tf.keras.layers.Dense(
    emb_dim,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.L1L2(L1, L2),
  )(
    emb
  )
  w = tf.keras.layers.Dense(
    input_dim,
    bias_initializer="ones",
    kernel_regularizer=tf.keras.regularizers.L1L2(L1, L2),
    activation="relu"
  )(
    h
  )
  w /= tf.math.reduce_max(w, axis=1, keepdims=True)
  w = tfa.activations.sparsemax(w)
  w /= tf.math.reduce_max(w, axis=1, keepdims=True)

  b = tf.keras.layers.Dense(
    1, activation="tanh", kernel_regularizer=tf.keras.regularizers.L1L2(L1, L2)
  )(
    h
  )
  # TODO: add true b's bounds:
  # to this end, we might try different b per feature.
  beta = tf.keras.layers.Dense(
    1, kernel_regularizer=tf.keras.regularizers.L1L2(L1, L2)
  )(
    h
  )

  return tf.keras.Model(inputs=emb, outputs=[w, b, beta])


def gen_leaf_model(emb_dim: int) -> tf.keras.Model:
  return tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(emb_dim,)),
    tf.keras.layers.Dense(
      emb_dim,
      activation="relu",
      kernel_regularizer=tf.keras.regularizers.L1L2(L1, L2),
    ),
    tf.keras.layers.Dense(
      1,
      activation="sigmoid",
      kernel_regularizer=tf.keras.regularizers.L1L2(L1, L2),
    ),
  ])


def build_tree(
  *,
  depth: int,
  inner_model_fn: Callable[[], tf.keras.Model],
  leaf_model_fn: Callable[[], tf.keras.Model],
  proba_reg_weight: float = 0.5,
  proba_reg_reduction_factor: float = 1.0,
  root_id: int = 0,
):

  if depth == 1:
    return LeafNode(leaf_model_fn, root_id)

  root = InnerNode(inner_model_fn, proba_reg_weight, root_id)

  root.left = build_tree(
    depth=depth - 1,
    inner_model_fn=inner_model_fn,
    leaf_model_fn=leaf_model_fn,
    proba_reg_weight=proba_reg_weight / proba_reg_reduction_factor,
    root_id=2 * root_id + 1,
  )

  root.right = build_tree(
    depth=depth - 1,
    inner_model_fn=inner_model_fn,
    leaf_model_fn=leaf_model_fn,
    proba_reg_weight=proba_reg_weight / proba_reg_reduction_factor,
    root_id=2 * root_id + 2,
  )

  return root
