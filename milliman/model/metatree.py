from collections.abc import Sequence, Callable
import functools
from typing import Callable

import tensorflow as tf

L2 = 5e-4
L1 = 5e-4

VECTOR = Sequence[float]


class TreeModel(tf.keras.Model):
  """Class representing an entire tree"""

  def __init__(
    self,
    *,
    depth: int,
    encoder_model_fn: Callable[[], tf.keras.Model],
    inner_model_fn: Callable[[], tf.keras.Model],
    leaf_model_fn: Callable[[], tf.keras.Model],
  ) -> None:
    super().__init__()
    self.encoder = encoder_model_fn()
    self.tree = build_tree(
      depth=depth, inner_model_fn=inner_model_fn, leaf_model_fn=leaf_model_fn
    )

  def call(self, x):
    emb = self.encoder(x)
    return self.tree((x, emb))


class LeafNode(tf.keras.layers.Layer):
  """Class represanting leaf nodes"""

  def __init__(self, *, model_fn: Callable[[], tf.keras.Model], id_: int = 0):
    super().__init__()
    self.id = tf.constant(id_)
    self.model = model_fn()

  def call(self, inputs):
    self.value = self.model(inputs)
    return self.value


class InnerNode(tf.keras.layers.Layer):
  """Class representing inner nodes"""

  def __init__(
    self,
    *,
    model_fn: Callable[[], tf.keras.Model],
    proba_reg_weight: float,
    beta: float,
    id_: int = 0,
  ) -> None:
    super().__init__()
    self.model = model_fn()
    self.proba_reg_weight = tf.constant(proba_reg_weight)
    self.beta = tf.constant(beta)
    self.id = tf.constant(id_)

  def call(self, inputs, *, training=None):
    x, emb = inputs
    del emb  # unused
    w, b = self.model(inputs)

    logits = self.beta * (w * x + b)

    w_max_idx = tf.math.argmax(w, axis=1)

    # Use only one feature as we do during inference.
    p_right = tf.nn.sigmoid(
      tf.gather(logits, w_max_idx, batch_dims=1, axis=1)[..., None]
    )
    if training:
      # We do not use this loss for evaluation (hence, for monitoring either)
      self.add_loss(
        self.proba_reg_weight *
        tf.keras.losses.binary_crossentropy([0.5], tf.math.reduce_mean(p_right))
      )
      self.add_loss(1e-4 * tf.keras.regularizers.l1_l2(l1=L1, l2=L2)(w))
    else:
      # During inference the behavior is different in this aspect:
      # 1. The system uses hard decision trees.

      # Hard decision tree.
      self.proba_right_orig = tf.squeeze(p_right)

      p_right = tf.where(p_right >= 0.5, 1.0, 0.0)

      # Bookeeping for tree printing.
      # Split by one feature only: the one that results in the largest logit.
      self.split_feature_idx = w_max_idx
      self.threshold = -tf.gather(
        b, w_max_idx, batch_dims=1, axis=1
      ) / tf.gather(
        w, self.split_feature_idx, batch_dims=1, axis=1
      )

      # Depending on the sing of beta, the inequality may be either:
      # 1. geq (>=)
      # 2. leq (<=)
      self.geq = self.beta >= 0

      self.x = tf.gather(x, w_max_idx, batch_dims=1, axis=1)
      self.w = tf.gather(w, w_max_idx, batch_dims=1, axis=1)
      self.ww = w
      self.b = tf.gather(b, w_max_idx, batch_dims=1, axis=1)
      self.proba_right = tf.squeeze(p_right)

    return (1 - p_right) * self.left(inputs) + p_right * self.right(inputs)


def gen_input_encoder(
  *,
  input_dim: int,
  emb_dim: int,
) -> tf.keras.Model:
  return tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(input_dim,)),
    tf.keras.layers.GaussianNoise(0.1),
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


def gen_inner_model(
  *, input_dim: int, emb_dim: int, b_limits: Sequence[VECTOR, VECTOR]
) -> tf.keras.Model:
  # X and EMB are combined via ResNet-like style:
  # 1. x is casted to the same dimension as emb (via matrix multiplication)
  # 2. they are summed.
  x = tf.keras.Input(shape=(input_dim,))
  emb = tf.keras.Input(shape=(emb_dim,))
  h = tf.keras.layers.Dense(emb_dim, use_bias=False)(x)
  h = h + emb
  h = tf.keras.layers.Dense(
    emb_dim,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.L1L2(L1, L2),
  )(
    h
  )
  w = tf.keras.layers.Dense(
    input_dim,
    bias_initializer="ones",
    kernel_regularizer=tf.keras.regularizers.L1L2(L1, L2),
    activation="relu"
  )(
    h
  )
  # w /= tf.math.reduce_max(w, axis=1, keepdims=True)
  # w = tfa.activations.sparsemax(w)
  w /= tf.math.reduce_max(w, axis=1, keepdims=True)
  w = tf.math.pow(w, 20)
  w /= tf.math.reduce_max(w, axis=1, keepdims=True)

  b = tf.keras.layers.Dense(
    input_dim,
    kernel_regularizer=tf.keras.regularizers.L1L2(L1, L2),
    activation="tanh"
  )(
    h
  )
  b_lo, b_hi = b_limits
  b = (b_hi - b_lo) / 2 * b + (b_hi + b_lo) / 2

  return tf.keras.Model(inputs=(x, emb), outputs=(w, b))


def gen_leaf_model(*, input_dim: int, emb_dim: int) -> tf.keras.Model:
  # X and EMB are combined via ResNet-like style:
  # 1. x is casted to the same dimension as emb (via matrix multiplication)
  # 2. they are summed.
  x = tf.keras.Input(shape=(input_dim,))
  emb = tf.keras.Input(shape=(emb_dim,))
  h = tf.keras.layers.Dense(emb_dim, use_bias=False)(x)
  h = h + emb
  h = tf.keras.layers.Dense(
    emb_dim,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.L1L2(L1, L2),
  )(
    h
  )
  h = tf.keras.layers.Dense(
    1,
    activation="sigmoid",
    kernel_regularizer=tf.keras.regularizers.L1L2(L1, L2),
  )(
    h
  )
  return tf.keras.Model(inputs=(x, emb), outputs=h)


def build_tree(
  *,
  depth: int,
  inner_model_fn: Callable[[], tf.keras.Model],
  leaf_model_fn: Callable[[], tf.keras.Model],
  proba_reg_weight: float = 0.5,
  proba_reg_reduction_factor: float = 1.0,
  beta=10.0,
  root_id: int = 0,
):

  if depth == 1:
    return LeafNode(model_fn=leaf_model_fn, id_=root_id)

  root = InnerNode(
    model_fn=inner_model_fn,
    proba_reg_weight=proba_reg_weight,
    beta=beta,
    id_=root_id
  )

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
