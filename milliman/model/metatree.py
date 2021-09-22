from collections.abc import Sequence
import functools
from typing import Any, Callable

import tensorflow as tf
import tensorflow_addons as tfa

L2 = 5e-4
L1 = 5e-4

_B_LO = tf.constant([
  -2.2338567283409025, -3.321779081441261, -2.7279813295174837,
  -4.773952579363098, -3.248870315174375, -1.3781703984907252,
  -0.9421637820708985, -1.053413018870969, -3.5943817696146394,
  -4.572632218970465, -2.29128784747792, -1.2089410496539776,
  -1.2375966910186262, -1.0672569667868559, -4.096068575814836,
  -1.370291152174099, -3.021023382643871, -6.591239977761729,
  -11.50362261782493, -0.4823542675723937, -2.4784787961282104,
  -4.41733183186644, -0.6352234031660235, -2.1343747458109497,
  -2.881375640452898, -7.000000000000001, -2.015760685012114,
  -0.752035723846475, -2.440598643597598
])

_B_HI = tf.constant([
  1.4584684424705066, 1.41999070088631, 1.9129225803474537, 1.0904665413522914,
  0.6654312693730648, 1.5697876731471898, 1.727485659452309, 1.6476460038751053,
  1.4788138851768238, 0.7187454693270836, 0.4364357804719845, 0.827170191868511,
  0.8080176743014172, 0.9369814684936247, 0.24413653763134782,
  0.7297719162919529, 0.33101365773767794, 0.15171652122725207,
  0.08692913816996169, 2.0731650308243945, 0.4034732923929645,
  0.22638100058185434, 1.5742493034984064, 0.46852128566581813,
  0.3470564496904051, 0.14285714285714288, 0.4960906358752555,
  1.3297240653479196, 0.409735538706166
])


class TreeModel(tf.keras.Model):
  """Class representing an entire tree"""

  def __init__(self, *, depth: int, input_dim: int, emb_dim: int) -> None:
    super().__init__()
    self.encoder = gen_input_encoder(input_dim=input_dim, emb_dim=emb_dim)
    self.tree = build_tree(
      depth=depth,
      inner_model_fn=functools.partial(
        gen_inner_model,
        input_dim=input_dim,
        emb_dim=emb_dim,
        blimits=(_B_LO, _B_HI)
      ),
      leaf_model_fn=functools.partial(
        gen_leaf_model, input_dim=input_dim, emb_dim=emb_dim
      ),
    )

  def call(self, x):
    emb = self.encoder(x)
    return self.tree((x, emb))


class LeafNode(tf.keras.layers.Layer):
  """Class represanting leaf nodes"""

  def __init__(self, *, model_fn: Callable[[], tf.keras.Model], id_: int = 0):
    super().__init__()
    self.id = id_
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
    self.proba_reg_weight = proba_reg_weight
    self.beta = beta
    self.id = id_

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
    else:
      # During inference the behavior is different in this aspect:
      # 1. The system uses hard decision trees.

      # Hard decision tree.
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
  *, input_dim: int, emb_dim: int, blimits: Sequence[Any, Any]
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
  lo, hi = blimits
  b = b * (hi - lo) / 2 + (hi + lo) / 2

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
