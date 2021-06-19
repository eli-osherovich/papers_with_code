from typing import Union

import tensorflow as tf


class TreeModel(tf.keras.Model):

  def __init__(self, depth, input_dim, emb_dim, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.f = gen_split_model(input_dim, emb_dim)
    self.g = gen_value_encoder(emb_dim)
    self.h = gen_input_encoder(input_dim, emb_dim)

    self.tree = build_tree(depth, self.f, self.g)

  def call(self, inputs):
    x, y = inputs
    h = self.h((x, y))
    r = tf.math.reduce_mean(h, axis=1)

    I = tf.ones(x.shape[:-1], dtype=tf.bool)
    return self.tree((x, r, h), mask=I)


class LeafNode(tf.keras.layers.Layer):

  def __init__(self, model, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.model = model

  def call(self, inputs, mask=None):
    _x, r, h = inputs
    rI = calc_rI(h, mask)
    return self.model((r, rI))


class InnerNode(tf.keras.layers.Layer):

  def __init__(self, model, l2=0.001, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.model = model
    self.l2 = l2

  def call(self, inputs, mask=tf.constant(True)):
    x, r, h = inputs
    rI = calc_rI(h, mask)
    w, b, beta = self.model((r, rI))

    pR = tf.nn.sigmoid(beta * (tf.einsum("bd, bnd -> bn", w, x) + b))
    self.add_loss(
        self.l2 *
        tf.math.reduce_mean(tf.keras.losses.MSE(tf.constant(0.5), pR)))
    maskR = tf.math.logical_and(mask, pR >= 0.5)
    maskL = tf.math.logical_and(mask, pR < 0.5)

    # tf.print(tf.math.reduce_sum(tf.cast(maskR, tf.int64)))
    # tf.print(tf.math.reduce_sum(tf.cast(maskL, tf.int64)))
    return (pR * self.right(inputs, maskR) +
            (1 - pR) * self.left(inputs, maskL))


@tf.function
def calc_rI(h: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
  # The code is equivalent to the two lines below, but does not use ragged (slow) tensors.
  # rI = tf.math.reduce_mean(tf.ragged.boolean_mask(h, mask), axis=1)
  # rI = tf.where(tf.math.is_nan(rI), tf.zeros_like(rI), rI)

  weights = tf.cast(mask, tf.keras.backend.floatx())
  sum_ = tf.math.reduce_sum(weights, axis=-1, keepdims=True)
  weights = weights * tf.math.reciprocal_no_nan(sum_)
  rI = tf.einsum("pr, pre -> pe", weights, h)
  return rI


def build_tree(depth, inner_model, leaf_model):
  if depth == 1:
    return LeafNode(leaf_model)

  root = InnerNode(inner_model)
  root.left = build_tree(depth - 1, inner_model, leaf_model)
  root.right = build_tree(depth - 1, inner_model, leaf_model)
  return root


def gen_value_encoder(emb_dim):
  # This is g(r, rI) in the paper.
  r = tf.keras.Input(shape=(emb_dim,))
  rI = tf.keras.Input(shape=(emb_dim,))
  x = tf.keras.layers.concatenate([r, rI])
  x = tf.keras.layers.Dense(20, activation='relu')(x)
  x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
  return tf.keras.Model(inputs=[r, rI], outputs=x)


def gen_input_encoder(input_dim, emb_dim):
  # This is h(x, y) in the paper.
  x = tf.keras.Input(shape=(None, input_dim))
  y = tf.keras.Input(shape=(None, 1))
  h = tf.keras.layers.concatenate((x, y))
  h = tf.keras.layers.Dense(emb_dim, activation="relu")(h)
  h = tf.keras.layers.Dense(emb_dim, activation="relu")(h)
  h = tf.keras.layers.Dense(emb_dim, activation="relu")(h)
  h = tf.keras.layers.Dense(emb_dim)(h)
  return tf.keras.Model(inputs=(x, y), outputs=h)


def gen_split_model(input_dim, emb_dim, l1=0.01):
  # This is f(r, rI) in the paper.
  r = tf.keras.Input(shape=(emb_dim,))
  rI = tf.keras.Input(shape=(emb_dim,))
  x = tf.keras.layers.concatenate([r, rI])
  x = tf.keras.layers.Dense(50, activation="relu")(x)
  w = tf.keras.layers.Dense(
      input_dim,
      activity_regularizer=tf.keras.regularizers.L1(l1),
      activation="softmax")(
          x)
  b = tf.keras.layers.Dense(1)(x)
  beta = tf.keras.layers.Dense(1)(x)
  return tf.keras.Model(inputs=[r, rI], outputs=[w, b, beta])
