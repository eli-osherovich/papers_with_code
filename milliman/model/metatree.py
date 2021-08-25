import functools
from typing import Callable

import tensorflow as tf


class TreeModel(tf.keras.Model):

  def __init__(self, *, depth: int, input_dim: int, emb_dim: int) -> None:
    super().__init__()
    self.encoder = gen_input_encoder(input_dim=input_dim, emb_dim=emb_dim)
    self.tree = build_tree(
      depth=depth,
      inner_model_fn=functools.partial(
        gen_inner_model, input_dim=input_dim, emb_dim=emb_dim),
      leaf_model_fn=functools.partial(gen_leaf_model, emb_dim=emb_dim))

  def call(self, x):
    emb = self.encoder(x)
    return self.tree((x, emb))


class LeafNode(tf.keras.layers.Layer):

  def __init__(self, model_fn: Callable[[], tf.keras.Model]):
    super().__init__()
    self.model = model_fn()

  def call(self, inputs):
    x, emb = inputs
    del x  # unused
    return self.model(emb)


class InnerNode(tf.keras.layers.Layer):

  def __init__(self,
               model_fn: Callable[[], tf.keras.Model],
               reg_weight=0.01) -> None:
    super().__init__()
    self.model = model_fn()
    self.reg_weight = reg_weight

  def call(self, inputs):
    x, emb = inputs
    w, b, beta = self.model(emb)

    pR = tf.nn.sigmoid(beta *
                       (tf.math.reduce_sum(w * x, axis=1, keepdims=True) + b))
    self.add_loss(
      self.reg_weight *
      tf.keras.losses.binary_crossentropy(0.5, tf.math.reduce_mean(pR, axis=0)))
    return pR * self.right(inputs) + (1 - pR) * self.left(inputs)


def gen_input_encoder(
  *,
  input_dim: int,
  emb_dim: int,
) -> tf.keras.Model:
  return tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(input_dim,)),
    tf.keras.layers.Dense(emb_dim, activation='relu'),
    tf.keras.layers.Dense(emb_dim, activation='relu'),
    tf.keras.layers.Dense(emb_dim, activation='relu'),
    tf.keras.layers.Dense(emb_dim, activation='relu'),
  ],)


def gen_inner_model(*,
                    input_dim: int,
                    emb_dim: int,
                    l1: float = 0.01) -> tf.keras.Model:
  emb = tf.keras.Input(shape=(emb_dim,))
  h = tf.keras.layers.Dense(50, activation='relu')(emb)
  w = tf.keras.layers.Dense(
    input_dim,
    activity_regularizer=tf.keras.regularizers.L1(l1),
    activation='softmax')(
      h)
  b = tf.keras.layers.Dense(1)(h)
  beta = tf.keras.layers.Dense(1)(h)
  return tf.keras.Model(inputs=emb, outputs=[w, b, beta])


def gen_leaf_model(emb_dim: int) -> tf.keras.Model:
  return tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(emb_dim,)),
    tf.keras.layers.Dense(emb_dim, activation='relu'),
    tf.keras.layers.Dense(1),
  ])


def build_tree(
  *,
  depth: int,
  inner_model_fn: Callable[[], tf.keras.Model],
  leaf_model_fn: Callable[[], tf.keras.Model],
):
  if depth == 1:
    return LeafNode(leaf_model_fn)

  root = InnerNode(inner_model_fn)
  root.left = build_tree(
    depth=depth - 1, inner_model_fn=inner_model_fn, leaf_model_fn=leaf_model_fn)
  root.right = build_tree(
    depth=depth - 1, inner_model_fn=inner_model_fn, leaf_model_fn=leaf_model_fn)
  return root
