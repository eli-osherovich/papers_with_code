from typing import Union, Optional

import tensorflow as tf


class TreeModel(tf.keras.Model):

  def __init__(self,
               n_classes: int,
               depth: int,
               leaf_initializer: Optional[float] = None):
    super().__init__()
    self.model = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      build_tree(
        n_classes=n_classes, depth=depth, leaf_initializer=leaf_initializer),
    ])

  def call(self, inputs: tf.Tensor):
    return self.model(inputs)


class LeafNode(tf.keras.layers.Layer):

  def __init__(self, *, dim, init_value=None):
    super().__init__(name='leaf')
    if init_value is not None:
      init_value = tf.keras.initializers.Constant(init_value)

    self.logits = self.add_weight(
      name='logits', shape=(dim,), trainable=True, initializer=init_value)

  def call(self, _inputs: tf.Tensor) -> tf.Variable:
    return self.logits


class InnerNode(tf.keras.layers.Layer):

  def __init__(self):
    super().__init__(name='inner_node')
    self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    pR = self.dense(inputs)
    return pR * self.right(inputs) + (1 - pR) * self.left(inputs)


def build_tree(
    *,
    n_classes: int,
    depth: int,
    leaf_initializer: Optional[float] = None) -> Union[LeafNode, InnerNode]:
  if depth == 1:
    return LeafNode(dim=n_classes, init_value=leaf_initializer)

  root = InnerNode()
  root.left = build_tree(n_classes=n_classes, depth=depth - 1)
  root.right = build_tree(n_classes=n_classes, depth=depth - 1)
  return root
