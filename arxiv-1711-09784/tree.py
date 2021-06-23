from typing import Union

import tensorflow as tf


class TreeModel(tf.keras.Model):

  def __init__(self, depth, n_classes=10):
    super().__init__()
    self.model = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      build_tree(depth, n_classes),
    ])

  def call(self, inputs: tf.Tensor):
    return self.model(inputs)


class LeafNode(tf.keras.layers.Layer):

  def __init__(self, dim):
    super().__init__(name="leaf")
    self.logits = self.add_weight(name="logits", shape=(dim,), trainable=True)

  def call(self, _inputs: tf.Tensor) -> tf.Variable:
    return self.logits


class InnerNode(tf.keras.layers.Layer):

  def __init__(self):
    super().__init__(name="inner_node")
    self.dense = tf.keras.layers.Dense(1, activation="sigmoid")

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    pR = self.dense(inputs)
    return pR * self.right(inputs) + (1 - pR) * self.left(inputs)


def build_tree(depth: int, n_classes: int) -> Union[LeafNode, InnerNode]:
  if depth == 1:
    return LeafNode(n_classes)

  root = InnerNode()
  root.left = build_tree(depth - 1, n_classes)
  root.right = build_tree(depth - 1, n_classes)
  return root
