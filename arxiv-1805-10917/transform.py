import functools
import itertools

import tensorflow as tf
from absl import flags

from ..common import utils

FLAGS = flags.FLAGS

flags.DEFINE_integer("tx", 10, "x-axis translation (in pixels).")
flags.DEFINE_integer("ty", 10, "y-axis translation (in pixels).")
flags.DEFINE_enum("fill_mode", "reflect",
                  ["reflect", "wrap", "constant", "nearest"], "Fill mode.")
flags.DEFINE_enum("interpolation", "bilinear", ["nearest", "bilinear"],
                  "Interpolation mode.")


def get_translation_transform(tx=0, ty=0, fill_value=0, output_shape=None):
  matrix = tf.constant(
    [
      [1, 0, tx],
      [0, 1, ty],
      [0, 0, 1],
    ],
    dtype=tf.keras.backend.floatx(),
  )
  transform = utils.matrix_to_transform(matrix)

  def _translation(images):
    nonlocal output_shape
    if output_shape is None:
      output_shape = tf.shape(images)[1:3]

    return tf.raw_ops.ImageProjectiveTransformV3(
      images=images,
      transforms=transform,
      output_shape=output_shape,
      fill_value=fill_value,
      interpolation=FLAGS.interpolation.upper(),
      fill_mode=FLAGS.fill_mode.upper())

  return _translation


def get_flip_lr_transform():
  return tf.image.flip_left_right


def get_rot90_transform(k=1):
  return functools.partial(tf.image.rot90, k=k)


def identity(images):
  return images


def combine_transforms(*transforms):

  def _transform(images):
    for t in transforms:
      images = t(images)
    return images

  return _transform


def get_transforms():
  lr_flips = [identity, get_flip_lr_transform()]
  translations = [
    get_translation_transform(tx, ty) for tx, ty in itertools.product(
      [0, FLAGS.tx, -FLAGS.tx], [0, FLAGS.ty, -FLAGS.ty])
  ]
  rotations = [get_rot90_transform(k) for k in [0, 1, 2, 3]]

  transforms = itertools.starmap(
    combine_transforms, itertools.product(lr_flips, translations, rotations))
  return list(transforms)
