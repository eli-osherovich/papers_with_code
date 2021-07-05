import tensorflow as tf
import tensorflow_datasets as tfds
from absl import flags

from . import transform

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "dataset",
  default="cifar10",
  help="TFDS dataset.",
)

flags.DEFINE_multi_integer(
  "normal",
  default=0,
  help="Classes defined as normal.",
)

flags.DEFINE_integer(
  "shuffle_buffer", 5000, "Shuffle buffer size.", lower_bound=1)
flags.DEFINE_integer("batch_size", 256, "Batch size.", lower_bound=1)
flags.DEFINE_string("cache", "", "Cache file for datasets.")


def preprocessing(x, y):
  return tf.image.convert_image_dtype(x, tf.keras.backend.floatx()), y


def filter_normal(x, y):
  del x  # unused parameter
  # Actual code is equivalent to the line below, but
  # suitable for a graph execution:
  # return y in FLAGS.normal
  return tf.math.reduce_any(tf.equal(y, FLAGS.normal))


def split_normal_anomalous(ds):
  normal = ds\
    .filter(filter_normal)\
    .batch(FLAGS.batch_size)\
    .map(preprocessing)

  anomalous = ds\
    .filter(lambda x, y: not filter_normal(x, y))\
    .batch(FLAGS.batch_size)\
    .map(preprocessing)

  return normal, anomalous


def get_datasets():
  train, test = tfds.load(
    FLAGS.dataset, split=["train", "test"], as_supervised=True)

  train_normal, train_anomalous = split_normal_anomalous(train)
  test_normal, test_anomalous = split_normal_anomalous(test)

  transforms = transform.get_transforms()

  @tf.function
  def apply_transforms(x, y):
    del y  # unused parameter
    data = [(t(x), tf.repeat(i, len(x))) for i, t in enumerate(transforms)]
    xx = tf.concat([d[0] for d in data], axis=0)
    yy = tf.concat([d[1] for d in data], axis=0)
    return tf.data.Dataset.from_tensor_slices((xx, yy))

  def prepare_dataset(ds, shuffle=True):
    ds = ds.flat_map(apply_transforms)
    if FLAGS.cache.lower() != "none":
      ds = ds.cache(FLAGS.cache)
    if shuffle:
      ds = ds.shuffle(FLAGS.shuffle_buffer)
    return ds.batch(
      FLAGS.batch_size,
      num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

  return (
    len(transforms),
    prepare_dataset(train_normal),
    prepare_dataset(train_anomalous),
    prepare_dataset(test_normal, shuffle=False),
    prepare_dataset(test_anomalous, shuffle=False),
  )
