import tensorflow as tf
from absl import flags

from ..transform import transform
from ...datasets import common as ds_common

FLAGS = flags.FLAGS

flags.DEFINE_string(
  'dataset',
  default='KddCup99',
  help='Dataset to use',
)
flags.DEFINE_string(
  'train_name',
  default='train',
  help='Name of the train config',
)

flags.DEFINE_string(
  'test_name',
  default='test',
  help='Name of the test config',
)

flags.DEFINE_integer(
  'normal',
  default=1,
  help='Normal label',
)

flags.DEFINE_integer(
  'shuffle_buffer', 1024, 'Shuffle buffer size.', lower_bound=1)

flags.DEFINE_integer('batch_size', 128, 'Batch size.', lower_bound=1)
flags.DEFINE_string('cache', '', 'Cache file for datasets.')


def filter_normal(x, y):
  del x  # unused parameter
  return y == FLAGS.normal


def filter_anomalous(x, y):
  return not filter_normal(x, y)


def split_normal_anomalous(ds, cache: bool = False):
  normal = ds\
    .filter(filter_normal)\
    .shuffle(FLAGS.shuffle_buffer)\
    .batch(FLAGS.batch_size, drop_remainder=True)

  anomalous = ds\
    .filter(filter_anomalous)\
    .shuffle(FLAGS.shuffle_buffer)\
    .batch(FLAGS.batch_size, drop_remainder=True)

  if cache:
    normal = normal.cache()
    anomalous = anomalous.cache()
  return normal, anomalous


def get_datasets():

  train, test = ds_common.load_dataset(FLAGS.dataset, FLAGS.train_name,
                                       FLAGS.test_name)

  train_normal, train_anomalous = split_normal_anomalous(train)
  test_normal, test_anomalous = split_normal_anomalous(test)

  # dimensionality of the input data
  x_dim = train.element_spec[0].shape[-1]
  transforms = transform.get_transforms(x_dim)

  @tf.function
  def apply_transforms(x, y):
    del y  # unused parameter
    xx, yy = transform.apply_transforms(transforms, x)
    return tf.data.Dataset.from_tensor_slices((xx, yy))

  def prepare_dataset(ds):
    ds = ds.flat_map(apply_transforms)
    if FLAGS.cache.lower() != 'none':
      ds = ds.cache(FLAGS.cache)

    # Perfectly balanced batch
    batch_size = FLAGS.batch_size * len(transforms)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

  return (
    len(transforms),
    prepare_dataset(train_normal),
    prepare_dataset(train_anomalous),
    prepare_dataset(test_normal),
    prepare_dataset(test_anomalous),
  )
