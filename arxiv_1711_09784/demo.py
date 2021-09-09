#!/usr/bin/env python3

# Python's relative import was invented by pure PERVERTS.
if __name__ == '__main__' and __package__ is None:
  import pathlib
  import sys
  module_path = pathlib.Path(__file__).parent.resolve()
  module_name = module_path.name
  pkg_path = module_path.parent
  pkg_name = pkg_path.name
  sys.path.append(pkg_path.parent.as_posix())
  __package__ = f'{pkg_name}.{module_name}'  # noqa: A001

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_datasets as tfds

from . import tree
from ..common import utils

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'mnist', 'Dataset to load')
flags.DEFINE_integer(
  'depth', 9, 'Tree depth (including root and leaves)', lower_bound=2)
flags.DEFINE_integer('epochs', 1000, 'Number of epochs', lower_bound=0)
flags.DEFINE_integer('batch_size', 1024, 'Batch size', lower_bound=1)
flags.DEFINE_integer('shuffle_buffer', 60000, 'Shuffle buffer size')
flags.DEFINE_integer('patience', 10, 'Early stopping patience')
flags.DEFINE_bool('keep_last', False,
                  'Keep last model (default is to keep the best model)')


def main(argv):
  (train_ds, val_ds), info = tfds.load(
    FLAGS.dataset,
    split=['train', 'test'],
    shuffle_files=False,
    as_supervised=True,
    with_info=True,
  )

  train_ds = train_ds\
    .map(lambda x, y: (x / 255, y), num_parallel_calls=tf.data.AUTOTUNE)\
    .cache()\
    .shuffle(FLAGS.shuffle_buffer)\
    .batch(FLAGS.batch_size)\
    .prefetch(tf.data.AUTOTUNE)

  val_ds = val_ds\
    .map(lambda x, y: (x / 255, y), num_parallel_calls=tf.data.AUTOTUNE)\
    .batch(FLAGS.batch_size)\
    .cache()\
    .prefetch(tf.data.AUTOTUNE)

  n_classes = info.features[info.supervised_keys[-1]].num_classes
  model = tree.TreeModel(n_classes=n_classes, depth=FLAGS.depth)

  early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_acc',
    patience=FLAGS.patience,
    restore_best_weights=not FLAGS.keep_last)

  model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['acc'])
  model.fit(
    train_ds,
    epochs=FLAGS.epochs,
    validation_data=val_ds,
    callbacks=[early_stop])
  model.save('model')


if __name__ == '__main__':
  utils.setup_omp()
  app.run(main)
