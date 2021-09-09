from absl import flags
import tensorflow as tf

from .. import data
from .. import model

FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 100, 'Number of training epochs', lower_bound=1)
flags.DEFINE_enum('keep', 'best', ['last', 'best'], 'Which model to keep.')
flags.DEFINE_integer('patience', 10, 'Early stopping patience.', lower_bound=0)
flags.DEFINE_string('monitor', 'val_accuracy',
                    'Early stopping quantity to monitor.')
flags.DEFINE_string('save_dir', './saved_model', 'Model save directory.')


def train():
  (n_transforms, train_normal, train_anomalous, test_normal,
   test_anomalous) = data.get_datasets()

  early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor=FLAGS.monitor,
    patience=FLAGS.patience,
    restore_best_weights=FLAGS.keep == 'best')

  m = model.get_model(n_transforms)

  m.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
  )

  m.fit(
    train_normal,
    epochs=FLAGS.epochs,
    validation_data=test_normal,
    callbacks=[early_stopping_cb])

  m.save(FLAGS.save_dir)
  return m
