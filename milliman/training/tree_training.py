import pathlib

import gin
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .. import data, model

GIN_CONFIG_FILE = 'tree_training_config.gin'


@gin.configurable
def tree_train(
  depth: int,
  epochs: int,
  train_size: float,
  batch_size: int,
  patience: int,
  shuffle_buf_size: int,
  random_state: int,
  keep_last: bool,
  scale_pos_weight=float,
):

  X, y = data.get_numpy()
  X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    stratify=y,
    train_size=train_size,
    random_state=random_state,
  )
  X_mean = X_train.mean(axis=0)
  X_std = X_train.std(axis=0)

  X_train = (X_train - X_mean) / X_std
  X_val = (X_val - X_mean) / X_std

  train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
  ).shuffle(shuffle_buf_size).batch(batch_size)

  val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

  m = model.get_tree_model(depth)

  early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    patience=patience,
    mode='max',
    restore_best_weights=not keep_last,
  )

  class_weight = {0: 1.0, 1: scale_pos_weight}
  m.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['acc', tf.keras.metrics.AUC(from_logits=True)]
  )
  m.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=[early_stop],
    class_weight=class_weight,
  )
  m.save('saved_model/tree_model')


gin.parse_config_file(pathlib.Path(__file__).parent.resolve() / GIN_CONFIG_FILE)
