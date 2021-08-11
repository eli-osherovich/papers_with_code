import gin
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .. import data, model


def _train_model(
  X_train, y_train, model_, X_val, y_val, epochs, batch_size, shuffle_buf_size,
  scale_pos_weight, callbacks, verbose
):
  X_mean = X_train.mean(axis=0)
  X_std = X_train.std(axis=0)

  X_train = (X_train - X_mean) / X_std
  X_val = (X_val - X_mean) / X_std

  train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
  ).shuffle(shuffle_buf_size).batch(batch_size)

  val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
  class_weight = {0: 1.0, 1: scale_pos_weight}

  model_.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=verbose
  )

  return model_


@gin.configurable
def train(
  X,
  y,
  epochs: int,
  test_size: float,
  batch_size: int,
  patience: int,
  shuffle_buf_size: int,
  random_state: int,
  keep_last: bool,
  scale_pos_weight=float,
):

  X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    stratify=y,
    test_size=test_size,
    random_state=random_state,
  )

  m = model.get_tree_model()

  early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    patience=patience,
    mode='max',
    restore_best_weights=not keep_last,
  )

  m.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['acc', tf.keras.metrics.AUC(from_logits=True)]
  )
  m = _train_model(
    X_train,
    y_train,
    m,
    X_val,
    y_val,
    epochs,
    batch_size,
    shuffle_buf_size,
    scale_pos_weight,
    [early_stop],
    verbose=2,
  )
  m.save('saved_model/tree_model')
