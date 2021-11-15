"""Training/tuning routines for TabNet.

"""
import gin
from sklearn import model_selection
import tensorflow as tf

from .. import models


def _train_model(
  model_, train_ds, *, val_ds=None, callbacks=None, fit_params=None
):
  fit_params = fit_params or {}
  model_.fit(
    train_ds, validation_data=val_ds, callbacks=callbacks, **fit_params
  )
  return model_


@gin.configurable
def train(data):
  train_ds, val_ds = get_ds(data)
  fit_params = {"epochs": 100000, "validation_freq": 1}

  model_ = models.get_model()
  model_ = _train_model(model_, train_ds, val_ds=val_ds, fit_params=fit_params)
  return model_.evaluate(val_ds)


@gin.configurable
def train_cv(X, y):
  raise NotImplementedError()


@gin.configurable
def tune(X, y):
  raise NotImplementedError()


@gin.configurable
def get_ds(data, test_size=0.25, shuffle_buffer=16384, batch_size=4096):
  train_data = data["train"]
  if "val" in data:
    val_data = data["val"]
  else:
    x_train, x_val, y_train, y_val = model_selection.train_test_split(
      *train_data, test_size=test_size, stratify=train_data[1]
    )
    train_data = (x_train, y_train)
    val_data = (x_val, y_val)

  train_ds = tf.data.Dataset.from_tensor_slices(train_data)\
      .shuffle(shuffle_buffer)\
      .batch(batch_size, drop_remainder=True)
  val_ds = tf.data.Dataset.from_tensor_slices(val_data)\
    .shuffle(shuffle_buffer)\
    .batch(batch_size, drop_remainder=True)
  return train_ds, val_ds
