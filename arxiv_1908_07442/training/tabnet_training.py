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
def train(
  X,
  y,
  *,
  batch_size: int,
  test_size: float,
):
  x_train, x_val, y_train, y_val = model_selection.train_test_split(
    X, y, test_size=test_size
  )
  train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(
    batch_size, drop_remainder=True
  )
  val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(
    batch_size, drop_remainder=True
  )
  fit_params = {"epochs": 100}

  model_ = models.get_model()
  model_ = _train_model(model_, train_ds, val_ds=val_ds, fit_params=fit_params)
  return model_.evaluate(val_ds)


@gin.configurable
def train_cv(X, y):
  raise NotImplementedError()


@gin.configurable
def tune(X, y):
  raise NotImplementedError()
