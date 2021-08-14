import gin
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from .. import model


def _train_model(
  X_train, y_train, model_, *, class_weight, eval_set, callbacks, fit_params
):
  X_val, y_val = eval_set
  model_.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    class_weight=class_weight,
    **fit_params,
  )

  return model_


@gin.configurable
def train(
  X, y, *, test_size: float, random_state: int, fit_params: dict, **model_args
):

  m = model.get_model(model.MODEL.TREE, **model_args)
  early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_acc',
    patience=fit_params.pop('patience'),
    restore_best_weights=True,
  )

  m.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['acc']
  )

  X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    stratify=y,
    test_size=test_size,
    random_state=random_state,
  )
  scale_pos_weight = fit_params.pop('scale_pos_weight')
  class_weight = {0: 1.0, 1: scale_pos_weight}

  pt = StandardScaler()
  X_train = pt.fit_transform(X_train)
  X_val = pt.transform(X_val)

  m = _train_model(
    X_train,
    y_train,
    m,
    class_weight=class_weight,
    eval_set=(X_val, y_val),
    callbacks=[early_stop],
    fit_params=fit_params,
  )


@gin.configurable
def train_cv(X, y, *, cv_params: dict, fit_params: dict, **model_args):
  m = model.get_model(model.MODEL.TREE, **model_args)
  early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=fit_params.pop('patience'),
    restore_best_weights=True,
  )
  m.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    # Accuracy's threshold is set to 0 (instead of 0.5) because our model
    # returns logits.
    metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0)]
  )
  scale_pos_weight = fit_params.pop('scale_pos_weight')
  class_weight = {0: 1.0, 1: scale_pos_weight}

  pt = StandardScaler()

  cv = RepeatedStratifiedKFold(**cv_params)
  scores = []
  for train_index, val_index in cv.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    X_train = pt.fit_transform(X_train)
    X_val = pt.transform(X_val)
    m = _train_model(
      X_train,
      y_train,
      m,
      class_weight=class_weight,
      eval_set=(X_val, y_val),
      callbacks=[early_stop],
      fit_params=fit_params,
    )
    pred = m.predict(X_val, batch_size=200)
    prob = tf.math.sigmoid(pred)
    score = np.mean(
      tf.keras.metrics.binary_accuracy(y_val[:, np.newaxis], prob)
    )
    scores.append(score)
  print(scores)
  return scores
