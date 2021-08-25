import gin
import numpy as np
import ray.tune
import tensorflow as tf
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .. import model


def _train_model(train_ds, model_, *, class_weight, eval_ds, callbacks,
                 fit_params):
  model_.fit(
    train_ds,
    validation_data=eval_ds,
    callbacks=callbacks,
    class_weight=class_weight,
    **fit_params,
  )

  return model_


@gin.configurable
def train(X, y, *, test_size: float, random_state: int, fit_params: dict,
          **model_args):

  m = model.get_model(model.MODEL.METATREE, **model_args)
  early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_acc',
    patience=fit_params.pop('patience'),
    restore_best_weights=True,
  )

  m.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['acc'],
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5))

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

  train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)).shuffle(1000).batch(
      128, drop_remainder=True)
  eval_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(
    len(X_val), drop_remainder=True)
  m = _train_model(
    train_ds,
    m,
    class_weight=class_weight,
    eval_ds=eval_ds,
    callbacks=[early_stop],
    fit_params=fit_params,
  )


@gin.configurable
def train_cv(X, y, *, cv_params: dict, fit_params: dict, **model_args):
  patience = fit_params.pop('patience')
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

    m = model.get_model(model.MODEL.METATREE, **model_args)
    early_stop = tf.keras.callbacks.EarlyStopping(
      monitor='val_accuracy',
      patience=patience,
      restore_best_weights=True,
    )
    m.compile(
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
      # Accuracy's threshold is set to 0 (instead of 0.5) because our model
      # returns logits.
      metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0)])

    m = _train_model(
      X_train,
      y_train,
      m,
      class_weight=class_weight,
      eval_set=(X_val, y_val),
      callbacks=[early_stop],
      fit_params=fit_params,
    )
    pred = m.predict(X_val)
    prob = tf.math.sigmoid(pred)
    score = np.mean(
      tf.keras.metrics.binary_accuracy(y_val[:, np.newaxis], prob))
    scores.append(score)
  return {'accuracy': scores}


@gin.configurable
def gen_search_space(*, depth_bounds, batch_size_list, scale_pos_weight_bounds):
  return {
    'depth': ray.tune.randint(*depth_bounds),
    'batch_size': ray.tune.choice(batch_size_list),
    'scale_pos_weight': ray.tune.uniform(*scale_pos_weight_bounds),
  }


@gin.configurable
def tune(X, y, *, metric, mode, num_samples, search_alg, num_cpus, cv_params,
         fit_params):
  config = gen_search_space()

  def trainable(config):
    model_args = {'depth': config.pop('depth')}
    trial_fit_params = fit_params | config
    res = train_cv(
      X, y, cv_params=cv_params, fit_params=trial_fit_params, **model_args)

    agg_res = {}
    for k, v in res.items():
      agg_res[k + '_mean'] = np.mean(v)
      agg_res[k + '_std'] = np.std(v)
      agg_res[k + '_min'] = np.min(v)
      agg_res[k + '_max'] = np.max(v)
    return agg_res

  ray.init(num_cpus=num_cpus)

  return ray.tune.run(
    trainable,
    config=config,
    num_samples=num_samples,
    search_alg=search_alg(metric=metric, mode=mode),
    max_failures=2,
  )
