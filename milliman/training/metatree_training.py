import gin
import numpy as np
import ray.tune
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from .. import model


def _train_model(
  train_ds, model_, *, class_weight, eval_ds, callbacks, fit_params
):
  model_.fit(
    train_ds,
    validation_data=eval_ds,
    callbacks=callbacks,
    class_weight=class_weight,
    **fit_params,
  )

  return model_


@gin.configurable
def train(
  X,
  y,
  *,
  test_size: float,
  random_state: int,
  fit_params: dict,
  batch_size,
  **model_args,
):

  X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    stratify=y,
    test_size=test_size,
    random_state=random_state,
  )
  train_ds, eval_ds = _prepare_datasets(
    X_train, y_train, X_val, y_val, batch_size
  )

  early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_acc",
    patience=fit_params.pop("patience"),
    restore_best_weights=True,
  )

  scale_pos_weight = fit_params.pop("scale_pos_weight")
  class_weight = {0: 1.0, 1: scale_pos_weight}

  m = model.get_model(model.MODEL.METATREE, **model_args)

  m = _train_model(
    train_ds,
    m,
    class_weight=class_weight,
    eval_ds=eval_ds,
    callbacks=[early_stop],
    fit_params=fit_params,
  )
  print_tree(m, eval_ds)
  return m.evaluate(eval_ds, return_dict=True)


@gin.configurable
def train_cv(
  X, y, *, cv_params: dict, fit_params: dict, batch_size, **model_args
):
  early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_acc",
    patience=fit_params.pop("patience"),
    restore_best_weights=True,
  )

  scale_pos_weight = fit_params.pop("scale_pos_weight")
  class_weight = {0: 1.0, 1: scale_pos_weight}

  cv = RepeatedStratifiedKFold(**cv_params)
  results = []
  for train_index, val_index in cv.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    train_ds, eval_ds = _prepare_datasets(
      X_train, y_train, X_val, y_val, batch_size
    )

    m = model.get_model(model.MODEL.METATREE, **model_args)

    m = _train_model(
      train_ds,
      m,
      class_weight=class_weight,
      eval_ds=eval_ds,
      callbacks=[early_stop],
      fit_params=fit_params,
    )
    results.append(m.evaluate(eval_ds, return_dict=True))
  return tf.nest.map_structure(lambda *x: x, *results)


@gin.configurable
def gen_search_space(*, depth_bounds, batch_size_list, scale_pos_weight_bounds):
  return {
    "depth": ray.tune.randint(*depth_bounds),
    "batch_size": ray.tune.choice(batch_size_list),
    "scale_pos_weight": ray.tune.uniform(*scale_pos_weight_bounds),
  }


@gin.configurable
def tune(
  X, y, *, metric, mode, num_samples, search_alg, num_cpus, cv_params,
  fit_params
):
  config = gen_search_space()

  def trainable(config):
    model_args = {"depth": config.pop("depth")}
    trial_fit_params = fit_params | config
    res = train_cv(
      X, y, cv_params=cv_params, fit_params=trial_fit_params, **model_args
    )

    agg_res = {}
    for k, v in res.items():
      agg_res[k + "_mean"] = np.mean(v)
      agg_res[k + "_std"] = np.std(v)
      agg_res[k + "_min"] = np.min(v)
      agg_res[k + "_max"] = np.max(v)
    return agg_res

  ray.init(num_cpus=num_cpus)

  return ray.tune.run(
    trainable,
    config=config,
    num_samples=num_samples,
    search_alg=search_alg(metric=metric, mode=mode),
    max_failures=2,
  )


def _prepare_datasets(X_train, y_train, X_val, y_val, batch_size):
  pt = StandardScaler()
  pt.fit(X_train)

  print(f"scale={pt.scale_.tolist()}")
  print(f"mean={pt.mean_.tolist()}")

  X_train = pt.transform(X_train)
  X_val = pt.transform(X_val)

  train_ds = (
    tf.data.Dataset.from_tensor_slices(
      (X_train, y_train)
    ).shuffle(1000).batch(batch_size, drop_remainder=True)
  )
  eval_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(
    len(X_val), drop_remainder=True
  )

  return train_ds, eval_ds


def print_tree(model, ds):
  import json

  model.run_eagerly = True
  model.predict(ds)

  nodes = []

  def _add_node(node):
    res = {"id": node.id}
    for a in [
      "value",
      "x",
      "b",
      "w",
      "beta",
      "threshold",
      "split_feature_idx",
      "geq",
      "proba_right",
      "ww",
    ]:
      if hasattr(node, a):
        try:
          res[a] = getattr(node, a).numpy().tolist()
        except AttributeError:
          res[a] = getattr(node, a)
    nodes.append(res)
    if hasattr(node, "left"):
      _add_node(node.left)
    if hasattr(node, "right"):
      _add_node(node.right)

  with open("/tmp/tree.json", "w") as f:
    _add_node(model.tree)
    json.dump(nodes, f)
