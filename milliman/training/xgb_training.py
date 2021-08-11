import gin
import numpy as np
import ray.tune
import ray.tune.suggest.hyperopt
from sklearn.model_selection import (
  RepeatedStratifiedKFold, cross_validate, train_test_split
)

from ...common import utils
from .. import model

gin.external_configurable(ray.tune.suggest.hyperopt.HyperOptSearch)


def _train_model(
  X_train, y_train, model_, *, eval_set, eval_metric, patience, verbose
):
  return model_.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    eval_metric=eval_metric,
    early_stopping_rounds=patience,
    verbose=verbose,
  )


@gin.configurable
def train(
  X, y, *, test_size: float, eval_metric: str, patience: int, random_state: int,
  verbose: int, **model_args
):
  m = model.get_xgb_model(**model_args)
  X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    stratify=y,
    test_size=test_size,
    random_state=random_state,
  )
  m = _train_model(
    X_train,
    y_train,
    m,
    eval_set=[(X_val, y_val)],
    eval_metric=eval_metric,
    patience=patience,
    verbose=verbose
  )
  return m.best_score


@gin.configurable
def train_cv(
  X, y, *, k_folds: int, n_repeats: int, eval_metric: str, patience: int,
  random_state: int, verbose: int, scoring, **model_args
):

  m = model.get_xgb_model(**model_args)
  cv = RepeatedStratifiedKFold(
    n_splits=k_folds, n_repeats=n_repeats, random_state=random_state
  )

  cv_res = cross_validate(
    m,
    X,
    y,
    cv=cv,
    scoring=scoring,
    fit_params={
      'eval_metric': eval_metric,
      'early_stopping_rounds': patience,
      'verbose': verbose,
    }
  )
  return cv_res


@gin.configurable
def gen_search_space(
  *,
  n_samples,
  n_features,
  colsample_bytree_bounds,
  learning_rate_bounds,
  max_depth_bounds,
  min_child_weight_bounds,
  scale_pos_weight_bounds,
  subsample_bounds,
):
  q_features = 1 / n_features
  q_samples = 1 / n_samples

  colsample_bytree_bounds = utils.make_divisible(
    q_features, colsample_bytree_bounds
  )

  subsample_bounds = utils.make_divisible(q_samples, subsample_bounds)

  return {
    'colsample_bytree': ray.tune.quniform(*colsample_bytree_bounds, q_features),
    'learning_rate': ray.tune.loguniform(*learning_rate_bounds),
    'max_depth': ray.tune.randint(*max_depth_bounds),
    'min_child_weight': ray.tune.uniform(*min_child_weight_bounds),
    'scale_pos_weight': ray.tune.uniform(*scale_pos_weight_bounds),
    'subsample': ray.tune.quniform(*subsample_bounds, q_samples),
  }


@gin.configurable
def tune(X, y, *, metric, mode, num_samples, search_alg, cv_params):

  k_folds = cv_params['k_folds']
  n_samples, n_features = X.shape
  train_samples = round(n_samples / k_folds * (k_folds - 1))
  config = gen_search_space(n_samples=train_samples, n_features=n_features)

  def trainable(model_args):
    res = train_cv(X, y, **cv_params, **model_args)

    agg_res = {}
    for k, v in res.items():
      agg_res[k + '_mean'] = np.mean(v)
      agg_res[k + '_std'] = np.std(v)
      agg_res[k + '_min'] = np.min(v)
      agg_res[k + '_max'] = np.max(v)
    return agg_res

  return ray.tune.run(
    trainable,
    config=config,
    metric=metric,
    mode=mode,
    num_samples=num_samples,
    search_alg=search_alg(),
    fail_fast=True,
  )
