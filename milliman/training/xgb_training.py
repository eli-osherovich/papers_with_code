import gin
import numpy as np
import ray.tune
import ray.tune.suggest.optuna
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split

from .. import model
from ...common import utils

gin.external_configurable(ray.tune.suggest.optuna.OptunaSearch)


def _train_model(X_train, y_train, model_, *, eval_set, fit_params: dict):
  return model_.fit(X_train, y_train, eval_set=eval_set, **fit_params)


@gin.configurable
def train(X, y, *, test_size: float, random_state: int, fit_params: dict,
          **model_args):
  m = model.get_model(model.MODEL.XGB, **model_args)
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
    eval_set=[(X_train, y_train), (X_val, y_val)],
    fit_params=fit_params)
  return m.best_score


@gin.configurable
def train_cv(X, y, *, scoring, cv_params: dict, fit_params: dict, **model_args):

  m = model.get_model(model.MODEL.XGB, **model_args)
  cv = RepeatedStratifiedKFold(**cv_params)
  cv_res = cross_validate(
    m, X, y, cv=cv, scoring=scoring, fit_params=fit_params)
  return cv_res


@gin.configurable
def gen_search_space(*, n_samples, n_features, colsample_bytree_bounds,
                     learning_rate_bounds, max_depth_bounds,
                     min_child_weight_bounds, scale_pos_weight_bounds,
                     subsample_bounds, gamma_bounds, max_delta_step_bounds,
                     reg_lambda_bounds, reg_alpha_bounds):
  q_features = 1 / n_features
  q_samples = 1 / n_samples

  colsample_bytree_bounds = utils.make_divisible(q_features,
                                                 colsample_bytree_bounds)

  subsample_bounds = utils.make_divisible(q_samples, subsample_bounds)

  return {
    'colsample_bytree': ray.tune.quniform(*colsample_bytree_bounds, q_features),
    'learning_rate': ray.tune.loguniform(*learning_rate_bounds),
    'max_depth': ray.tune.randint(*max_depth_bounds),
    'min_child_weight': ray.tune.uniform(*min_child_weight_bounds),
    'scale_pos_weight': ray.tune.uniform(*scale_pos_weight_bounds),
    'subsample': ray.tune.quniform(*subsample_bounds, q_samples),
    'gamma': ray.tune.uniform(*gamma_bounds),
    'max_delta_step': ray.tune.uniform(*max_delta_step_bounds),
    'reg_lambda': ray.tune.uniform(*reg_lambda_bounds),
    'reg_alpha': ray.tune.uniform(*reg_alpha_bounds),
  }


@gin.configurable
def tune(X, y, *, metric, mode, num_samples, search_alg, scoring: str,
         cv_params: dict, fit_params: dict):

  n_splits = cv_params['n_splits']
  n_samples, n_features = X.shape
  train_samples = round(n_samples / n_splits * (n_splits - 1))
  config = gen_search_space(n_samples=train_samples, n_features=n_features)

  def trainable(model_args):
    res = train_cv(
      X,
      y,
      scoring=scoring,
      cv_params=cv_params,
      fit_params=fit_params,
      **model_args)

    # do not save unused parameters.
    res.pop('fit_time')
    res.pop('score_time')

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
    num_samples=num_samples,
    search_alg=search_alg(metric=metric, mode=mode),
    fail_fast=True,
  )
