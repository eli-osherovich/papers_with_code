import gin
import numpy as np
import ray.tune
from sklearn.model_selection import (
  RepeatedStratifiedKFold, cross_validate, train_test_split
)

from .. import model


def _train_model(
  X_train, y_train, model_, *, eval_set, patience, eval_metric, verbose
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
      'verbose': verbose,
      'eval_metric': eval_metric,
      'early_stopping_rounds': patience,
    }
  )
  return cv_res
