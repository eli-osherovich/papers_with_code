import pathlib

import gin
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from .. import model

GIN_CONFIG_FILE = 'xgb_training_config.gin'


def _train_model(X_train, y_train, model_, X_val, y_val, patience, verbose):
  return model_.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric='auc',
    early_stopping_rounds=patience,
    verbose=verbose,
  )


@gin.configurable
def xgb_train(
  X, y, *, test_size: float, patience: int, random_state: int, **model_args
):
  m = model.get_xgb_model(**model_args)
  X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    stratify=y,
    test_size=test_size,
    random_state=random_state,
  )
  return _train_model(X_train, y_train, m, X_val, y_val, patience, verbose=True)


@gin.configurable
def xgb_train_cv(
  X, y, *, k_folds: int, n_repeats: int, patience: int, random_state: int,
  **model_args
):

  m = model.get_xgb_model(**model_args)
  cv = RepeatedStratifiedKFold(
    n_splits=k_folds, n_repeats=n_repeats, random_state=random_state
  )
  results = []
  for train_index, test_index in cv.split(X, y, y):
    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = y[train_index], y[test_index]
    m = _train_model(X_train, y_train, m, X_val, y_val, patience, verbose=False)
    results.append(m.best_score)
  return results


def xgb_tune():
  pass


gin.parse_config_file(pathlib.Path(__file__).parent.resolve() / GIN_CONFIG_FILE)
