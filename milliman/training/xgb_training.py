import pathlib

import gin
from sklearn.model_selection import train_test_split

from .. import data, model

GIN_CONFIG_FILE = 'xgb_training_config.gin'


@gin.configurable
def xgb_train(
  colsample_bytree: float,
  learning_rate: float,
  max_depth: int,
  min_child_weight: int,
  scale_pos_weight: float,
  subsample: float,
  random_state: int,
  test_size: float,
  n_estimators: int,
  patience: int,
):

  X, y = data.get_numpy()
  X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    stratify=y,
    test_size=test_size,
    random_state=random_state,
  )

  m = model.get_xgb_model()
  m.set_params(
    n_estimators=n_estimators,
    objective='binary:logistic',
    colsample_bytree=colsample_bytree,
    learning_rate=learning_rate,
    max_depth=max_depth,
    min_child_weight=min_child_weight,
    scale_pos_weight=scale_pos_weight,
    subsample=subsample,
  )

  m.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric='auc',
    early_stopping_rounds=patience,
    verbose=True,
  )


gin.parse_config_file(pathlib.Path(__file__).parent.resolve() / GIN_CONFIG_FILE)
