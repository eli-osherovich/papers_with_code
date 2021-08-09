import pathlib

import gin
from sklearn.model_selection import train_test_split

from .. import data, model

GIN_CONFIG_FILE = 'xgb_training_config.gin'


@gin.configurable
def xgb_train(random_state: int, test_size: float, patience: int, **kwargs):

  X, y = data.get_numpy()
  X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    stratify=y,
    test_size=test_size,
    random_state=random_state,
  )

  m = model.get_xgb_model(**kwargs)
  m.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric='auc',
    early_stopping_rounds=patience,
    verbose=True,
  )


gin.parse_config_file(pathlib.Path(__file__).parent.resolve() / GIN_CONFIG_FILE)
