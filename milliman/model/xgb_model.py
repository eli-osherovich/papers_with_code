import pathlib

import gin
import xgboost as xgb

GIN_CONFIG_FILE = 'xgb_model_config.gin'


@gin.configurable
def get_xgb_model(
  *,
  n_estimators,
  objective,
  colsample_bytree,
  learning_rate,
  max_depth,
  min_child_weight,
  scale_pos_weight,
  subsample,
  random_state,
  n_jobs,
):
  return xgb.XGBClassifier(
    n_estimators=n_estimators,
    objective=objective,
    colsample_bytree=colsample_bytree,
    learning_rate=learning_rate,
    max_depth=max_depth,
    min_child_weight=min_child_weight,
    scale_pos_weight=scale_pos_weight,
    subsample=subsample,
    random_state=random_state,
    n_jobs=n_jobs,
    use_label_encoder=False,
  )


gin.parse_config_file(pathlib.Path(__file__).parent.resolve() / GIN_CONFIG_FILE)
