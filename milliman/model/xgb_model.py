import gin
import xgboost as xgb


@gin.configurable
def get_model(
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
  gamma,
  max_delta_step,
  reg_lambda,
  reg_alpha,
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
    gamma=gamma,
    max_delta_step=max_delta_step,
    reg_lambda=reg_lambda,
    reg_alpha=reg_alpha,
    n_jobs=n_jobs,
    use_label_encoder=False,
  )
