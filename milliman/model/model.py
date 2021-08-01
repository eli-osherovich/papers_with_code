import xgboost as xgb

xgb.set_config(verbosity=2)


def get_model():
  return xgb.XGBClassifier(use_label_encoder=False)
