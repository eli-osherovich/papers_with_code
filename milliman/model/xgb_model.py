import xgboost as xgb


def get_xgb_model(**kwargs):
  return xgb.XGBClassifier(use_label_encoder=False, **kwargs)
