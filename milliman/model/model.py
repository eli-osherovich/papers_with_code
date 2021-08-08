import enum

from .tree_model import get_tree_model
from .xgb_model import get_xgb_model


@enum.unique
class MODEL(enum.Enum):
  XGB = 1
  TREE = 2


def get_model(model: MODEL, **kwargs):
  if model == MODEL.XGB:
    return get_xgb_model(**kwargs)
  elif model == MODEL.TREE:
    return get_tree_model(**kwargs)
