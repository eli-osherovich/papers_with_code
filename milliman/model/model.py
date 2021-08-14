import enum
from absl import flags

from .tree_model import get_tree_model
from .xgb_model import get_xgb_model

FLAGS = flags.FLAGS


@enum.unique
class MODEL(enum.Enum):
  XGB = 1
  TREE = 2


flags.DEFINE_enum_class('model', MODEL.TREE, MODEL, 'Model to train')


def get_model(model: MODEL, **model_args):
  if model == MODEL.XGB:
    return get_xgb_model(**model_args)
  elif model == MODEL.TREE:
    return get_tree_model(**model_args)
