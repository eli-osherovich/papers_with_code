import enum

from absl import flags

from . import metatree_model, tree_model, xgb_model

FLAGS = flags.FLAGS


@enum.unique
class MODEL(enum.Enum):
  XGB = 1
  TREE = 2
  METATREE = 3


flags.DEFINE_enum_class('model', MODEL.TREE, MODEL, 'Model to train')


def get_model(model: MODEL, **model_args):
  if model == MODEL.XGB:
    return xgb_model.get_model(**model_args)
  elif model == MODEL.TREE:
    return tree_model.get_model(**model_args)
  elif model == MODEL.METATREE:
    return metatree_model.get_model(**model_args)
