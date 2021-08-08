from absl import flags

from .. import model
from . import tree_training, xgb_training

FLAGS = flags.FLAGS

flags.DEFINE_enum_class(
  'model', model.MODEL.TREE, model.MODEL, 'Model to train'
)


def train():
  if FLAGS.model == model.MODEL.TREE:
    return tree_training.tree_train()
  elif FLAGS.model == model.MODEL.XGB:
    return xgb_training.xgb_train()
