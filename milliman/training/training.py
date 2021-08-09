from absl import flags

from .. import data, model
from . import tree_training, xgb_training

FLAGS = flags.FLAGS

flags.DEFINE_enum_class(
  'model', model.MODEL.TREE, model.MODEL, 'Model to train'
)


def train():
  X, y = data.get_numpy()
  if FLAGS.model == model.MODEL.TREE:
    return tree_training.tree_train(X, y)
  elif FLAGS.model == model.MODEL.XGB:
    return xgb_training.xgb_train(X, y)
