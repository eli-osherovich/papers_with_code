from absl import flags

from .. import data, model
from . import tree_training, xgb_training

FLAGS = flags.FLAGS

flags.DEFINE_enum_class(
  'model', model.MODEL.TREE, model.MODEL, 'Model to train'
)

flags.DEFINE_enum('action', 'train', ['train', 'train_cv'], 'Action to perform')


def _get_action():
  if FLAGS.action == 'train':
    return 'train'
  elif FLAGS.action == 'train_cv':
    return 'train_cv'
  else:
    raise RuntimeError('Unknown action: %s', FLAGS.action)


def _get_model_module():
  if FLAGS.model == model.MODEL.TREE:
    return tree_training
  elif FLAGS.model == model.MODEL.XGB:
    return xgb_training
  else:
    raise RuntimeError('Unknown model: %s', FLAGS.model)


def train():
  X, y = data.get_numpy()

  model_module = _get_model_module()
  action = _get_action()
  train_fn = getattr(model_module, action)
  return train_fn(X, y)
