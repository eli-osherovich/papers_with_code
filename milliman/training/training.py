from absl import flags

from .. import data, model
from . import metatree_training, tree_training, xgb_training

FLAGS = flags.FLAGS

flags.DEFINE_enum('action', 'train', ['train', 'train_cv', 'tune'],
                  'Action to perform')


def _get_action():
  if FLAGS.action == 'train':
    return 'train'
  elif FLAGS.action == 'train_cv':
    return 'train_cv'
  elif FLAGS.action == 'tune':
    return 'tune'
  else:
    raise RuntimeError('Unknown action: %s', FLAGS.action)


def _get_model_module():
  if FLAGS.model == model.MODEL.TREE:
    return tree_training
  elif FLAGS.model == model.MODEL.XGB:
    return xgb_training
  elif FLAGS.model == model.MODEL.METATREE:
    return metatree_training
  else:
    raise RuntimeError('Unknown model: %s', FLAGS.model)


def train():
  X, y = data.get_numpy()

  model_module = _get_model_module()
  action = _get_action()
  train_fn = getattr(model_module, action)
  return train_fn(X, y)
