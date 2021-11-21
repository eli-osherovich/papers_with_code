import json

from absl import flags
import numpy as np

from . import metatree_training
from . import tree_training
from . import xgb_training
from .. import data
from .. import model

FLAGS = flags.FLAGS

flags.DEFINE_enum(
  "action", "train", ["train", "train_cv", "tune"], "Action to perform"
)


def _get_action():
  if FLAGS.action == "train":
    return "train"
  elif FLAGS.action == "train_cv":
    return "train_cv"
  elif FLAGS.action == "tune":
    return "tune"
  else:
    raise RuntimeError("Unknown action: %s" % FLAGS.action)


def _get_model_training_module():
  if FLAGS.model == model.MODEL.TREE:
    return tree_training
  elif FLAGS.model == model.MODEL.XGB:
    return xgb_training
  elif FLAGS.model == model.MODEL.METATREE:
    return metatree_training
  else:
    raise RuntimeError("Unknown model: %s" % FLAGS.model)


def train():
  X, y = data.get_dataframe()
  ds_data = {}
  ds_data["cols"] = X.columns.tolist()
  with open("/tmp/dataset_columns.json", "w") as f:
    json.dump(ds_data, f, indent=2)

  X.to_numpy(dtype=np.float32)
  y = y.to_numpy(dtype=np.float32)

  model_module = _get_model_training_module()
  action = _get_action()
  train_fn = getattr(model_module, action)
  return train_fn(X, y)