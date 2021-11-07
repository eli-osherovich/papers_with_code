#!/usr/bin/env python3

# Python's relative import was invented by pure PERVERTS.
if __name__ == "__main__" and __package__ is None:
  import os
  import pathlib
  import sys

  module_path = pathlib.Path(__file__).parent.resolve()
  module_name = module_path.name
  pkg_path = module_path.parent
  pkg_name = pkg_path.name
  sys.path.append(pkg_path.parent.as_posix())
  os.environ["PYTHONPATH"] = pkg_path.parent.as_posix()
  __package__ = f"{pkg_name}.{module_name}"  # noqa: A001

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from . import training

FLAGS = flags.FLAGS

_SEED = 42


def main(argv):
  del argv  # unused parameter
  tf.random.set_seed(_SEED)
  np.random.seed(_SEED)

  res = training.training_fn()
  print(res)


if __name__ == "__main__":
  app.run(main)
