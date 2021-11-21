#!/usr/bin/env python3

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from pwc.common import utils
from pwc.{{cookiecutter.project|lower|trim|replace(" ", "_")|replace("-", "_")}} import training

FLAGS = flags.FLAGS

_SEED = 42


def main(argv):
  del argv  # unused parameter
  tf.random.set_seed(_SEED)
  np.random.seed(_SEED)

  res = training.training_fn()
  print(res)


if __name__ == "__main__":
  utils.setup_omp()
  app.run(main)
