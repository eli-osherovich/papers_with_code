#!/usr/bin/env python3

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from pwc.arxiv_1908_07442 import training
from pwc.common import utils

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
