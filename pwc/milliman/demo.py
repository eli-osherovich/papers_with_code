#!/usr/bin/env python3

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from pwc.milliman import training

FLAGS = flags.FLAGS

SEED = 42


def main(argv):
  del argv  # unused parameter
  tf.random.set_seed(SEED)
  np.random.seed(SEED)

  res = training.train()
  print(res)


if __name__ == "__main__":
  app.run(main)
