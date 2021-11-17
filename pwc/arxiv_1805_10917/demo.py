#!/usr/bin/env python3

from absl import app

from pwc.arxiv_1805_10917 import training


def main(argv):
  del argv  # unused parameter
  model = training.train()
  del model  # unused


if __name__ == "__main__":
  app.run(main)
