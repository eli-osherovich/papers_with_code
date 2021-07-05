#!/usr/bin/env python3

# Python's relative import was invented by pure PERVERTS.
if __name__ == "__main__" and __package__ is None:
  import os
  import sys
  module_path = os.path.abspath(os.path.join(__file__, ".."))
  module_name = os.path.basename(module_path)
  pkg_path = os.path.abspath(os.path.join(__file__, "..", ".."))
  pkg_name = os.path.basename(pkg_path)
  sys.path.append(os.path.join(pkg_path, ".."))
  __package__ = f"{pkg_name}.{module_name}"  # noqa: A001

from absl import app

from . import training


def main(argv):
  del argv  # unused parameter
  model = training.train()
  del model  # unused


if __name__ == "__main__":
  app.run(main)
