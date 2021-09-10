#!/usr/bin/env python3

# Python's relative import was invented by pure PERVERTS.
if __name__ == "__main__" and __package__ is None:
  import pathlib
  import sys

  module_path = pathlib.Path(__file__).parent.resolve()
  module_name = module_path.name
  pkg_path = module_path.parent
  pkg_name = pkg_path.name
  sys.path.append(pkg_path.parent.as_posix())
  __package__ = f"{pkg_name}.{module_name}"  # noqa: A001

from absl import app

from . import training


def main(argv):
  del argv  # unused parameter
  model = training.train()
  del model  # unused


if __name__ == "__main__":
  app.run(main)
