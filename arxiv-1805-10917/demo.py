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

from absl import app, flags

from . import data, model

FLAGS = flags.FLAGS

flags.DEFINE_integer("epochs", 100, "Number of training epochs", lower_bound=1)


def main(_argv):
  (n_transforms, train_normal, train_anomalous, test_normal,
   test_anomalous) = data.get_datasets()
  m = model.get_model(n_transforms)

  m.fit(train_normal, epochs=FLAGS.epochs, validation_data=test_normal)


if __name__ == "__main__":
  app.run(main)
