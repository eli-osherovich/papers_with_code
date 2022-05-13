#!/usr/bin/env python3

from absl import app
from absl import flags

from pwc.city.default_flow import flow_automl as default_flow
from pwc.city.fraud_flow import flow_automl as fraud_flow
from pwc.city.test import flow_automl as test_flow

FLAGS = flags.FLAGS

flags.DEFINE_enum(
  "dataset", "fraud", ["fraud", "default", "test"], "Dataset to use"
)
flags.DEFINE_float("lr", 0.025, "Learning rate", lower_bound=1e-10)
flags.DEFINE_integer("cv_folds", 5, "Number of tuning trials", lower_bound=2)
flags.DEFINE_integer(
  "max_trees", 5000, "Maximal number of trees for boosters", lower_bound=1
)
flags.DEFINE_integer(
  "num_leaves", 128, "Number of tuning trials", lower_bound=2
)
flags.DEFINE_integer("patience", 50, "Patience during early stopping")
flags.DEFINE_integer("timeout", 3600, "Trials timeout (seconds)", lower_bound=1)
flags.DEFINE_integer("trials", 150, "Number of tuning trials", lower_bound=1)


def main(argv):
  del argv  # not used

  if FLAGS.dataset == "fraud":
    fraud_flow.run()
  elif FLAGS.dataset == "default":
    default_flow.run()
  elif FLAGS.dataset == "test":
    test_flow.run()


if __name__ == "__main__":
  app.run(main)
