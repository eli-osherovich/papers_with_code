#!/usr/bin/env python3

from absl import app
from absl import flags

from pwc.city.default_flow import flow_automl as default_flow
from pwc.city.fraud_flow import flow_automl as fraud_flow

FLAGS = flags.FLAGS

flags.DEFINE_enum("dataset", "fraud", ["fraud", "default"], "Dataset to use")

flags.DEFINE_integer(
  "max_trees", 2000, "Maximal number of trees for boosters", lower_bound=1
)
flags.DEFINE_integer("trials", 150, "Number of tuning trials", lower_bound=0)
flags.DEFINE_integer("cv_folds", 5, "Number of tuning trials", lower_bound=2)
flags.DEFINE_integer(
  "num_leaves", 128, "Number of tuning trials", lower_bound=2
)
flags.DEFINE_integer("patience", 50, "Patience during early stopping")
flags.DEFINE_integer("timeout", 3600, "Trials timeout (seconds)", lower_bound=1)


def main(argv):
  del argv  # not used

  if FLAGS.dataset == "fraud":
    fraud_flow.run()
  if FLAGS.dataset == "default":
    default_flow.run()


if __name__ == "__main__":
  app.run(main)
