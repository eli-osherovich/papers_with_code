#!/usr/bin/env python3

from absl import app
from absl import flags

from pwc.common import utils

FLAGS = flags.FLAGS

flags.DEFINE_enum("dataset", "fraud", ["fraud", "default"], "Dataset to use")

flags.DEFINE_integer(
  "max_trees", 3000, "Maximal number of trees for boosters", lower_bound=1
)
flags.DEFINE_integer("trials", 150, "Number of tuning trials", lower_bound=0)
flags.DEFINE_integer("cv_folds", 5, "Number of tuning trials", lower_bound=2)

from pwc.city.default_flow import flow_automl as default_flow
from pwc.city.fraud_flow import flow_automl as fraud_flow


def main(argv):
  del argv  # not used

  if FLAGS.dataset == "fraud":
    fraud_flow.run()
  if FLAGS.dataset == "default":
    default_flow.run()


if __name__ == "__main__":
  utils.setup_omp()
  app.run(main)
