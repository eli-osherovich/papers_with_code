"""Iris download config.
"""
from .. import utils

DATASETS = {
  "train": {
    "uri":
      utils.make_uri("iris.csv"),
    "checksum":
      "f13ffa8fdd56fd8e6c8d16d4081a3fbd3114bcd0aae4256c43205169cd9d1449",
  },
}
