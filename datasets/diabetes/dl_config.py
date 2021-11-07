"""Diabetes download config.
"""
from .. import utils

DATASETS = {
  "train": {
    "uri": utils.make_uri("diabetes.csv"),
    "checksum": "FIXME: add real checksum",
  },
  
  
}
