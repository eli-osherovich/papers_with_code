"""Breast Cancer download config.
"""
from .. import utils

DATASETS = {
  "train": {
    "uri": utils.make_uri("breast_cancer.csv"),
    "checksum": "FIXME: add real checksum",
  },
  
  
}
