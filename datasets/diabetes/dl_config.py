"""Diabetes download config.
"""
from .. import utils

DATASETS = {
  "train": {
    "uri":
      utils.make_uri("diabetes.csv"),
    "checksum":
      "9af41baf65fe75e3e76283978180c9399dcc167656377219b64b793503ce7c76",
  },
}
