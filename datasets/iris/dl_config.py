"""Iris download config.
"""
import pathlib

_FILE = pathlib.Path(__file__)
_DS_FILE = "iris.csv"

ds_path = _FILE.parent / _DS_FILE

DATASETS = {
  "train": {
    "uri":
      ds_path.as_uri(),
    "checksum":
      "f13ffa8fdd56fd8e6c8d16d4081a3fbd3114bcd0aae4256c43205169cd9d1449",
  },
}
