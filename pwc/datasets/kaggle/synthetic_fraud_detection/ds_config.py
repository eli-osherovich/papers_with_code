"""Synthetic Fraud Detection dataset config.
"""

from pwc.datasets import dataset
from pwc.datasets import utils

# flake8: noqa: E501
# pylint: disable=line-too-long

SPLITS = {
  "train":
    dataset.DatasetFile(
      uri=utils.make_uri("archive.zip"),
      checksum="f7eef9ffad5cfa64a034143a5c9b30491d189420b273d5ad5723ca40b596613d",
      file_accessor_args={"name": "PS_20174392719_1491204439457_log.csv"},
    ),
}
