"""Sarcos dataset config.
"""

from .. import dataset
from .. import utils

# flake8: noqa: E501
# pylint: disable=line-too-long

SPLITS = {
  "train":
    dataset.DatasetFile(
      uri=utils.make_uri("sarcos_train.csv"),
      checksum="1b1110fbfc59ca8b0fa2996e917d7e9621ffe603b0c12b84e0e6dda24df251f8",
    ),
  "val":
    dataset.DatasetFile(
      uri=utils.make_uri("sarcos_test.csv"),
      checksum="94a12314c2df7df37dda04cb858f78c1504b9207b5c0c954298bdbbbf88b4ed4",
    ),
}
