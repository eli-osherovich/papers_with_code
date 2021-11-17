"""Iris dataset config.
"""

from pwc.datasets import dataset
from pwc.datasets import utils

# flake8: noqa: E501
# pylint: disable=line-too-long

SPLITS = {
  "train":
    dataset.DatasetFile(
      uri=utils.make_uri("iris.csv"),
      checksum="0ba79ae755c686ee02dfe1d2943772a46ded2433c4fde6dd7ad3b01c41ff5d3d",
    ),
}
