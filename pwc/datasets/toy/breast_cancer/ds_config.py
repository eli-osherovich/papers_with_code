"""Breast Cancer dataset config.
"""

from pwc.datasets import dataset
from pwc.datasets import utils

# flake8: noqa: E501
# pylint: disable=line-too-long

SPLITS = {
  "train":
    dataset.DatasetFile(
      uri=utils.make_uri("breast_cancer.csv"),
      checksum="FIXME: add real checksum",
    ),
}
