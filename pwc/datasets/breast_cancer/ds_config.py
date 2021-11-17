"""Breast Cancer dataset config.
"""

from .. import dataset
from .. import utils

# flake8: noqa: E501
# pylint: disable=line-too-long

SPLITS = {
  "train":
    dataset.DatasetFile(
      uri=utils.make_uri("breast_cancer.csv"),
      checksum="FIXME: add real checksum",
    ),
}
