"""Diabetes dataset config.
"""

from pwc.datasets import dataset
from pwc.datasets import utils

# flake8: noqa: E501
# pylint: disable=line-too-long

SPLITS = {
  "train":
    dataset.DatasetFile(
      uri=utils.make_uri("diabetes.csv"),
      checksum="9af41baf65fe75e3e76283978180c9399dcc167656377219b64b793503ce7c76",
    ),
}
