"""HELOC dataset config.
"""

from pwc.datasets import dataset
from pwc.datasets import utils

# flake8: noqa: E501
# pylint: disable=line-too-long

SPLITS = {
  "train":
    dataset.DatasetFile(
      uri=utils.make_uri("heloc_dataset_v1.csv.gz"),
      checksum="28dbd94a3f975858bf19c82ba43871d63432139713386d9e6331fd5f47c5f0d6",
    ),
}
