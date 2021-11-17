"""Rossmann Store Sales dataset config.
"""

from .. import dataset
from .. import utils

# flake8: noqa: E501
# pylint: disable=line-too-long

SPLITS = {
  "train":
    dataset.DatasetFile(
      uri="https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/4594/860645/bundle/archive.zip",
      checksum="52ce715e02dc70cac16b14548580d656997f5d43ce3544220d5e574d26483cf3",
      file_accessor_args={"name": "train.csv"},
      file_reader_args={
        "low_memory": False,
        "parse_dates": ["Date"],
      }
    ),
  "val":
    dataset.DatasetFile(
      uri=utils.make_uri("rossmann-store-sales.zip"),
      checksum="52ce715e02dc70cac16b14548580d656997f5d43ce3544220d5e574d26483cf3",
      file_accessor_args={"name": "test.csv"},
      file_reader_args={
        "parse_dates": ["CompetitionOpenSinceYear", "Promo2SinceYear"]
      }
    ),
  "store":
    dataset.DatasetFile(
      uri=utils.make_uri("rossmann-store-sales.zip"),
      checksum="52ce715e02dc70cac16b14548580d656997f5d43ce3544220d5e574d26483cf3",
      file_accessor_args={"name": "store.csv"},
    ),
}
