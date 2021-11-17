"""UCI Poker Hand dataset config.
"""

from pwc.datasets import dataset
from pwc.datasets import utils
from pwc.datasets.typing import CategoricalType
from pwc.datasets.typing import IntType

# flake8: noqa: E501
# pylint: disable=line-too-long

Suit = CategoricalType(range(1, 5))
Card = CategoricalType(range(1, 14))
feature_dict = {
  "S1": Suit,
  "C1": Card,
  "S2": Suit,
  "C2": Card,
  "S3": Suit,
  "C3": Card,
  "S4": Suit,
  "C4": Card,
  "S5": Suit,
  "C5": Card,
  "Hand": IntType,
}

SPLITS = {
  "train":
    dataset.DatasetFile(
      uri=utils.make_uri(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data"
      ),
      checksum="37becdf87d5f8cbf2b91d6471e965a25b86cb4a6d878c0f94a4025969fca464f",
      file_reader_args={
        "names": feature_dict,
        "dtype": feature_dict
      }
    ),
  "val":
    dataset.DatasetFile(
      uri=utils.make_uri(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data"
      ),
      checksum="3cd75958e19dd321ed5ca3f7f154c0f6aad544aab9f37731ac545b5f66b232c7",
      file_reader_args={
        "names": feature_dict,
        "dtype": feature_dict
      }
    ),
}
