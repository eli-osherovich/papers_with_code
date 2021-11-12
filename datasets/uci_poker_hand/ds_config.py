"""UCI Poker Hand dataset config.
"""

from .. import dataset
from .. import utils
from ..typing import FloatType
from ..typing import IntType

# flake8: noqa: E501
# pylint: disable=line-too-long

feature_dict = {
  "S1": FloatType,
  "C1": FloatType,
  "S2": FloatType,
  "C2": FloatType,
  "S3": FloatType,
  "C3": FloatType,
  "S4": FloatType,
  "C4": FloatType,
  "S5": FloatType,
  "C5": FloatType,
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
