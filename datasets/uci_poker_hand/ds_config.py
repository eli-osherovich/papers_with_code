"""UCI Poker Hand dataset config.
"""

import numpy as np
import pandas as pd

from .. import dataset
from .. import utils

# flake8: noqa: E501
# pylint: disable=line-too-long

INT_TYPE = np.int32
FLOAT_TYPE = np.float32
BOOL_TYPE = np.int32
CATEGORICAL_TYPE = pd.CategoricalDtype

feature_dict = {
  "S1": FLOAT_TYPE,
  "C1": FLOAT_TYPE,
  "S2": FLOAT_TYPE,
  "C2": FLOAT_TYPE,
  "S3": FLOAT_TYPE,
  "C3": FLOAT_TYPE,
  "S4": FLOAT_TYPE,
  "C4": FLOAT_TYPE,
  "S5": FLOAT_TYPE,
  "C5": FLOAT_TYPE,
  "Hand": INT_TYPE,
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
