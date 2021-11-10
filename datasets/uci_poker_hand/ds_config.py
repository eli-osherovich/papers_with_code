"""UCI Poker Hand dataset schema.
"""
from absl import logging
import numpy as np
import pandas as pd

_INT_TYPE = np.int32
_FLOAT_TYPE = np.float32
_BOOL_TYPE = np.int32
_CATEGORICAL_TYPE = pd.CategoricalDtype

logging.error("Please review the schema below and remove this exception")
feature_dict = {
  # Example:
  #
  # "col1": _INT_TYPE,
  # "col2": CATEGORICAL_TYPE([
  #   "cat",
  #   "dog",
  #   "human",
  #   "unknown",
  # ])
  "S1": _FLOAT_TYPE,
  "C1": _FLOAT_TYPE,
  "S2": _FLOAT_TYPE,
  "C2": _FLOAT_TYPE,
  "S3": _FLOAT_TYPE,
  "C3": _FLOAT_TYPE,
  "S4": _FLOAT_TYPE,
  "C4": _FLOAT_TYPE,
  "S5": _FLOAT_TYPE,
  "C5": _FLOAT_TYPE,
  "Hand": _INT_TYPE,
}
