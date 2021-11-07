"""Breast Cancer dataset schema.
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
  
  "feature0": _FLOAT_TYPE,
  "feature1": _FLOAT_TYPE,
  "feature2": _FLOAT_TYPE,
  "feature3": _FLOAT_TYPE,
  "feature4": _FLOAT_TYPE,
  "feature5": _FLOAT_TYPE,
  "feature6": _FLOAT_TYPE,
  "feature7": _FLOAT_TYPE,
  "feature8": _FLOAT_TYPE,
  "feature9": _FLOAT_TYPE,
  "feature10": _FLOAT_TYPE,
  "feature11": _FLOAT_TYPE,
  "feature12": _FLOAT_TYPE,
  "feature13": _FLOAT_TYPE,
  "feature14": _FLOAT_TYPE,
  "feature15": _FLOAT_TYPE,
  "feature16": _FLOAT_TYPE,
  "feature17": _FLOAT_TYPE,
  "feature18": _FLOAT_TYPE,
  "feature19": _FLOAT_TYPE,
  "feature20": _FLOAT_TYPE,
  "feature21": _FLOAT_TYPE,
  "feature22": _FLOAT_TYPE,
  "feature23": _FLOAT_TYPE,
  "feature24": _FLOAT_TYPE,
  "feature25": _FLOAT_TYPE,
  "feature26": _FLOAT_TYPE,
  "feature27": _FLOAT_TYPE,
  "feature28": _FLOAT_TYPE,
  "feature29": _FLOAT_TYPE,
  "target": _BOOL_TYPE,
}
