"""Diabetes dataset schema.
"""
import numpy as np
import pandas as pd

_INT_TYPE = np.int32
_FLOAT_TYPE = np.float32
_BOOL_TYPE = np.int32
_CATEGORICAL_TYPE = pd.CategoricalDtype

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
  "age": _FLOAT_TYPE,
  "sex": _FLOAT_TYPE,
  "bmi": _FLOAT_TYPE,
  "bp": _FLOAT_TYPE,
  "s1 tc": _FLOAT_TYPE,
  "s2 ldl": _FLOAT_TYPE,
  "s3 hdl": _FLOAT_TYPE,
  "s4 tch": _FLOAT_TYPE,
  "s5 ltg": _FLOAT_TYPE,
  "s6 glu": _FLOAT_TYPE,
  "target": _FLOAT_TYPE,
}
