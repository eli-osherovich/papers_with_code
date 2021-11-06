"""Iris dataset schema.
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
  "sepal length": _FLOAT_TYPE,
  "sepal width": _FLOAT_TYPE,
  "petal length": _FLOAT_TYPE,
  "petal width": _FLOAT_TYPE,
  "target": _INT_TYPE,
}
