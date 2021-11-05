"""{{cookiecutter.dataset}} dataset schema.
"""
import numpy as np
import pandas as pd

_INT_TYPE = np.int32
_FLOAT_TYPE = np.float32
_BOOL_TYPE = np.int32
_CATEGORICAL_TYPE = pd.CategoricalDtype

raise NotImplementedError("Add dataset schema")
feature_dict = {
  # "col1": _INT_TYPE,
  # "col2": CATEGORICAL_TYPE([
  #   "cat",
  #   "dog",
  #   "human",
  #   "unknown",
  # ])
  {% for n in range(cookiecutter.num_features|int) %}
  "f{{n}}": _FLOAT_TYPE,
  {%- endfor %}
}
