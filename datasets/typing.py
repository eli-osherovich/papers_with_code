import enum

import numpy as np
import pandas as pd

IntType = np.int32
FloatType = np.float32
BoolType = np.int32
CategoricalType = pd.CategoricalDtype


@enum.unique
class DatasetType(enum.Enum):
  RAW_DATA = 0
  REGRESSION = 1
  BINARY = 2
  MULTICLASS = 3
