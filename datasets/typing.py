import enum

import numpy as np
import pandas as pd

BoolType = np.int32
CategoricalType = pd.CategoricalDtype
FloatType = np.float32
IntType = np.int32


@enum.unique
class DatasetType(enum.Enum):
  RAW_DATA = 0
  REGRESSION = 1
  BINARY = 2
  MULTICLASS = 3
