"""Implementation of the KDD Cup 09 dataset.
"""

import pandas as pd

from .. import dataset
from ..typing import DatasetType


class KDDCup09(dataset.Dataset):
  """KDD Cup 09 dataset."""

  def __init__(self) -> None:
    super().__init__(type=DatasetType.BINARY, target_columns=["target"])

  def as_dataframe(self, split: str) -> pd.DataFrame:
    if split == "train_churn_small":
      x = super().as_dataframe("train_features_small")
      y = super().as_dataframe("labels_churn_small")
      return pd.concat((x, y), axis=1)
    elif split == "train_appetency_small":
      x = super().as_dataframe("train_features_small")
      y = super().as_dataframe("labels_appetency_small")
      return pd.concat((x, y), axis=1)
    elif split == "train_upselling_small":
      x = super().as_dataframe("train_features_small")
      y = super().as_dataframe("labels_upselling_small")
      return pd.concat((x, y), axis=1)
    else:
      return super().as_dataframe(split)
