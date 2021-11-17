"""Implementation of the KDD Cup 99 dataset.
"""
from pwc.datasets import dataset
from pwc.datasets.typing import DatasetType


class KDDCup99(dataset.Dataset):
  """KDD Cup 99 dataset."""

  def __init__(self) -> None:
    super().__init__(type=DatasetType.BINARY, target_columns=["label"])
