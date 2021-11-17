"""Implementation of the UCI Adult dataset.
"""

from pwc.datasets import dataset
from pwc.datasets.typing import DatasetType


class Adult(dataset.Dataset):
  """UCI Adult dataset."""

  def __init__(self) -> None:
    super().__init__(type=DatasetType.BINARY, target_columns=["target"])
