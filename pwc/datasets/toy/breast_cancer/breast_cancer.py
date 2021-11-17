"""Implementation of the Breast Cancer dataset.
"""

from pwc.datasets import dataset
from pwc.datasets.typing import DatasetType


class BreastCancer(dataset.Dataset):
  """Breast Cancer dataset."""

  def __init__(self) -> None:
    super().__init__(type=DatasetType.BINARY, target_columns=["target"])
