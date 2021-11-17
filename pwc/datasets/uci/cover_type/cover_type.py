"""Implementation of the UCI Cover Type dataset.
"""

from pwc.datasets import dataset
from pwc.datasets.typing import DatasetType


class CoverType(dataset.Dataset):
  """UCI Cover Type dataset."""

  def __init__(self) -> None:
    super().__init__(type=DatasetType.MULTICLASS, target_columns=["CoverType"])
