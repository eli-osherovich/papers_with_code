"""Implementation of the UCI Mushroom dataset.
"""

from pwc.datasets import dataset
from pwc.datasets.typing import DatasetType


class Mushroom(dataset.Dataset):
  """UCI Mushroom dataset."""

  def __init__(self) -> None:
    super().__init__(type=DatasetType.BINARY, target_columns=["target"])
