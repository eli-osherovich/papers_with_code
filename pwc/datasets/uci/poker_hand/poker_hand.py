"""Implementation of the UCI Poker Hand dataset.
"""

from pwc.datasets import dataset
from pwc.datasets.typing import DatasetType


class PokerHand(dataset.Dataset):
  """UCI Poker Hand dataset."""

  def __init__(self) -> None:
    super().__init__(type=DatasetType.MULTICLASS, target_columns=["Hand"])
