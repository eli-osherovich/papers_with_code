"""Implementation of the UCI Poker Hand dataset.
"""

from .. import dataset
from ..typing import DatasetType


class UCIPokerHand(dataset.Dataset):
  """UCI Poker Hand dataset."""

  def __init__(self) -> None:
    super().__init__(type=DatasetType.MULTICLASS, target_columns=["Hand"])
