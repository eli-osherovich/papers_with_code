"""Implementation of the UCI Mushroom dataset.
"""

from .. import dataset
from ..typing import DatasetType


class UCIMushroom(dataset.Dataset):
  """UCI Mushroom dataset."""

  def __init__(self) -> None:
    super().__init__(type=DatasetType.BINARY, target_columns=["target"])
