"""Implementation of the UCI Cover Type dataset.
"""

from .. import dataset
from ..typing import DatasetType


class UCICoverType(dataset.Dataset):
  """UCI Cover Type dataset."""

  def __init__(self) -> None:
    super().__init__(type=DatasetType.MULTICLASS, target_columns=["CoverType"])
