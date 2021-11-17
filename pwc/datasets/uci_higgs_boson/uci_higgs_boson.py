"""Implementation of the UCI Higgs Boson dataset.
"""

from .. import dataset
from ..typing import DatasetType


class UCIHiggsBoson(dataset.Dataset):
  """UCI Higgs Boson dataset."""

  def __init__(self) -> None:
    super().__init__(type=DatasetType.BINARY, target_columns=["target"])
