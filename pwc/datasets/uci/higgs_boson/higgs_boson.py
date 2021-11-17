"""Implementation of the UCI Higgs Boson dataset.
"""

from pwc.datasets import dataset
from pwc.datasets.typing import DatasetType


class HiggsBoson(dataset.Dataset):
  """UCI Higgs Boson dataset."""

  def __init__(self) -> None:
    super().__init__(type=DatasetType.BINARY, target_columns=["target"])
