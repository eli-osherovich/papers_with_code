"""Implementation of the HELOC dataset.
"""

from pwc.datasets import dataset
from pwc.datasets.typing import DatasetType


class HELOC(dataset.Dataset):
  """HELOC dataset."""

  def __init__(self) -> None:
    super().__init__(
      type=DatasetType.BINARY, target_columns=["RiskPerformance"]
    )
