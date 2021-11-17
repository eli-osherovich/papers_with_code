"""Implementation of the Synthetic Fraud Detection dataset.
"""

from pwc.datasets import dataset
from pwc.datasets.typing import DatasetType


class SyntheticFraudDetection(dataset.Dataset):
  """Synthetic Fraud Detection dataset."""

  def __init__(self) -> None:
    super().__init__(type=DatasetType.BINARY, target_columns=["isFraud"])
