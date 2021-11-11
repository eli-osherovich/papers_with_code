"""Implementation of the Breast Cancer dataset.
"""

from .. import dataset


class BreastCancer(dataset.Dataset):
  """Breast Cancer dataset."""

  def __init__(self) -> None:
    super().__init__(target_columns=["target"])
