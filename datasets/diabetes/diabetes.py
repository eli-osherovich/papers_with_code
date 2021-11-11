"""Implementation of the Diabetes dataset.
"""

from .. import dataset


class Diabetes(dataset.Dataset):
  """Diabetes dataset."""

  def __init__(self) -> None:
    super().__init__(target_columns=["target"])
