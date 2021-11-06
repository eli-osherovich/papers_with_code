"""Implementation of the Iris dataset.
"""
import pandas as pd

from .. import dataset


class Iris(dataset.Dataset):
  """Iris dataset."""

  def __init__(self) -> None:
    super().__init__(target_columns=["target"], df_args={"skiprows": 1})
