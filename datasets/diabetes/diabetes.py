"""Implementation of the Diabetes dataset.
"""
import pandas as pd

from .. import dataset


class Diabetes(dataset.Dataset):
  """Diabetes dataset."""

  def __init__(self) -> None:
    super().__init__(target_columns=["target"], df_args={"skiprows": 1})
