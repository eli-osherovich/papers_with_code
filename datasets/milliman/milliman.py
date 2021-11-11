import pandas as pd

from .. import dataset


class MillimanDataset(dataset.Dataset):
  """Milliman dataset."""

  def __init__(self) -> None:
    super().__init__(target_columns=["default"])


class MillimanDataset2(MillimanDataset):
  """Another version of the Milliman dataset: with ordinal variables."""

  def as_dataframe(self, split: str):
    df = super().as_dataframe(split)
    df["checking_balance_unknown"] = df["checking_balance"] == "unknown"
    df["checking_balance"] = df["checking_balance"].map({
      "< 0 DM": 0,
      "1 - 200 DM": 200,
      "> 200 DM": 400
    }).astype(float)

    df["credit_history"] = df["credit_history"].map({
      "poor": 0,
      "critical": 1,
      "good": 2,
      "very good": 3,
      "perfect": 4
    }).astype(float)

    df["savings_balance"] = df["savings_balance"].map({
      "unknown": 0,
      "< 100 DM": 100,
      "100 - 500 DM": 500,
      "500 - 1000 DM": 1000,
      "> 1000 DM": 1500
    }).astype(float)

    df["employment_duration"] = df["employment_duration"].map({
      "unemployed": 0,
      "< 1 year": 1,
      "1 - 4 years": 4,
      "4 - 7 years": 7,
      "> 7 years": 10
    }).astype(float)
    df = df.fillna(df.median(numeric_only=True))
    return df
