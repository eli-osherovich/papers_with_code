import pandas as pd

from .. import dataset


class MillimanDataset(dataset.Dataset):
  """Milliman dataset."""

  def __init__(self) -> None:
    super().__init__(
      target_columns=["default"],
      df_args={
        "sep": "\\s+,",
        "header": 0,
        "engine": "python",
        "true_values": ["yes"],
        "false_values": ["no"],
      }
    )


class MillimanDataset2(MillimanDataset):
  """Another version of the Milliman dataset: with ordinal variables."""

  def _generate_dataframe(self, split_name):
    X, y = self._read_df(split_name)

    X["checking_balance_unknown"] = X["checking_balance"] == "unknown"
    X["checking_balance"] = X["checking_balance"].map({
      "< 0 DM": 0,
      "1 - 200 DM": 200,
      "> 200 DM": 400
    }).astype(float)

    X["credit_history"] = X["credit_history"].map({
      "poor": 0,
      "critical": 1,
      "good": 2,
      "very good": 3,
      "perfect": 4
    }).astype(float)

    X["savings_balance"] = X["savings_balance"].map({
      "unknown": 0,
      "< 100 DM": 100,
      "100 - 500 DM": 500,
      "500 - 1000 DM": 1000,
      "> 1000 DM": 1500
    }).astype(float)

    X["employment_duration"] = X["employment_duration"].map({
      "unemployed": 0,
      "< 1 year": 1,
      "1 - 4 years": 4,
      "4 - 7 years": 7,
      "> 7 years": 10
    }).astype(float)
    X = X.fillna(X.median())
    X = pd.get_dummies(X, prefix_sep="__:__")
    return X, y
