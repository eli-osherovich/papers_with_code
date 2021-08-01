import pandas as pd

from .. import dataset, io


class KddCup99(dataset.Dataset):
  """KDD Cup 99 dataset."""

  _TARGET_COLUMN = 'label'
  _NORMAL_LABEL = 'normal.'

  def _generate_dataframe(self, ds_name):
    """Generate dataframes"""
    ds_path = self.download_dataset(ds_name)
    file_accessor = io.FileAccessor(ds_path)
    file_reader = io.PandasCSVReader(
      header=None, names=self.feature_dict, dtype=self.feature_dict
    )
    X = file_accessor.read(file_reader)
    target = X.pop(self._TARGET_COLUMN)
    X = pd.get_dummies(X)
    y = target == self._NORMAL_LABEL

    return X, y
