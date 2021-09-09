import pandas as pd

from .. import dataset
from .. import io


class MillimanDataset(dataset.Dataset):
  """Milliman dataset."""

  _TARGET_COLUMN = 'default'

  def _generate_dataframe(self, split_name):
    ds_path = self.download_dataset(split_name)
    file_accessor = io.FileAccessor(ds_path)
    file_reader = io.PandasCSVReader(
      sep='\\s+,',
      header=0,
      engine='python',
      names=self.feature_dict,
      dtype=self.feature_dict,
      true_values=['yes'],
      false_values=['no'],
    )
    X = file_accessor.read(file_reader)
    X.drop(['percent_of_income'], axis=1, inplace=True)
    y = X.pop(self._TARGET_COLUMN)
    X = pd.get_dummies(X)
    print(f'cols=[{X.columns}]')
    return X, y
