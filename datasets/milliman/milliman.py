import numpy as np
import pandas as pd
import tensorflow as tf

from .. import dataset, io


class MillimanDataset(dataset.Dataset):
  """Milliman dataset."""

  _TARGET_COLUMN = 'default'

  def _generate_df(self, ds_name):
    ds_path = self.download_dataset(ds_name)
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
    df = file_accessor.read(file_reader)
    return df

  def _generate_ds(self, ds_name):
    """Generate dataframe"""

    df = self._generate_df(self, ds_name)
    target = df.pop(self._TARGET_COLUMN)
    df = pd.get_dummies(df)

    return tf.data.Dataset.from_tensor_slices(
      (df.to_numpy(dtype=np.float32), target.to_numpy(dtype=np.float32))
    )
