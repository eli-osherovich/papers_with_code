import numpy as np
import pandas as pd
import tensorflow as tf

from .. import dataset, io


class KddCup99(dataset.Dataset):
  """KDD Cup 99 dataset."""

  _TARGET_COLUMN = 'label'
  _NORMAL_LABEL = 'normal.'

  def _generate_ds(self, ds_name):
    """Generate dataframe"""
    ds_path = self.download_dataset(ds_name)
    file_accessor = io.FileAccessor(ds_path)
    file_reader = io.PandasCSVReader(
      header=None, names=self.feature_dict, dtype=self.feature_dict
    )
    df = file_accessor.read(file_reader)
    target = df.pop(self._TARGET_COLUMN)
    df = pd.get_dummies(df)
    target = target == self._NORMAL_LABEL

    return tf.data.Dataset.from_tensor_slices(
      (df.to_numpy(dtype=np.float32), target.to_numpy(dtype=np.float32))
    )
