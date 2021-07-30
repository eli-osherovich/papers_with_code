import gzip

import numpy as np
import pandas as pd
import tensorflow as tf

from .. import dataset


class KddCup99(dataset.Dataset):
  """KDD Cup 99 dataset."""

  _TARGET_COLUMN = 'label'
  _NORMAL_LABEL = 'normal.'

  def _generate_ds(self, ds_name):
    """Generate dataframe"""
    ds_path = self.download_dataset(ds_name)
    with ds_path.open('rb') as f:
      with gzip.open(f, 'rt', newline='') as gz:
        df = pd.read_csv(gz, names=self.feature_dict, dtype=self.feature_dict)
        target = df.pop(self._TARGET_COLUMN)
        df = pd.get_dummies(df)
        target = target == self._NORMAL_LABEL

        return tf.data.Dataset.from_tensor_slices(
          (df.to_numpy(dtype=np.float32), target.to_numpy(dtype=np.float32))
        )
