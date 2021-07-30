import numpy as np
import pandas as pd
import tensorflow as tf

from .. import dataset


class MillimanDataset(dataset.Dataset):
  """Milliman dataset."""

  _TARGET_COLUMN = 'default'

  def _get_df(self, ds_name):
    ds_path = self.download_dataset(ds_name)
    with ds_path.open('rt', newline='') as f:
      df = pd.read_csv(
        f,
        sep='\\s+,',
        header=0,
        engine='python',
        names=self.feature_dict,
        dtype=self.feature_dict,
        true_values=['yes'],
        false_values=['no'],
      )
      return df

  def _generate_ds(self, ds_name):
    """Generate dataframe"""

    df = self._get_df(self, ds_name)
    target = df.pop(self._TARGET_COLUMN)
    df = pd.get_dummies(df)

    return tf.data.Dataset.from_tensor_slices(
      (df.to_numpy(dtype=np.float32), target.to_numpy(dtype=np.float32))
    )
