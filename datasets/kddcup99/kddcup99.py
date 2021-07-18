import gzip

import pandas as pd
import tensorflow as tf

from .. import download
from . import dl_config, ds_config


class KddCup99():
  """KDD Cup 99 dataset."""

  _TARGET_COLUMN = 'label'
  _NORMAL_LABEL = 'normal.'

  feature_dict = ds_config.feature_dict

  def _generate_ds(self, ds_name):
    """Generate dataframe"""
    ds_path = download.download_dataset(ds_name, dl_config.DATASETS)
    with ds_path.open('rb') as f:
      with gzip.open(f, 'rt', newline='') as gz:
        df = pd.read_csv(
          gz, header=0, names=self.feature_dict, dtype=self.feature_dict
        )
        target = df.pop(self._TARGET_COLUMN)
        df = pd.get_dummies(df)
        target = target == self._NORMAL_LABEL

        return tf.data.Dataset.from_tensor_slices((df.values, target.values))

  def get_dfs(self):
    df_train = self._generate_ds('train')
    df_test = self._generate_ds('test')
    return df_train, df_test
