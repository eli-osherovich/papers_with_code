from functools import cache
import os
import pathlib
import tempfile

import tensorflow as tf

_DATASETS_DIR = 'pwc-datasets'


def download_dataset(ds_name: str, dl_config: dict, cache_dir=None):
  if cache_dir is None:
    cache_dir = os.path.join(tempfile.gettempdir(), _DATASETS_DIR)

  if ds_name not in dl_config:
    raise RuntimeError(
      f'Dataset name {ds_name} cannot be found in the download config {dl_config}'  # noqa E501
    )

  os.makedirs(cache_dir, exist_ok=True)

  url = dl_config[ds_name]['url']
  checksum = dl_config[ds_name].get('checksum')

  data_path = tf.keras.utils.get_file(
    origin=url, file_hash=checksum, cache_dir=cache_dir, cache_subdir='.'
  )
  return pathlib.Path(data_path)
