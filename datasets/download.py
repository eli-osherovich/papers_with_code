import pathlib
import tempfile
from typing import Optional, Union

import tensorflow as tf

_DATASETS_DIR = 'pwc-datasets'


def download_dataset(
  ds_name: str,
  dl_config: dict[str, str],
  cache_dir: Optional[Union[str, pathlib.Path]] = None,
):
  if cache_dir is None:
    cache_dir = pathlib.Path(tempfile.gettempdir()) / _DATASETS_DIR
  else:
    cache_dir = pathlib.Path(cache_dir)  # in case it is `str`.

  if ds_name not in dl_config:
    raise RuntimeError(
      f'Dataset name {ds_name} cannot be found in the download config {dl_config}'  # noqa E501
    )

  cache_dir.mkdir(parents=True, exist_ok=True)

  url = dl_config[ds_name]['url']
  checksum = dl_config[ds_name].get('checksum')

  data_path = tf.keras.utils.get_file(
    origin=url, file_hash=checksum, cache_dir=cache_dir, cache_subdir='.'
  )
  return pathlib.Path(data_path)
