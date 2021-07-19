from . import *  # noqa F403

from absl import logging


def load_dataset(ds_cls):
  try:
    ds = globals()[ds_cls]()
    return ds.get_datasets()
  except KeyError as e:
    logging.error('Cannot find DS class %s', ds_cls)
    raise RuntimeError(f'Unable to load DS class called {ds_cls}') from e
