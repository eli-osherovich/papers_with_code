from absl import logging

from . import *  # noqa F403
from . import dataset


# TODO: consider class registry instead of blind import.
def cls_factory(cls, **kwargs):
  try:
    obj = globals()[cls](**kwargs)
    return obj
  except KeyError as e:
    logging.error("Cannot find DS class %s", cls)
    raise ValueError(f"Unable to load class called {cls}") from e


def load_dataset(ds_cls, *splits, **kwargs) -> dataset.DS_RESULT:
  ds = cls_factory(ds_cls, **kwargs)
  return ds.as_dataset(*splits)


def load_dataframe(ds_cls, *splits, **kwargs) -> dataset.DF_RESULT:
  ds = cls_factory(ds_cls, **kwargs)
  return ds.as_dataframe(*splits)


def load_numpy(ds_cls, *splits, **kwargs) -> dataset.NP_RESULT:
  ds = cls_factory(ds_cls, **kwargs)
  return ds.as_numpy(*splits)
