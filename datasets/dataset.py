import importlib
import abc

from . import download


class Dataset(abc.ABC):

  @property
  def cls_package(self):
    return self.__module__.rsplit('.', 1)[0]

  @property
  def config_datasets(self):
    dl_config = importlib.import_module('.dl_config', self.cls_package)
    return dl_config.DATASETS

  @property
  def feature_dict(self):
    ds_config = importlib.import_module('.ds_config', self.cls_package)
    return ds_config.feature_dict

  def download_dataset(self, ds_name):
    return download.download_dataset(ds_name, self.config_datasets)

  def get_datasets(self, *splits):
    res = tuple(self._generate_ds(s) for s in splits)
    if len(res) == 1:
      return res[0]
    else:
      return res

  @abc.abstractmethod
  def _generate_ds(self, ds_name):
    raise NotImplementedError
