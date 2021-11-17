"""KDD Cup 09 dataset config.
"""

from pwc.datasets import dataset

# flake8: noqa: E501
# pylint: disable=line-too-long

SPLITS = {
  "train_features_small":
    dataset.DatasetFile(
      uri="https://kdd.org/cupfiles/KDDCupData/2009/orange_small_train.data.zip",
      checksum="31ccb810bdbb71c16e079326443166dc3dfbf73cd358fc4a4ce7440fb1bc6040",
      file_accessor_args={"name": "orange_small_train.data"},
      file_reader_args={"sep": "\t"}
    ),
  "labels_churn_small":
    dataset.DatasetFile(
      uri="https://kdd.org/cupfiles/KDDCupData/2009/orange_small_train_churn.labels",
      checksum="fe8891cc574bd55a214514e522a5bed1eec2c3f347a49a36e51620009e7b6f5b",
      file_reader_args={
        "names": ["target"],
      }
    ),
  "labels_appetency_small":
    dataset.DatasetFile(
      uri="https://kdd.org/cupfiles/KDDCupData/2009/orange_small_train_appetency.labels",
      checksum="edbfa40e7513804cf25c3f8b3c8f4a6cf5c77116cffc2f87ef770351250a963c",
      file_reader_args={"names": ["target"]}
    ),
  "labels_upselling_small":
    dataset.DatasetFile(
      uri="https://kdd.org/cupfiles/KDDCupData/2009/orange_small_train_upselling.labels",
      checksum="86effe68394fe1ab21c2d855f74adf70f442990aa95dfe5c97340fc924440e68",
      file_reader_args={"names": ["target"]}
    ),
}
