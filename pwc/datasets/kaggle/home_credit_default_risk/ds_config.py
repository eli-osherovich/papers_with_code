"""Home Credit Default Risk dataset config.
"""

from pwc.datasets import dataset
from pwc.datasets import utils

# flake8: noqa: E501
# pylint: disable=line-too-long

SPLITS = {
  "train":
    dataset.DatasetFile(
      uri=utils.make_uri("home-credit-default-risk.zip"),
      checksum="4e7a243b13f6e3d40f682a9a0cfd59c70910ca6c5a4289f71ca1f25efbd8e3de",
      file_accessor_args={"name": "application_train.csv"},
      file_reader_args={
        "true_values": ["Y"],
        "false_values": ["N"],
      }
    ),
  "bureau":
    dataset.DatasetFile(
      uri=utils.make_uri("home-credit-default-risk.zip"),
      checksum="4e7a243b13f6e3d40f682a9a0cfd59c70910ca6c5a4289f71ca1f25efbd8e3de",
      file_accessor_args={"name": "bureau.csv"},
      file_reader_args={
        "true_values": ["Y"],
        "false_values": ["N"],
      }
    ),
  "bureau_balance":
    dataset.DatasetFile(
      uri=utils.make_uri("home-credit-default-risk.zip"),
      checksum="4e7a243b13f6e3d40f682a9a0cfd59c70910ca6c5a4289f71ca1f25efbd8e3de",
      file_accessor_args={"name": "bureau_balance.csv"},
      file_reader_args={
        "true_values": ["Y"],
        "false_values": ["N"],
      }
    ),
  "pos_cache_balance":
    dataset.DatasetFile(
      uri=utils.make_uri("home-credit-default-risk.zip"),
      checksum="4e7a243b13f6e3d40f682a9a0cfd59c70910ca6c5a4289f71ca1f25efbd8e3de",
      file_accessor_args={"name": "POS_CASH_balance.csv"},
      file_reader_args={
        "true_values": ["Y"],
        "false_values": ["N"],
      }
    ),
  "credit_card_balance":
    dataset.DatasetFile(
      uri=utils.make_uri("home-credit-default-risk.zip"),
      checksum="4e7a243b13f6e3d40f682a9a0cfd59c70910ca6c5a4289f71ca1f25efbd8e3de",
      file_accessor_args={"name": "credit_card_balance.csv"},
      file_reader_args={
        "true_values": ["Y"],
        "false_values": ["N"],
      }
    ),
  "previous_application":
    dataset.DatasetFile(
      uri=utils.make_uri("home-credit-default-risk.zip"),
      checksum="4e7a243b13f6e3d40f682a9a0cfd59c70910ca6c5a4289f71ca1f25efbd8e3de",
      file_accessor_args={"name": "previous_application.csv"},
      file_reader_args={
        "true_values": ["Y"],
        "false_values": ["N"],
      }
    ),
  "installments_payments":
    dataset.DatasetFile(
      uri=utils.make_uri("home-credit-default-risk.zip"),
      checksum="4e7a243b13f6e3d40f682a9a0cfd59c70910ca6c5a4289f71ca1f25efbd8e3de",
      file_accessor_args={"name": "installments_payments.csv"},
      file_reader_args={
        "true_values": ["Y"],
        "false_values": ["N"],
      }
    ),
}
