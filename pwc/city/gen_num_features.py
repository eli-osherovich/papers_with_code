#!/usr/bin/env python

import itertools
import pathlib

from absl import app
from absl import flags
import numpy as np
import pandas as pd
from prefect import Flow
from prefect import task
from prefect.utilities import logging
import pyarrow as pa
import pyarrow.feather as feather

from pwc.city import task_lib
from pwc.datasets import HomeCreditDefaultRisk as DataSet

FLAGS = flags.FLAGS

flags.DEFINE_float("null_thresh", 0.6, "NULLity threshold", lower_bound=0)

_logger = logging.get_logger()

dump_feather = task(task_lib.dump_feather)
remove_null_columns = task(task_lib.remove_null_columns)
set_nans = task(task_lib.set_nans)


@task
def load_data() -> dict[str, pd.DataFrame]:
  ds = DataSet()
  res = {}
  res["train"] = ds.as_dataframe("train")
  # res["previous_application"] = ds.as_dataframe("previous_application")
  # res["installments_payments"] = ds.as_dataframe("installments_payments")
  # res["bureau"] = ds.as_dataframe("bureau")
  # res["bureau_balance"] = ds.as_dataframe("bureau_balance")
  # res["pos_cache_balance"] = ds.as_dataframe("pos_cache_balance")
  # res["credit_card_balance"] = ds.as_dataframe("credit_card_balance")

  return res


@task
def gen_num_features(data: dict[str, pd.DataFrame]) -> None:
  for key, df in data.items():
    num_columns = df.select_dtypes(include="number", exclude="boolean").columns
    auto_feats = {"SK_ID_CURR": df["SK_ID_CURR"]}

    for c0, c1 in itertools.combinations(num_columns, 2):
      if "SK_ID_CURR" in {c0, c1}:
        continue

      minus_res = df[c0] - df[c1]
      plus_res = df[c0] + df[c1]
      if 0 in df[c1]:
        # Add some random number to the denominator to avoid division by zero.
        div_res = df[c0] / (0.1003145241251567954 + df[c1])
      else:
        div_res = df[0] / df[1]

      auto_feats[f"autofeat__{c0}__div__{c1}"] = pa.array(
        pd.to_numeric(div_res, downcast="float")
      )
      auto_feats[f"autofeat__{c0}__minus__{c1}"] = pa.array(
        pd.to_numeric(minus_res, downcast="float")
      )
      auto_feats[f"autofeat__{c0}__plus__{c1}"] = pa.array(
        pd.to_numeric(plus_res, downcast="float")
      )

    fpath = pathlib.Path(
      __file__
    ).parent / "data" / f"{key}_autofeat_num_0.feather"

    feather.write_feather(pa.table(auto_feats), fpath)


@task
def gen_bool_features(data: dict[str, pd.DataFrame]) -> None:
  for key, df in data.items():
    num_columns = df.select_dtypes(include="bool").columns
    auto_feats = {"SK_ID_CURR": df["SK_ID_CURR"]}
    for c0, c1 in itertools.combinations(num_columns, 2):
      if "TARGET" in {c0, c1}:
        continue

      or_res = df[c0] | df[c1]
      and_res = df[c0] & df[c1]
      xor_res = df[c0] ^ df[c1]

      auto_feats[f"autofeat__{c0}__or__{c1}"] = pa.array(or_res)
      auto_feats[f"autofeat__{c0}__and__{c1}"] = pa.array(and_res)
      auto_feats[f"autofeat__{c0}__xor__{c1}"] = pa.array(xor_res)

    fpath = pathlib.Path(
      __file__
    ).parent / "data" / f"{key}_autofeat_bool_0.feather"

    feather.write_feather(pa.table(auto_feats), fpath)


def main(argv):
  del argv  # not used

  with Flow("feature_engineering") as flow:
    data = load_data()
    data = set_nans(data)
    dump_feather(data)
    data = remove_null_columns(data, FLAGS.null_thresh, logger=_logger)
    gen_num_features(data)
    gen_bool_features(data)
  flow.run()


if __name__ == "__main__":
  app.run(main)
