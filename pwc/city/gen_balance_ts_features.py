#!/usr/bin/env python

import pathlib

from absl import app
from absl import flags
import more_itertools
import tsfresh

from pwc.datasets import HomeCreditDefaultRisk as Dataset

FLAGS = flags.FLAGS

flags.DEFINE_integer("chunk_size", 1000, "Chunk size", lower_bound=1)
flags.DEFINE_string("id_col", "SK_ID_CURR", "ID Column")
flags.DEFINE_string("time_col", "MONTHS_BALANCE", "Time column")
flags.DEFINE_enum(
  "data_type", "pos_cache_balance",
  ["pos_cache_balance", "credit_card_balance"], "Data part"
)


def get_ts_params():
  params = tsfresh.feature_extraction.ComprehensiveFCParameters()
  del params["fft_aggregated"]
  del params["fft_coefficient"]
  del params["symmetry_looking"]
  return params


def load_data():
  ds = Dataset()
  df = ds.as_dataframe(FLAGS.data_type)
  df = df.select_dtypes(include=["number", "bool"])
  df = df.dropna()
  return df


def main(argv):
  del argv  # not used
  data_dir = pathlib.Path(__file__).parent / "data"
  df = load_data()

  ids = df[FLAGS.id_col].unique()
  for i, chunk in enumerate(more_itertools.chunked(ids, FLAGS.chunk_size)):
    cur_df = df[df[FLAGS.id_col].isin(chunk)]
    cur_ts_df = tsfresh.extract_features(
      cur_df,
      column_id=FLAGS.id_col,
      column_sort=FLAGS.time_col,
      disable_progressbar=True,
      default_fc_parameters=get_ts_params()
    )
    cur_ts_df.index.rename(FLAGS.id_col, inplace=True)
    cur_ts_df.reset_index(inplace=True)
    file_name = f"{FLAGS.data_type}_autofeat_ts_{i}.feather"
    cur_ts_df.to_feather(data_dir / file_name)


if __name__ == "__main__":
  app.run(main)
