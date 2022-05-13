#!/usr/bin/env python

import pathlib

from absl import app
from absl import flags
import more_itertools
import tsfresh

from pwc.city import task_lib
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
  # Features selected from a previous (good) model
  # Orered by total importance (descending)
  ts_features = [
    "cwt_coefficients",
    "agg_linear_trend",
    "change_quantiles",
    "partial_autocorrelation",
    "quantile",
    "energy_ratio_by_chunks",
    "autocorrelation",
    "index_mass_quantile",
    "linear_trend",
    "ar_coefficient",
    "time_reversal_asymmetry_statistic",
    "c3",
    "fourier_entropy",
    "percentage_of_reoccurring_datapoints_to_all_datapoints",
    "ratio_value_number_to_time_series_length",
    "lempel_ziv_complexity",
    "sum_of_reoccurring_data_points",
    "sum_of_reoccurring_values",
    "ratio_beyond_r_sigma",
    "first_location_of_maximum",
    "binned_entropy",
    "sample_entropy",
    "spkt_welch_density",
    "first_location_of_minimum",
    "median",
    "percentage_of_reoccurring_values_to_all_values",
    "matrix_profile",
    "approximate_entropy",
    "abs_energy",
    "mean_second_derivative_central",
    "agg_autocorrelation",
    "benford_correlation",
    "mean",
    "augmented_dickey_fuller",
    "variation_coefficient",
    "last_location_of_minimum",
    "cid_ce",
    "inter",
    "root_mean_square",
    "minimum",
    "last_location_of_maximum",
    "skewness",
    "kurtosis",
    "maximum",
    "max_langevin_fixed_point",
    "count_above",
    "mean_abs_change",
    "standard_deviation",
    "mean_change",
    "count_below",
    "has_duplicate_max",
    "permutation_entropy",
    "absolute_sum_of_changes",
    "longest_strike_above_mean",
    "longest_strike_below_mean",
    "sum_values",
    "variance",
    "number_peaks",
  ]
  params = tsfresh.feature_extraction.ComprehensiveFCParameters()
  for key in params:
    if key not in ts_features:
      del params[key]
  return params


def load_data():
  ds = Dataset()
  df = ds.as_dataframe(FLAGS.data_type)
  df = task_lib.set_nans(df)
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
