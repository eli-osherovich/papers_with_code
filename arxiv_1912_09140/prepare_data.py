#!/usr/bin/env python3
import argparse
import os

import movielens_data


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--dataset",
    type=str,
    default="latest-small",
    choices=["1m", "10m", "20m", "25m", "latest-small", "latest"],
    help="Name of the Movielens dataset, e.g. 100k-ratings, 1m-ratings")

  parser.add_argument(
    "--cache-dir", type=str, default="datasets", help="Cache directory")

  args = parser.parse_args()
  return args


def main():
  args = parse_args()
  ds_zip = movielens_data.download_dataset(args.dataset, args.cache_dir)
  ds = movielens_data.MovieLens(ds_zip)

  output_dir = os.path.join(os.path.dirname(ds_zip), "users")
  os.makedirs(output_dir, exist_ok=True)

  for k, v in ds.full_table.groupby("user"):
    file_name = os.path.join(output_dir, f"{k}.csv")
    v.drop(["user", "item", "timestamp", "title"], axis=1, inplace=True)
    v.to_csv(file_name, index=False)


if __name__ == "__main__":
  main()
