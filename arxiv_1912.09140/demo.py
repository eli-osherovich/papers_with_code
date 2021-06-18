#!/usr/bin/env python3

import argparse

import tensorflow as tf

import movielens_data
import tree


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--dataset",
      type=str,
      default="latest-small",
      choices=["1m", "10m", "20m", "25m", "latest-small", "latest"],
      help="Name of the Movielens dataset, e.g. '20m' or 'latest-small'")

  parser.add_argument(
      "--depth",
      type=int,
      default=5,
      help="Tree depth (including root and leaves)")

  parser.add_argument(
      "-e", "--epochs", type=int, default=2000, help="Maximal number of epochs")

  parser.add_argument(
      "-r", "--ru", type=int, default=4, help="Ratings per user in each batch")

  parser.add_argument(
      "-u", "--ub", type=int, default=1024, help="Users per batch")

  parser.add_argument("--emb", default=512, help="Embedding dimension")

  parser.add_argument(
      "--patience", type=int, default=200, help="Early stopping patience")

  parser.add_argument(
      "--keep-last",
      default=False,
      dest="keep_last",
      action="store_true",
      help="Keep last model (by default) the best model is kept")

  parser.add_argument(
      "--cache-dir", type=str, default="datasets", help="Cache directory")

  args = parser.parse_args()
  return args


def main():
  args = parse_args()
  ds_zip = movielens_data.download_dataset(args.dataset, args.cache_dir)
  ds, nusers, actual_users_per_batch, rating_min, rating_max = movielens_data.prepare_dataset(
      ds_zip, args.ru, args.ub)

  model = tree.TreeModel(args.depth, ds.element_spec[0][0].shape[-1], args.emb)
  early_stop = tf.keras.callbacks.EarlyStopping(
      monitor='root_mean_squared_error',
      patience=args.patience,
      restore_best_weights=not args.keep_last)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss='mse',
      metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
  model.fit(
      ds,
      epochs=args.epochs,
      steps_per_epoch=nusers // actual_users_per_batch,
      callbacks=[early_stop])

  print(f"Reminder: RMSE should be multiplied by {rating_max - rating_min}")


if __name__ == "__main__":
  main()
