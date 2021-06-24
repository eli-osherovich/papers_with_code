#!/usr/bin/env python3

# Python's relative import was invented by pure PERVERTS.
if __name__ == "__main__" and __package__ is None:
  import os
  import sys
  module_path = os.path.abspath(os.path.join(__file__, ".."))
  module_name = os.path.basename(module_path)
  pkg_path = os.path.abspath(os.path.join(__file__, "..", ".."))
  pkg_name = os.path.basename(pkg_path)
  sys.path.append(os.path.join(pkg_path, ".."))
  __package__ = f"{pkg_name}.{module_name}"  # noqa A001

import argparse

import tensorflow as tf
import tensorflow_datasets as tfds

from . import tree


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--depth",
    type=int,
    default=9,
    help="Tree depth (including root and leaves)")
  parser.add_argument(
    "--epochs", type=int, default=1000, help="Number of epochs")
  parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
  parser.add_argument(
    "--shuffle-buffer", type=int, default=60000, help="Shuffle buffer size")
  parser.add_argument(
    "--patience", type=int, default=10, help="Early stopping patience")
  parser.add_argument(
    "--keep-last",
    default=False,
    dest="keep_last",
    action="store_true",
    help="Keep last model (by default) the best model is kept")

  args = parser.parse_args()
  return args


def main():
  args = parse_args()

  train_ds, val_ds = tfds.load(
    "mnist", split=["train", "test"], shuffle_files=False, as_supervised=True)

  train_ds = train_ds\
    .map(lambda x, y: (x / 255, y), num_parallel_calls=tf.data.AUTOTUNE)\
    .cache()\
    .shuffle(args.shuffle_buffer)\
    .batch(args.batch_size)\
    .prefetch(tf.data.AUTOTUNE)

  val_ds = val_ds\
    .map(lambda x, y: (x / 255, y), num_parallel_calls=tf.data.AUTOTUNE)\
    .batch(args.batch_size)\
    .cache()\
    .prefetch(tf.data.AUTOTUNE)

  model = tree.TreeModel(args.depth)

  early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_acc",
    patience=args.patience,
    restore_best_weights=not args.keep_last)

  model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["acc"])
  model.fit(
    train_ds,
    epochs=args.epochs,
    validation_data=val_ds,
    callbacks=[early_stop])
  model.save("model")


if __name__ == "__main__":
  main()
