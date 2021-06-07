#!/usr/bin/env python3
import argparse
import logging

import tensorflow as tf
import tensorflow_datasets as tfds

import tree


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
      "--keep-best", default=False, dest="keep_best", action="store_true")

  args = parser.parse_args()
  return args


def main():
  args = parse_args()

  train_ds, val_ds = tfds.load(
      "mnist", split=["train", "test"], as_supervised=True)

  train_ds = train_ds.map(lambda x, y: (x / 255, y)).cache().shuffle(
      args.shuffle_buffer).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

  val_ds = val_ds.map(lambda x, y: (x / 255, y)).cache().batch(
      args.batch_size).prefetch(tf.data.AUTOTUNE)

  model = tree.TreeModel(args.depth)

  early_stop = tf.keras.callbacks.EarlyStopping(
      monitor="val_acc",
      patience=args.patience,
      restore_best_weights=args.keep_best)

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
