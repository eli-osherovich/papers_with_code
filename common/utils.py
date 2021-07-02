import logging

import numpy as np
import tensorflow as tf


def roundrobin_generator(arr, batch_size=1, rng=np.random.default_rng()):
  assert isinstance(batch_size, (int, np.integer)), \
    f"Batch must be an integral type, got {type(batch_size)}"

  assert batch_size > 0, f"Batch must be strictly positive, got {batch_size}"

  # Make trivial cases indexable in the way we use it.
  if isinstance(arr, (list, tuple)):
    arr = np.asarray(arr)

  arr_len = len(arr)

  if batch_size > arr_len:
    logging.warning(
      "Batch size %d is larger than the number of elements in the sequence: %d",
      batch_size,
      arr_len,
    )

  multiplicity = np.ceil(batch_size / arr_len)
  logging.info("Maximal element multiplicity in a batch: %d", multiplicity)

  all_indices = np.repeat(np.arange(arr_len), multiplicity)
  all_indices_len = len(all_indices)
  cur_perm = rng.permutation(all_indices)

  while True:
    start = 0
    end = batch_size
    used_idx_batches = []
    while end <= all_indices_len:
      batch_idx = cur_perm[start:end]
      used_idx_batches.append(batch_idx)
      yield arr[cur_perm[start:end]]
      start, end = end, end + batch_size
    used_indices = np.concatenate(used_idx_batches)
    cur_perm = np.concatenate((cur_perm[start:], rng.permutation(used_indices)))


def matrix_to_transform(matrix):
  """Convert a 3x3 affine transformation matrix to a vector"""
  transform = tf.reshape(matrix, [1, -1])
  # The last element is always 1.
  return transform[:, 0:8]
