import os
from typing import Optional

from absl import logging
import numpy as np
import psutil
import tensorflow as tf


def roundrobin_generator(arr, batch_size=1, rng=np.random.default_rng()):
  assert isinstance(
    batch_size, (int, np.integer)
  ), f"Batch must be an integral type, got {type(batch_size)}"

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


def make_divisible(q, values, dtype=None):
  """Adjust `values` to closes values divisible by `q`."""
  return (np.asanyarray(values, dtype=dtype) / q).round() * q


def setup_omp():
  """Optimize environment for multi-core machines"""
  """A set of setting that should optimize multi-core environments.
  See following references:
  https://software.intel.com/content/www/us/en/develop/articles/tips-to-improve-performance-for-popular-deep-learning-frameworks-on-multi-core-cpus.html
  https://software.intel.com/content/www/us/en/develop/articles/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference.html
  """
  num_cores = psutil.cpu_count(logical=False)
  num_threads = psutil.cpu_count(logical=True)
  os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
  os.environ["KMP_BLOCKTIME"] = "0"
  os.environ["OMP_DYNAMIC"] = "TRUE"
  os.environ["OMP_NUM_THREADS"] = str(num_cores)
  os.environ["OMP_SCHEDULE"] = "DYNAMIC"
  # tf.config.threading.set_intra_op_parallelism_threads(num_cores)
  tf.config.threading.set_inter_op_parallelism_threads(num_threads)


@tf.function
def weighted_row_sum(
  x: tf.Tensor, weights: tf.Tensor, keepdims: float = False
) -> tf.Tensor:

  tf.ensure_shape(x, (None, None))
  tf.ensure_shape(weights, (None, 1))

  weights = tf.cast(weights, tf.keras.backend.floatx())
  sum_ = tf.math.reduce_sum(weights)
  weights = weights * tf.math.reciprocal_no_nan(sum_)
  res = tf.math.reduce_sum(weights * x, axis=0, keepdims=keepdims)
  return res
