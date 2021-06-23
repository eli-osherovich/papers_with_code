import logging

import numpy as np


def roundrobin_generator(arr, batch=1, rng=np.random.default_rng()):
  assert isinstance(
    batch,
    (int, np.integer)), f"Batch must be an integral type, got {type(batch)}"
  assert batch > 0, f"Batch must be strictly positive, got {batch}"

  arr_len = len(arr)

  if batch > arr_len:
    logging.warning(
      "Batch size %d is larger than the number of elements in the sequence: %d",
      batch,
      arr_len,
    )

  multiplicity = np.ceil(batch / arr_len)
  logging.info("Maximal element multiplicity in a batch: %d", multiplicity)

  all_indices = np.repeat(np.arange(arr_len), multiplicity)
  all_indices_len = len(all_indices)
  cur_perm = rng.permutation(all_indices)

  while True:
    start = 0
    end = batch
    used_idx_batches = []
    while end < all_indices_len:
      batch_idx = cur_perm[start:end]
      used_idx_batches.append(batch_idx)
      yield arr[cur_perm[start:end]]
      start, end = end, end + batch
    used_indices = np.concatenate(used_idx_batches)
    cur_perm = np.concatenate((cur_perm[start:], rng.permutation(used_indices)))
