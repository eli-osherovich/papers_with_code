"""UCI Poker Hand download config.
"""

from .. import utils

DATASETS = {
  "train": {
    "uri":
      utils.make_uri(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data"
      ),
    "checksum":
      "FIXME: add real checksum",
  },
  "val": {
    "uri":
      utils.make_uri(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data"
      ),
    "checksum":
      "FIXME: add real checksum",
  },
}
