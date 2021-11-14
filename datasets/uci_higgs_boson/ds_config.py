"""UCI Higgs Boson dataset config.
"""

from .. import dataset
from .. import utils
from ..typing import BoolType
from ..typing import FloatType

# flake8: noqa: E501
# pylint: disable=line-too-long

feature_dict = {
  "target": BoolType,
  "lepton pT": FloatType,
  "lepton eta": FloatType,
  "lepton phi": FloatType,
  "missing energy magnitude": FloatType,
  "missing energy phi": FloatType,
  "jet 1 pt": FloatType,
  "jet 1 eta": FloatType,
  "jet 1 phi": FloatType,
  "jet 1 b-tag": FloatType,
  "jet 2 pt": FloatType,
  "jet 2 eta": FloatType,
  "jet 2 phi": FloatType,
  "jet 2 b-tag": FloatType,
  "jet 3 pt": FloatType,
  "jet 3 eta": FloatType,
  "jet 3 phi": FloatType,
  "jet 3 b-tag": FloatType,
  "jet 4 pt": FloatType,
  "jet 4 eta": FloatType,
  "jet 4 phi": FloatType,
  "jet 4 b-tag": FloatType,
  "m_jj": FloatType,
  "m_jjj": FloatType,
  "m_lv": FloatType,
  "m_jlv": FloatType,
  "m_bb": FloatType,
  "m_wbb": FloatType,
  "m_wwbb": FloatType
}
SPLITS = {
  "train":
    dataset.DatasetFile(
      uri=utils.make_uri(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
      ),
      checksum="ea302c18164d4e3d916a1e2e83a9a8d07069fa6ebc7771e4c0540d54e593b698",
    ),
}
