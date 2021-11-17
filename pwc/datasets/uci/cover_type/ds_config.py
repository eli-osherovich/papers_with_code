"""UCI Cover Type dataset config.
"""

from pwc.datasets import dataset
from pwc.datasets.typing import BoolType
from pwc.datasets.typing import FloatType
from pwc.datasets.typing import IntType

# flake8: noqa: E501
# pylint: disable=line-too-long

feature_dict = {
  "Elevation": FloatType,  # Elevation in meters
  "Aspect": FloatType,  # Aspect in degrees azimuth
  "Slope": FloatType,  # Slope in degrees
  "Horizontal_Distance_To_Hydrology":
    FloatType,  # Horz Dist to nearest surface water features
  "Vertical_Distance_To_Hydrology":
    FloatType,  # Vert Dist to nearest surface water features
  "Horizontal_Distance_To_Roadways": FloatType,  # Horz Dist to nearest roadway
  "Hillshade_9am":
    FloatType,  # Hillshade index at 9am, summer solstice [0, 255]
  "Hillshade_Noon":
    FloatType,  # Hillshade index at noon, summer soltice [0, 255]
  "Hillshade_3pm": FloatType,  # Hillshade index at 3pm, summer solstice
  "Horizontal_Distance_To_Fire_Points":
    FloatType,  # Horz Dist to nearest wildfire ignition points
  "Wilderness_Area_Rawah": BoolType,  # Rawah wilderness area (binary)
  "Wilderness_Area_Neota": BoolType,  # Neota Wilderness Area (binary)
  "Wilderness_Area_Comanche": BoolType,  # Comanche Wilderness Area (binary)
  "Wilderness_Area_CahceLaPoudre":
    BoolType,  # Cache la Poudre Wilderness Area (binary)
  "SoilType_0": BoolType,
  "SoilType_1": BoolType,
  "SoilType_2": BoolType,
  "SoilType_3": BoolType,
  "SoilType_4": BoolType,
  "SoilType_5": BoolType,
  "SoilType_6": BoolType,
  "SoilType_7": BoolType,
  "SoilType_8": BoolType,
  "SoilType_9": BoolType,
  "SoilType_10": BoolType,
  "SoilType_11": BoolType,
  "SoilType_12": BoolType,
  "SoilType_13": BoolType,
  "SoilType_14": BoolType,
  "SoilType_15": BoolType,
  "SoilType_16": BoolType,
  "SoilType_17": BoolType,
  "SoilType_18": BoolType,
  "SoilType_19": BoolType,
  "SoilType_20": BoolType,
  "SoilType_21": BoolType,
  "SoilType_22": BoolType,
  "SoilType_23": BoolType,
  "SoilType_24": BoolType,
  "SoilType_25": BoolType,
  "SoilType_26": BoolType,
  "SoilType_27": BoolType,
  "SoilType_28": BoolType,
  "SoilType_29": BoolType,
  "SoilType_30": BoolType,
  "SoilType_31": BoolType,
  "SoilType_32": BoolType,
  "SoilType_33": BoolType,
  "SoilType_34": BoolType,
  "SoilType_35": BoolType,
  "SoilType_36": BoolType,
  "SoilType_37": BoolType,
  "SoilType_38": BoolType,
  "SoilType_39": BoolType,
  "CoverType": IntType,
}

SPLITS = {
  "train":
    dataset.DatasetFile(
      uri="https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
      checksum="614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771",
      file_reader_args={
        "names": feature_dict,
        "dtype": feature_dict
      }
    )
}
