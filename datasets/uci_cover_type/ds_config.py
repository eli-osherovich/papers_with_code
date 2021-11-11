"""UCI Cover Type dataset config.
"""
import numpy as np
import pandas as pd

from .. import dataset

# flake8: noqa: E501
# pylint: disable=line-too-long

INT_TYPE = np.int32
FLOAT_TYPE = np.float32
BOOL_TYPE = np.int32
CATEGORICAL_TYPE = pd.CategoricalDtype

feature_dict = {
  "Elevation": FLOAT_TYPE,  # Elevation in meters
  "Aspect": FLOAT_TYPE,  # Aspect in degrees azimuth
  "Slope": FLOAT_TYPE,  # Slope in degrees
  "Horizontal_Distance_To_Hydrology":
    FLOAT_TYPE,  # Horz Dist to nearest surface water features
  "Vertical_Distance_To_Hydrology":
    FLOAT_TYPE,  # Vert Dist to nearest surface water features
  "Horizontal_Distance_To_Roadways": FLOAT_TYPE,  # Horz Dist to nearest roadway
  "Hillshade_9am":
    FLOAT_TYPE,  # Hillshade index at 9am, summer solstice [0, 255]
  "Hillshade_Noon":
    FLOAT_TYPE,  # Hillshade index at noon, summer soltice [0, 255]
  "Hillshade_3pm": FLOAT_TYPE,  # Hillshade index at 3pm, summer solstice
  "Horizontal_Distance_To_Fire_Points":
    FLOAT_TYPE,  # Horz Dist to nearest wildfire ignition points
  "Wilderness_Area_Rawah": BOOL_TYPE,  # Rawah wilderness area (binary)
  "Wilderness_Area_Neota": BOOL_TYPE,  # Neota Wilderness Area (binary)
  "Wilderness_Area_Comanche": BOOL_TYPE,  # Comanche Wilderness Area (binary)
  "Wilderness_Area_CahceLaPoudre":
    BOOL_TYPE,  # Cache la Poudre Wilderness Area (binary)
  "SoilType_0": BOOL_TYPE,
  "SoilType_1": BOOL_TYPE,
  "SoilType_2": BOOL_TYPE,
  "SoilType_3": BOOL_TYPE,
  "SoilType_4": BOOL_TYPE,
  "SoilType_5": BOOL_TYPE,
  "SoilType_6": BOOL_TYPE,
  "SoilType_7": BOOL_TYPE,
  "SoilType_8": BOOL_TYPE,
  "SoilType_9": BOOL_TYPE,
  "SoilType_10": BOOL_TYPE,
  "SoilType_11": BOOL_TYPE,
  "SoilType_12": BOOL_TYPE,
  "SoilType_13": BOOL_TYPE,
  "SoilType_14": BOOL_TYPE,
  "SoilType_15": BOOL_TYPE,
  "SoilType_16": BOOL_TYPE,
  "SoilType_17": BOOL_TYPE,
  "SoilType_18": BOOL_TYPE,
  "SoilType_19": BOOL_TYPE,
  "SoilType_20": BOOL_TYPE,
  "SoilType_21": BOOL_TYPE,
  "SoilType_22": BOOL_TYPE,
  "SoilType_23": BOOL_TYPE,
  "SoilType_24": BOOL_TYPE,
  "SoilType_25": BOOL_TYPE,
  "SoilType_26": BOOL_TYPE,
  "SoilType_27": BOOL_TYPE,
  "SoilType_28": BOOL_TYPE,
  "SoilType_29": BOOL_TYPE,
  "SoilType_30": BOOL_TYPE,
  "SoilType_31": BOOL_TYPE,
  "SoilType_32": BOOL_TYPE,
  "SoilType_33": BOOL_TYPE,
  "SoilType_34": BOOL_TYPE,
  "SoilType_35": BOOL_TYPE,
  "SoilType_36": BOOL_TYPE,
  "SoilType_37": BOOL_TYPE,
  "SoilType_38": BOOL_TYPE,
  "SoilType_39": BOOL_TYPE,
  "CoverType": INT_TYPE,
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
