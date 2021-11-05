"""UCI Cover Type dataset schema.
"""
import numpy as np
import pandas as pd

_INT_TYPE = np.int32
_FLOAT_TYPE = np.float32
_BOOL_TYPE = np.int32
_CATEGORICAL_TYPE = pd.CategoricalDtype

feature_dict = {
  "Elevation": _FLOAT_TYPE,  # Elevation in meters
  "Aspect": _FLOAT_TYPE,  # Aspect in degrees azimuth
  "Slope": _FLOAT_TYPE,  # Slope in degrees
  "Horizontal_Distance_To_Hydrology":
    _FLOAT_TYPE,  # Horz Dist to nearest surface water features
  "Vertical_Distance_To_Hydrology":
    _FLOAT_TYPE,  # Vert Dist to nearest surface water features
  "Horizontal_Distance_To_Roadways":
    _FLOAT_TYPE,  # Horz Dist to nearest roadway
  "Hillshade_9am":
    _FLOAT_TYPE,  # Hillshade index at 9am, summer solstice [0, 255]
  "Hillshade_Noon":
    _FLOAT_TYPE,  # Hillshade index at noon, summer soltice [0, 255]
  "Hillshade_3pm": _FLOAT_TYPE,  # Hillshade index at 3pm, summer solstice
  "Horizontal_Distance_To_Fire_Points":
    _FLOAT_TYPE,  # Horz Dist to nearest wildfire ignition points
  "Wilderness_Area_Rawah": _BOOL_TYPE,  # Rawah wilderness area (binary)
  "Wilderness_Area_Neota": _BOOL_TYPE,  # Neota Wilderness Area (binary)
  "Wilderness_Area_Comanche": _BOOL_TYPE,  # Comanche Wilderness Area (binary)
  "Wilderness_Area_CahceLaPoudre":
    _BOOL_TYPE,  # Cache la Poudre Wilderness Area (binary)
  "SoilType_0": _BOOL_TYPE,
  "SoilType_1": _BOOL_TYPE,
  "SoilType_2": _BOOL_TYPE,
  "SoilType_3": _BOOL_TYPE,
  "SoilType_4": _BOOL_TYPE,
  "SoilType_5": _BOOL_TYPE,
  "SoilType_6": _BOOL_TYPE,
  "SoilType_7": _BOOL_TYPE,
  "SoilType_8": _BOOL_TYPE,
  "SoilType_9": _BOOL_TYPE,
  "SoilType_10": _BOOL_TYPE,
  "SoilType_11": _BOOL_TYPE,
  "SoilType_12": _BOOL_TYPE,
  "SoilType_13": _BOOL_TYPE,
  "SoilType_14": _BOOL_TYPE,
  "SoilType_15": _BOOL_TYPE,
  "SoilType_16": _BOOL_TYPE,
  "SoilType_17": _BOOL_TYPE,
  "SoilType_18": _BOOL_TYPE,
  "SoilType_19": _BOOL_TYPE,
  "SoilType_20": _BOOL_TYPE,
  "SoilType_21": _BOOL_TYPE,
  "SoilType_22": _BOOL_TYPE,
  "SoilType_23": _BOOL_TYPE,
  "SoilType_24": _BOOL_TYPE,
  "SoilType_25": _BOOL_TYPE,
  "SoilType_26": _BOOL_TYPE,
  "SoilType_27": _BOOL_TYPE,
  "SoilType_28": _BOOL_TYPE,
  "SoilType_29": _BOOL_TYPE,
  "SoilType_30": _BOOL_TYPE,
  "SoilType_31": _BOOL_TYPE,
  "SoilType_32": _BOOL_TYPE,
  "SoilType_33": _BOOL_TYPE,
  "SoilType_34": _BOOL_TYPE,
  "SoilType_35": _BOOL_TYPE,
  "SoilType_36": _BOOL_TYPE,
  "SoilType_37": _BOOL_TYPE,
  "SoilType_38": _BOOL_TYPE,
  "SoilType_39": _BOOL_TYPE,
  "CoverType": _INT_TYPE,
}
