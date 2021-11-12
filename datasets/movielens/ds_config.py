from ..typing import BoolType
from ..typing import CategoricalType
from ..typing import FloatType
from ..typing import IntType

# flake8: noqa: E501
# pylint: disable=line-too-long

feature_dict = {
  "user":
    IntType,
  "item":
    IntType,
  "rating":
    FloatType,
  "timestamp":
    IntType,
  "title":
    str,
  "release date":
    str,
  "video release date":
    str,
  "release year":
    FloatType,
  "IMDb URL":
    str,
  "genres":
    str,
  "unknown":
    BoolType,  # coerce with '(no genres listed)' in later datasets.
  "Action":
    BoolType,
  "Adventure":
    BoolType,
  "Animation":
    BoolType,
  "Children":
    BoolType,
  "Comedy":
    BoolType,
  "Crime":
    BoolType,
  "Documentary":
    BoolType,
  "Drama":
    BoolType,
  "Fantasy":
    BoolType,
  "Film-Noir":
    BoolType,
  "Horror":
    BoolType,
  "IMAX":
    BoolType,
  "Musical":
    BoolType,
  "Mystery":
    BoolType,
  "Romance":
    BoolType,
  "Sci-Fi":
    BoolType,
  "Thriller":
    BoolType,
  "War":
    BoolType,
  "Western":
    BoolType,
  "age":
    IntType,
  "gender":
    CategoricalType(["M", "F"]),
  "occupation":
    CategoricalType([
      "administrator",
      "artist",
      "doctor",
      "educator",
      "engineer",
      "entertainment",
      "executive",
      "healthcare",
      "homemaker",
      "lawyer",
      "librarian",
      "marketing",
      "none",
      "other",
      "programmer",
      "retired",
      "salesman",
      "scientist",
      "student",
      "technician",
      "writer",
    ]),
  "occupation_num":
    IntType,
  "zipcode":
    str,  # A couple of zipcodes are not numerical...
}
