"""UCI Adult dataset config.
"""

from .. import dataset
from .. import utils
from ..typing import CategoricalType
from ..typing import FloatType

# flake8: noqa: E501
# pylint: disable=line-too-long

feature_dict = {
  "age":
    FloatType,
  "workclass":
    CategoricalType([
      "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
      "State-gov", "Without-pay", "Never-worked", "?"
    ]),
  "fnlwgt":
    FloatType,
  "education":
    CategoricalType([
      "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
      "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th",
      "10th", "Doctorate", "5th-6th", "Preschool"
    ]),
  "education-num":
    FloatType,
  "marital-status":
    CategoricalType([
      "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
      "Married-spouse-absent", "Married-AF-spouse"
    ]),
  "occupation":
    CategoricalType([
      "Tech-support", "Craft-repair", "Other-service", "Sales",
      "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
      "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
      "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces",
      "?"
    ]),
  "relationship":
    CategoricalType([
      "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative",
      "Unmarried"
    ]),
  "race":
    CategoricalType([
      "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
    ]),
  "sex":
    CategoricalType(["Female", "Male"]),
  "capital-gain":
    FloatType,
  "capital-loss":
    FloatType,
  "hours-per-week":
    FloatType,
  "native-country":
    CategoricalType([
      "United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
      "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece",
      "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy",
      "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France",
      "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia",
      "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia",
      "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands",
      "?"
    ]),
  "target":
    CategoricalType(["<=50K", ">50K"])
}

SPLITS = {
  "train":
    dataset.DatasetFile(
      uri=utils.make_uri(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
      ),
      checksum="5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
      file_reader_args={
        "names": feature_dict,
        "dtype": feature_dict,
        "sep": r",\s+",
        "engine": "python",  # avoid pandas warning
      }
    ),
  "val":
    dataset.DatasetFile(
      uri=utils.make_uri(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
      ),
      checksum="a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05",
      file_reader_args={
        "names": feature_dict,
        "dtype": feature_dict,
        "skiprows": 1,
        "sep": r",\s+|\.$",  # test file has periods at the end of each line
        "usecols": feature_dict,
        "engine": "python",  # avoid pandas warning
      }
    ),
}
