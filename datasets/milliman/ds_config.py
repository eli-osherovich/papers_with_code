from .. import dataset
from .. import utils
from ..typing import BoolType
from ..typing import CategoricalType
from ..typing import FloatType
from ..typing import IntType

# flake8: noqa: E501
# pylint: disable=line-too-long

feature_dict = {
  "checking_balance":
    CategoricalType([
      "< 0 DM",
      "1 - 200 DM",
      "> 200 DM",
      "unknown",
    ]),
  "months_loan_duration":
    IntType,
  "credit_history":
    CategoricalType([
      "critical",
      "good",
      "poor",
      "perfect",
      "very good",
    ]),
  "purpose":
    CategoricalType([
      "furniture/appliances",
      "education",
      "car",
      "business",
      "renovations",
      "car0",
    ]),
  "amount":
    IntType,
  "savings_balance":
    CategoricalType([
      "< 100 DM",
      "100 - 500 DM",
      "500 - 1000 DM",
      "> 1000 DM",
      "unknown",
    ]),
  "employment_duration":
    CategoricalType([
      "unemployed",
      "< 1 year",
      "1 - 4 years",
      "4 - 7 years",
      "> 7 years",
    ]),
  "percent_of_income":
    FloatType,
  "years_at_residence":
    IntType,
  "age":
    IntType,
  "other_credit":
    CategoricalType([
      "none",
      "bank",
      "store",
    ]),
  "housing":
    CategoricalType([
      "own",
      "rent",
      "other",
    ]),
  "existing_loans_count":
    IntType,
  "job":
    CategoricalType([
      "unemployed",
      "unskilled",
      "skilled",
      "management",
    ]),
  "dependents":
    IntType,
  "phone":
    BoolType,
  "default":
    BoolType,
}

SPLITS = {
  "train":
    dataset.DatasetFile(
      uri=utils.make_uri("milliman.csv"),
      checksum="553471e860ac5512a4c0c52c7b55340e2a8b58f84eb504171516aa3b58b368f4",
      file_reader_args={
        "sep": "\\s+,",
        "header": 0,
        "engine": "python",
        "true_values": ["yes"],
        "false_values": ["no"],
      }
    )
}
