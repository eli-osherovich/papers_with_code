import numpy as np
import pandas as pd

from .. import dataset
from .. import utils

INT_TYPE = np.int32
FLOAT_TYPE = np.float32
BOOL_TYPE = np.int32
CATEGORICAL_TYPE = pd.CategoricalDtype

feature_dict = {
  "checking_balance":
    CATEGORICAL_TYPE([
      "< 0 DM",
      "1 - 200 DM",
      "> 200 DM",
      "unknown",
    ]),
  "months_loan_duration":
    INT_TYPE,
  "credit_history":
    CATEGORICAL_TYPE([
      "critical",
      "good",
      "poor",
      "perfect",
      "very good",
    ]),
  "purpose":
    CATEGORICAL_TYPE([
      "furniture/appliances",
      "education",
      "car",
      "business",
      "renovations",
      "car0",
    ]),
  "amount":
    INT_TYPE,
  "savings_balance":
    CATEGORICAL_TYPE([
      "< 100 DM",
      "100 - 500 DM",
      "500 - 1000 DM",
      "> 1000 DM",
      "unknown",
    ]),
  "employment_duration":
    CATEGORICAL_TYPE([
      "unemployed",
      "< 1 year",
      "1 - 4 years",
      "4 - 7 years",
      "> 7 years",
    ]),
  "percent_of_income":
    FLOAT_TYPE,
  "years_at_residence":
    INT_TYPE,
  "age":
    INT_TYPE,
  "other_credit":
    CATEGORICAL_TYPE([
      "none",
      "bank",
      "store",
    ]),
  "housing":
    CATEGORICAL_TYPE([
      "own",
      "rent",
      "other",
    ]),
  "existing_loans_count":
    INT_TYPE,
  "job":
    CATEGORICAL_TYPE([
      "unemployed",
      "unskilled",
      "skilled",
      "management",
    ]),
  "dependents":
    INT_TYPE,
  "phone":
    BOOL_TYPE,
  "default":
    BOOL_TYPE,
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
