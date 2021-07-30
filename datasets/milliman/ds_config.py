import numpy as np
import pandas as pd

_INT_TYPE = np.int32
_FLOAT_TYPE = np.float32
_BOOL_TYPE = np.int32
_CATEGORICAL_TYPE = pd.api.types.CategoricalDtype

feature_dict = {
  'checking_balance':
    _CATEGORICAL_TYPE([
      '< 0 DM',
      '1 - 200 DM',
      '> 200 DM',
      'unknown',
    ]),
  'months_loan_duration':
    _INT_TYPE,
  'credit_history':
    _CATEGORICAL_TYPE([
      'critical',
      'good',
      'poor',
      'perfect',
      'very good',
    ]),
  'purpose':
    _CATEGORICAL_TYPE(
      [
        'furniture/appliances',
        'education',
        'car',
        'business',
        'renovations',
        'car0',
      ]
    ),
  'amount':
    _INT_TYPE,
  'savings_balance':
    _CATEGORICAL_TYPE(
      [
        '< 100 DM',
        '100 - 500 DM',
        '500 - 1000 DM',
        '> 1000 DM',
        'unknown',
      ]
    ),
  'employment_duration':
    _CATEGORICAL_TYPE(
      [
        'unemployed',
        '< 1 year',
        '1 - 4 years',
        '4 - 7 years',
        '> 7 years',
      ]
    ),
  'percent_of_income':
    _FLOAT_TYPE,
  'years_at_residence':
    _INT_TYPE,
  'age':
    _INT_TYPE,
  'other_credit':
    _CATEGORICAL_TYPE([
      'none',
      'bank',
      'store',
    ]),
  'housing':
    _CATEGORICAL_TYPE([
      'own',
      'rent',
      'other',
    ]),
  'existing_loans_count':
    _INT_TYPE,
  'job':
    _CATEGORICAL_TYPE([
      'unemployed',
      'unskilled',
      'skilled',
      'management',
    ]),
  'dependents':
    _INT_TYPE,
  'phone':
    _BOOL_TYPE,
  'default':
    _BOOL_TYPE,
}
