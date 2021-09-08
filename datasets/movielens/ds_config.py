import numpy as np
import pandas as pd

_INT_TYPE = np.int32
_FLOAT_TYPE = np.float32
_BOOL_TYPE = np.uint8
_CATEGORICAL_TYPE = pd.api.types.CategoricalDtype

feature_dict = {
  'user':
    _INT_TYPE,
  'item':
    _INT_TYPE,
  'rating':
    _FLOAT_TYPE,
  'timestamp':
    _INT_TYPE,
  'title':
    str,
  'release date':
    str,
  'video release date':
    str,
  'release year':
    _FLOAT_TYPE,
  'IMDb URL':
    str,
  'genres':
    str,
  'unknown':  # coerce with '(no genres listed)' in later datasets.
    _BOOL_TYPE,
  'Action':
    _BOOL_TYPE,
  'Adventure':
    _BOOL_TYPE,
  'Animation':
    _BOOL_TYPE,
  'Children':
    _BOOL_TYPE,
  'Comedy':
    _BOOL_TYPE,
  'Crime':
    _BOOL_TYPE,
  'Documentary':
    _BOOL_TYPE,
  'Drama':
    _BOOL_TYPE,
  'Fantasy':
    _BOOL_TYPE,
  'Film-Noir':
    _BOOL_TYPE,
  'Horror':
    _BOOL_TYPE,
  'IMAX':
    _BOOL_TYPE,
  'Musical':
    _BOOL_TYPE,
  'Mystery':
    _BOOL_TYPE,
  'Romance':
    _BOOL_TYPE,
  'Sci-Fi':
    _BOOL_TYPE,
  'Thriller':
    _BOOL_TYPE,
  'War':
    _BOOL_TYPE,
  'Western':
    _BOOL_TYPE,
  'age':
    _INT_TYPE,
  'gender':
    _CATEGORICAL_TYPE(['M', 'F']),
  'occupation':
    _CATEGORICAL_TYPE([
      'administrator',
      'artist',
      'doctor',
      'educator',
      'engineer',
      'entertainment',
      'executive',
      'healthcare',
      'homemaker',
      'lawyer',
      'librarian',
      'marketing',
      'none',
      'other',
      'programmer',
      'retired',
      'salesman',
      'scientist',
      'student',
      'technician',
      'writer',
    ]),
  'occupation_num':
    _INT_TYPE,
  'zipcode':
    str  # A couple of zipcodes are not numerical...
}
