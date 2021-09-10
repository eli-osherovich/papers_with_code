import pathlib

_FILE = pathlib.Path(__file__)
_DS_FILE = "milliman.csv"

ds_path = _FILE.parent / _DS_FILE

DATASETS = {
  "train": {
    "uri":
      ds_path.as_uri(),
    "checksum":
      "553471e860ac5512a4c0c52c7b55340e2a8b58f84eb504171516aa3b58b368f4",
  }
}
