"""Common interface to Dataset loading."""
from absl import flags

from ...datasets import common as ds_common
from ...datasets import dataset

FLAGS = flags.FLAGS

{%- macro dataset_for_task(task)-%}
{%if task == "regression" %}"Diabetes"{% endif -%}
{%if task == "binary" %}"BreastCancer"{% endif -%}
{%if task == "multiclass" %}"Iris"{% endif -%}
{%- endmacro %}

flags.DEFINE_string(
  "dataset",
  default={{dataset_for_task(cookiecutter.task)}},
  help="Dataset to use",
)

flags.DEFINE_string(
  "train_name",
  default="train",
  help="Name of the train config",
)


def get_numpy() -> dataset.NP_RESULT:
  return ds_common.load_numpy(FLAGS.dataset, FLAGS.train_name)


def get_dataframe() -> dataset.DF_RESULT:
  return ds_common.load_dataframe(FLAGS.dataset, FLAGS.train_name)


def get_dataset() -> dataset.DS_RESULT:
  return ds_common.load_dataset(FLAGS.dataset, FLAGS.train_name)
