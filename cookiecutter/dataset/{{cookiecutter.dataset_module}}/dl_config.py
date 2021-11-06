"""{{cookiecutter.dataset}} download config.
"""

raise NotImplementedError("Please review the data below and remove this exception")
{%- macro split_config(name, uri, checksum="FIXME: add real checksum") -%}
{%- if uri -%}
  "{{name}}": {
    "uri": "{{uri}}",
    "checksum": "{{checksum}}",
  },
{%- endif -%}
{%- endmacro %}

DATASETS = {
  {{split_config("train", cookiecutter.train_file_uri)}}
  {{split_config("val", cookiecutter.val_file_uri)}}
  {{split_config("test", cookiecutter.test_file_uri)}}
}
