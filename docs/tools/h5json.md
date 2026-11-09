# h5json Package

This package provides CLI tools for conversion between HDF5 files and HDF5/JSON,
and HDF5/JSON validation.

## Installation

The standard method using the *pip* tool is recommended:

```
pip install h5json
```

If interested in an unreleased version, install directly from the repository:

```
pip install git+https://github.com/HDFGroup/hdf5-json.git@{LABEL}
```

`{LABEL}` is a branch, tag, or commit identifier. *pip* [documentation](https://pip.pypa.io/en/stable/cli/pip_install/#pip-install-examples) explains
available install features.

For development, create a fork of the [repository](https://github.com/HDFGroup/hdf5-json.git)  and execute:

```shell
$ mkdir my-h5json-dev
$ cd my-h5json-dev
$ git clone https://{MYUSERNAME}@github.com/{MYUSERNAME}/hdf5-json.git
$ cd h5json
$ pip install -e .
```

[GitHub documentation](https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories)
explains this workflow in great detail.

### Verification

To verify h5json was installed correctly convert an HDF5 file to HDF5/JSON and
back. Run the following commands:

```shell
$ h5tojson sample.h5 > sample.json
$ h5jvalidate sample.json
$ jsontoh5 sample.json new-sample.h5
```

The file `sample.json` should contain HDF5/JSON description of the original
file and the file `new-sample.h5` should be an HDF5 equivalent to the original
file `sample.h5`.

## CLI Tools

The h5json distribution provides three command-line tools described below.

By default, `jsontoh5` and `h5tojson` print warnings and errors to `stderr` and
do not write a log file.

### jsontoh5

Generate an HDF5 file with the content, storage features, and data described in
an HDF5/JSON file.

Usage:

```
jsontoh5 [-h] [--nodata] [--log FILE] <json_file> <h5_file>
```

where:

`<json_file>`

: Input HDF5/JSON file.

`<h5_file>`

: Output HDF5 file that will be created. It can't be the same file as
`<json_file>`.

Options:

`-h`

: Print help message.

`--nodata`

: Don't write dataset values. Datasets are created with their fill values.

`--log FILE`

: Append DEBUG output to `FILE`.

### h5tojson

Convert the input HDF5 file to its HDF5/JSON representation. Output is printed
to `stdout`.

Usage:

```
h5tojson [-h] [--nodata] [--data-limit N] [--log FILE] <hdf5_file>
```

where:

`<hdf5_file>`

: HDF5 file.

Options:

`-h`
: Print help message.

`--nodata`
: Omit dataset values. Attribute values are still included. Same as
`--data-limit 0`.

`--data-limit N`
: Omit the values of datasets larger than about `N` bytes.

`--log FILE`
: Append DEBUG output to `FILE`.

### h5jvalidate

Validate generated HDF5/JSON files against the schema. Validation errors are
printed to `stderr`. Command's exit status indicates the overall success (`0`)
or failure (`1`).

Validation is structural only: it checks that a file has the layout the schema
describes. It is not a safety check, and passing it does not mean a file from an
untrusted source is safe to convert with `jsontoh5`.

Usage:

```
h5jvalidate [-h|--help] [-s|--stop] JSON_LOC [JSON_LOC ...]
```

where:

`JSON_LOC`

: HDF5/JSON location (files or folders). If a folder, all files with `.json`
  extension will be selected for validation.

Options:

`-s`, `--stop`
: Stop after first HDF5/JSON file failed validation (default: False)

`-h`, `--help`
: Print help message.
