##############
h5json Package
##############

This package provides CLI tools for conversion between HDF5 files and HDF5/JSON,
and HDF5/JSON validation.

Installation
============

The standard method using the *pip* tool is recommended::

  pip install h5json

If interested in an unreleased version, install directly from the repository::

  pip install git+https://github.com/HDFGroup/hdf5-json.git@{LABEL}

``{LABEL}`` is a branch, tag, or commit identifier. *pip* `documentation
<https://pip.pypa.io/en/stable/cli/pip_install/#pip-install-examples>`_ explains
available install features.

For development, create a fork of the `repository
<https://github.com/HDFGroup/hdf5-json.git>`_  and execute::

  $ mkdir my-h5json-dev
  $ cd my-h5json-dev
  $ git clone https://{MYUSERNAME}@github.com/{MYUSERNAME}/hdf5-json.git
  $ cd h5json
  $ pip install -e .

`GitHub documentation
<https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories>`_
explains this workflow in great detail.

Verification
------------

To verify h5json was installed correctly convert an HDF5 file to HDF5/JSON and
back. Run the following commands:

.. code-block:: shell

  $ h5tojson sample.h5 > sample.json
  $ h5jvalidate sample.json
  $ jsontoh5 sample.json new-sample.h5

The file ``sample.json`` should contain HDF5/JSON description of the original
file and the file ``new-sample.h5`` should be an HDF5 equivalent to the original
file ``sample.h5``.


CLI Tools
=========

The h5json distribution provides three command-line tools described below.

jsontoh5
--------

Generate an HDF5 file with the content, storage features, and data described in
an HDF5/JSON file.

Usage::

  jsontoh5.py [-h] <json_file> <h5_file>

where:

``<json_file>``
  Input HDF5/JSON file.

``<h5_file>``
  Output HDF5 file that will be created.

Options:

-h
  Print help message.

h5tojson
--------

Convert the input HDF5 file to its HDF5/JSON representation. Output is printed
to `stdout`.

Usage::

  h5tojson [-h] [-D] [-d] <hdf5_file>

where:

``<hdf5_file>``
  HDF5 file.

Options:

-h
  Print help message.
-D
  Suppress all data output. Output HDF5/JSON will not contain any dataset or
  attribute values.
-d
  Suppress data output for datasets only.


h5jvalidate
-----------

Validate generated HDF5/JSON files against the schema. Validation errors are
printed to `stderr`. Command's exit status indicates the overall success (``0``)
or failure (``1``).

Usage::

  h5jvalidate [-h|--help] [-s|--stop] JSON_LOC [JSON_LOC ...]

where:

``JSON_LOC``
  HDF5/JSON location (files or folders). If a folder, all files with ``.json``
  extension will be selected for validation.

Options:

-s, --stop
  Stop after first HDF5/JSON file failed validation (default: False)
-h, --help
  Print help message.
