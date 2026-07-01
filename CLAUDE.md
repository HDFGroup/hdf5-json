# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`h5json` is a Python package for bidirectional conversion between HDF5 files and a JSON representation. It provides CLI tools (`h5tojson`, `jsontoh5`, `h5jvalidate`), a library API, and JSON Schema definitions for the h5json format.

## Commands

**Install for development:**
```bash
pip install -e .
```

**Run all tests:**
```bash
python testall.py
```

**Run a single unit test:**
```bash
python test/unit/<test_name>.py
# e.g. python test/unit/hdf5db_test.py
```

**Run integration tests (from repo root):**
```bash
cd test/integ && python h5tojson_test.py
cd test/integ && python jsontoh5_test.py
```

**Build distribution:**
```bash
python -m build
```

**Linting:** flake8 with max line length 120 (E402, C901, F401 ignored). See `setup.cfg`.

## Architecture

The core design separates storage backends from the in-memory object model via abstract base classes.

### Central Class: `Hdf5db` (`src/h5json/hdf5db.py`)

`Hdf5db` is an in-memory store for HDF5 objects (groups, datasets, committed datatypes). It tracks object state (new, dirty, deleted) and delegates I/O to pluggable reader/writer instances. Opening reads from the reader into `_db`; closing flushes to the writer.  Updates to the object state live in memory until the next flush.  Reads from Hdfdb must take into account 'dirty' objects that have not yet been flushed to memory.

### Storage Backends

Abstract base classes in `src/h5json/h5reader.py` and `src/h5json/h5writer.py` define the interface. Two concrete pairs exist:

| Backend | Reader | Writer | Storage |
|---------|--------|--------|---------|
| h5py | `h5pystore/h5py_reader.py` | `h5pystore/h5py_writer.py` | `.h5` HDF5 files via h5py |
| json | `jsonstore/h5json_reader.py` | `jsonstore/h5json_writer.py` | h5json `.json` files |

### CLI Apps (`src/h5json/apps/`)

Each app wires a reader to a writer through `Hdf5db`:
- `h5tojson`: `H5pyReader` → `H5JsonWriter` — converts HDF5 to JSON
- `jsontoh5`: `H5JsonReader` → `H5pyWriter` — converts JSON to HDF5
- `validator.py`: validates a JSON file against the h5json JSON Schema

### Key Utility Modules

- `hdf5dtype.py` — bidirectional mapping between HDF5/numpy dtypes and h5json type descriptors; exports `getTypeItem`, `createDataType`, `Reference`
- `selections.py` — HDF5 dataspace selection types (hyperslabs, point selections, fancy indexing); used by both reader backends
- `array_util.py` — conversion between JSON array representations and numpy arrays
- `query_util.py` — expression parser and evaluator for dataset value queries (used by `Hdf5db.getDatasetValuesByUuid`)
- `filters.py` — HDF5 compression filter handling
- `objid.py` — UUID generation and classification for HDF5 object IDs (Schema 1 vs Schema 2)
- `shape_util.py`, `dset_util.py` — helpers for dataspace and dataset operations

### JSON Schema

`src/h5json/schema/` contains JSON Schema files defining the h5json format (`hdf5.schema.json` is the root, with sub-schemas for groups, datasets, datatypes, etc.).

### Test Data

`data/hdf5/` contains a large set of `.h5` test files covering various HDF5 features. Integration tests use `data/hdf5/` as input and write output to `test/integ/h5_out/` and `test/integ/json_out/`.

Tests use `unittest` (not pytest) and are run directly with `python`.

## Do / Don't
- When making a change to any of the h5json/src/*_util.py files, update the corresponding unit test (e.g. if modifying src/h5json/array_util.py, the test would be test/unit/array_util_test.py) to validate the change
- Be cafeful with memory usage - dataset spaces can be arbitrary large.  Avoid reading all the dataset data into memory unless it's known to be relatively small (a few KB)
- When making changes to hdf5db, first add a test to test/unit/hdf5db_test.py to verify functionality without any storage backend.  Once those tests pass, add tests for the json and h5py storage backends
- For tests with the storage backends, it's important to close the db and re-open, and then verify that any changes made to the model were persisted to storage correctly
- Files (HDF5 or JSON) created by the test cases should be in the test/unit/out directory.  These should always be created by the test case (never re-opened from a previous run)
- Files in data/hdf5/ and data/json/ should oly be modified if there is a change to the schema
- DON'T read/write datasets data element by element if it can be avoided.  Rely on Numpy operations as much as possible for performance
- DON'T make any changes to the schema without authorization
