# Python API Reference

Reference documentation for the classes and functions in the `h5json` Python
package (`src/h5json/`). This covers the library internals — the in-memory
object model, the pluggable storage backends, and the supporting utility
modules — as distinct from the [CLI tools](../tools/h5json.md) and the
[HDF5/JSON schema](../schema/index.md) itself.

Only public classes, methods, and functions (names not starting with `_`) are
documented.

## Core

```{toctree}
:maxdepth: 1

hdf5db
storage_plugin
```

## Types and Selections

```{toctree}
:maxdepth: 1

hdf5dtype
selections
```

## Utilities

```{toctree}
:maxdepth: 1

array_util
dset_util
shape_util
filters
query_util
objid
link_util
time_util
track_util
h5py_util
```

## Storage Backends

```{toctree}
:maxdepth: 1

h5pystore/h5py_plugin
jsonstore/h5json_plugin
```
