# h5py_util

Predicates and dtype conversion helpers for bridging h5py's "special dtype" metadata for object/region
references and variable-length types with h5json's own equivalents defined in `hdf5dtype.py`. This is
needed because h5py identity-checks reference dtypes against its own `h5py.Reference`/
`h5py.RegionReference` classes, so a reference dtype produced by `hdf5dtype` must be translated to h5py's
own classes before being handed to h5py, and vice versa.

## is_reference(val)

Returns `True` if `val` is an instance of, or is itself, a class named `"Reference"` (checked by class
name rather than an `isinstance` check against a specific `Reference` class, so it matches either
h5py's or h5json's `Reference` type).

## is_regionreference(val)

Same as `is_reference` but matches the class name `"RegionReference"`.

## has_reference(dtype)

Returns `True` if `dtype` (a `numpy.dtype`) is, or contains, a Reference or RegionReference type. For a
compound dtype, recurses into each field; for a leaf dtype, checks `dtype.metadata["ref"]` (matched via
`is_reference`/`is_regionreference`) or recurses into `dtype.metadata["vlen"]` for variable-length types.
Returns `False` for non-`np.dtype` input.

## convert_dtype(srcdt, to_h5py=True)

Recursively converts `srcdt` between h5py's special-dtype representation and h5json's own
(`hdf5dtype.special_dtype`), returning the converted `numpy.dtype`. Compound dtypes are rebuilt
field-by-field. For a leaf dtype: a `"ref"` metadata entry is rebuilt using `h5py.special_dtype(ref=...)`
or `hdf5dtype.special_dtype(ref=...)` depending on `to_h5py`, with the reference class chosen via
`is_reference`/`is_regionreference` (raising `TypeError` if it's neither); a `"vlen"` metadata entry is
rebuilt similarly, recursively converting the base type if it is itself a dtype; and a plain `"U"`-kind
(unicode) dtype is converted to a variable-length string special dtype. Any other dtype is returned
unchanged.
