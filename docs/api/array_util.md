# array_util

Utility functions for converting between numpy arrays (used internally to represent HDF5 dataset and attribute values) and their JSON-compatible representations. This module is the low-level bridge between the two storage backends: it turns numpy arrays into nested Python lists/scalars suitable for `json.dump` (and back again), handles the byte-level encoding used for variable-length (vlen) types, opaque (`H5T_OPAQUE`) data, and object/region references, and provides a handful of shape/size helper functions used elsewhere in the codebase (e.g. `dset_util.py`). Attribute values in this codebase are always immediately converted to JSON-compatible form via these functions rather than being kept as raw numpy arrays.

## bytesArrayToList(data)

Recursively converts a numpy array (or a `bytes`/`str`/`list`/`tuple` value) into a JSON-serializable nested list/scalar structure. Dispatches to specialized handling first: arrays whose dtype metadata tags them as region references (`dtype.metadata["ref"] is RegionReference`) go through `_regionRefArrayToList`, and opaque (`H5T_OPAQUE`) arrays go through `_opaqueArrayToList`. Otherwise it walks scalar/0-d values down to `bytes` leaves and UTF-8 decodes them (raising `ValueError` on a `UnicodeDecodeError`), passing other leaf values through unchanged. This is the function used to serialize attribute and dataset values to JSON.

## toTuple(rank, data, encoding=None)

Recursively converts a nested list/tuple structure into nested tuples, e.g. `[[1,2],[3,4]] -> ((1,2),(3,4))`. `rank` controls how many levels are converted to tuples (levels below 0 become plain tuples of leaves) vs. left as lists at intermediate depth. If `encoding` is given, leaf values are `.encode()`'d with that codec (surrogate-escaped). Used internally by `jsonToArray` to reshape/re-type nested JSON data before handing it to `numpy.array()`.

## getArraySize(arr)

Returns the size in bytes of a numpy array as `arr.dtype.itemsize` multiplied by the product of `arr.shape`. Note this reflects the *fixed* per-element itemsize and is not accurate for vlen dtypes (see `getByteArraySize` for that case).

## getNumElements(dims)

Returns the total number of elements described by a shape. Accepts either a single `int` (returned as-is) or a `list`/`tuple` of dimension extents (returns their product); raises `ValueError` for any other type.

## jsonToArray(data_shape, data_dtype, data_json)

Converts a JSON value (`data_json`) into a numpy array of shape `data_shape` and dtype `data_dtype`. This is the primary JSON-to-numpy entry point and handles several special cases: `None` input (yields a zero/default array), region-reference-tagged dtypes (delegates to the internal `_regionRefJsonToArray`), opaque dtypes (delegates to `_opaqueJsonToArray`), vlen dtypes including vlen compound fields (built element-by-element via an internal `fillVlenArray` helper that also decodes `bytes` to `str` for string vlen types), and the regular fixed-size case (built via `numpy.array()`, retrying with UTF-8 surrogate-escaped encoding on a `UnicodeEncodeError`, and retrying as a tuple if a plain list conversion produces the wrong element count). It also accounts for `H5T_ARRAY` (subarray) dtypes, whose fixed inner dims get absorbed into the resulting array's shape in addition to `data_shape`. Raises `ValueError` if the resulting array's element count doesn't match the expected count after all fallback strategies are exhausted. Because vlen and compound elements are filled via Python-level loops rather than a single vectorized `numpy.array()` call, this path is not O(1)-call cheap for very large vlen datasets, but it is only used for JSON-sourced data (never bulk `.h5` reads).

## getElementSize(e, dt)

Computes the number of bytes needed to serialize a single element `e` of numpy dtype `dt` as a byte stream. Recurses field-by-field for compound dtypes; for vlen-like dtypes (variable-length strings/sequences, or region-reference-tagged elements — see `_isVlenLike`) it accounts for a 4-byte length prefix plus the payload size, handling `int` (only `0`, meaning an uninitialized/null element), `bytes`, `str`, `np.ndarray`, and `list`/`tuple` element representations; raises `ValueError`/`TypeError` on unexpected values. Used by `getByteArraySize` to size a buffer before serialization.

## getByteArraySize(arr)

Returns the total number of bytes needed to store a numpy array `arr` as a flat byte stream. For fixed-size (non-vlen-like) dtypes this is just `itemsize * size`; for vlen-like dtypes it reshapes to 1-D and sums `getElementSize()` over every element, since each vlen element has a variable, data-dependent byte length. Used to preallocate the buffer for `arrayToBytes`.

## copyBuffer(src, des, offset)

Copies the bytes of `src` into the `des` buffer starting at `offset`, byte-by-byte, and returns the new offset (`offset + len(src)`). A low-level helper used by `copyElement`/`arrayToBytes`; the inline comment notes a vectorized `des[offset:] = src[:]` slice assignment would be an equivalent, faster alternative.

## copyElement(e, dt, buffer, offset)

Serializes one element `e` of dtype `dt` into `buffer` at `offset`, returning the updated offset. Recurses over compound dtype fields; for fixed-size (non-vlen-like) elements it converts via `numpy.asarray(e, dtype=dt).tobytes()`, zero-padding short values (e.g. fixed-length strings) up to `dt.itemsize`. For vlen-like elements it writes a 4-byte little-endian length prefix (as `np.int32`) followed by the payload, supporting `int` (`0` only, for null/uninitialized), `bytes`, `str` (UTF-8 encoded), `np.ndarray` (raw bytes for non-object dtypes, or recursive element-by-element for object arrays), and `list`/`tuple`. Raises `ValueError` if a vlen element's byte length exceeds `MAX_VLEN_ELEMENT` (1,000,000 bytes), and `TypeError`/`ValueError` for unsupported types/values.

## getElementCount(buffer, offset=0)

Reads a 4-byte little-endian `int32` length prefix from `buffer` at `offset` (as written by `copyElement` for vlen elements) and returns it as a Python `int`. Raises `TypeError` if the bytes can't be parsed, `ValueError` if the count is negative, and `ValueError` if it exceeds `MAX_VLEN_ELEMENT` (variable-length elements are expected to be under ~1MB).

## readElement(buffer, offset, arr, index, dt)

Reads a single element of dtype `dt` out of `buffer` starting at `offset`, storing it into `arr` at position `index`, and returns the updated offset. Recurses over compound dtype fields. For fixed-size (non-vlen-like) elements it reads exactly `dt.itemsize` bytes via `numpy.frombuffer`. For vlen-like elements it first reads the length prefix via `getElementCount`, then reads that many bytes as the payload — decoding to `str` for vlen string types, leaving as `bytes` for byte vlen, or as a nested array for vlen-of-numeric-type. This is the inverse of `copyElement`, used by `bytesToArray` to deserialize a raw byte buffer (produced by another backend/version) back into a numpy array of vlen elements.

## encodeData(data, encoding="base64")

Base64-encodes `data` (a `str`, which is first UTF-8 encoded, or `bytes`) and returns the encoded bytes. Only `"base64"` is a supported `encoding` value — anything else raises `ValueError`. Raises `TypeError` if `data` isn't ultimately `str`/`bytes`, and `ValueError` if the string can't be encoded or the base64 encode step fails.

## decodeData(data, encoding="base64")

Inverse of `encodeData`: base64-decodes `data` back to `bytes`. Only `"base64"` is supported (`ValueError` otherwise); raises `ValueError` if decoding fails.

## arrayToBytes(arr, encoding=None)

Serializes a numpy array `arr` to a flat `bytes` object, optionally base64-encoding the result (`encoding` truthy). For vlen-like dtypes it preallocates a `bytearray` sized via `getByteArraySize` and fills it element-by-element with `copyElement`; for fixed-size dtypes it just calls `arr.tobytes()` (a fast, vectorized path). Raises `TypeError` for object-dtype arrays that aren't vlen (no defined byte representation). The vlen path is inherently a per-element Python loop since each element has a data-dependent length; the fixed-size path is the fast, fully-vectorized common case.

## array_for_new_object(data, specified_dtype=None)

Prepares a numpy array suitable for creating a new HDF5 dataset or attribute from arbitrary input `data` (list, scalar, existing array, etc.). Chooses a target dtype (`specified_dtype` if given — with a special-case workaround for `float16` targets to sidestep an h5py conversion bug — otherwise `guess_dtype(data)`), builds the array with `np.asarray(data, order="C", dtype=as_dtype)`, and re-applies the dtype via `.view()` if needed so tagged/metadata dtypes (e.g. h5py string dtypes) survive `asarray`'s no-op behavior when `data` was already an ndarray.

## bytesToArray(data, dt, shape, encoding=None)

Deserializes a raw byte buffer `data` (optionally base64-decoded first, if `encoding` is set) into a numpy array of dtype `dt` and shape `shape`. Fixed-size dtypes are read directly via the fast, vectorized `numpy.frombuffer`; vlen-like dtypes are read element-by-element via `readElement` since each element's length must be parsed from its own length prefix. Also guards against a non-writeable array (a known numpy behavior with `frombuffer`-backed arrays) by copying if needed. This is the inverse of `arrayToBytes`.

## getNumpyValue(value, dt=None, encoding=None)

Converts a single scalar `value` (e.g. from a JSON fill value or scalar attribute) to a numpy scalar of dtype `dt`. If `encoding == "base64"`, `value` must be a string, which is base64-decoded and reconstructed via `bytesToArray`; raises `ValueError` if `encoding` is set but `value` isn't a string, or if the base64 string is malformed. Otherwise, list values are converted to tuples (for compound dtypes), the string `"nan"` is special-cased to `np.nan` for float dtypes, and the value is built via `np.asarray(value, dtype=dt.base)`. Returns a numpy scalar (via `arr[()]`).

## squeezeArray(data)

Removes any 1-extent dimensions from a numpy array `data` via `.squeeze()`, returning the input unchanged if it's already rank ≤ 1. Raises `TypeError` if `data` isn't an `ndarray`. Note: the loop that decides `can_reduce` contains an unconditional `break` on its first iteration, so in practice this always attempts a squeeze for any array with rank > 1 (it does not actually check whether a 1-extent dimension exists before squeezing).

## IndexIterator

Iterator class that walks through every index tuple in a (hyperslab) selection over a dataset shape, without materializing the values — useful for iterating chunk/element coordinates of an arbitrarily large dataset without loading data. Constructed over a dataset `shape` (a tuple of extents) restricted to a selection `sel` (a `slice`, tuple of `slice`s, or `None` for the entire dataspace); raises `ValueError` if `shape` has zero rank, if `sel`'s rank doesn't match `shape`'s rank, or if any per-dimension slice falls outside `[0, shape[dim])` or has `stop <= start`. Iterating yields the next index tuple within the selection, advancing the last dimension first (honoring each slice's `step`, default 1) and carrying over into higher dimensions when a dimension's slice is exhausted (odometer-style), until the entire selection has been enumerated.

## ndarray_compare(arr1, arr2)

Deep-compares two values (numpy arrays, `np.void` compound scalars, or plain Python scalars/bytes/str) for equality, treating an "empty"/zero value in one argument as equal to the corresponding uninitialized numpy representation in the other (e.g. `0`, `b""`, and `""` are all treated as equivalent to an empty/null vlen element), and comparing `str` against `bytes` by UTF-8 encoding/decoding. For vlen-dtype arrays it recurses element-by-element (a per-element Python loop, since vlen elements have no built-in vectorized comparison); for all other array dtypes it delegates to the vectorized `numpy.array_equal`. The function's own docstring/comment flags that the vlen element-by-element path is slow for multi-megabyte vlen arrays and "needs to be optimized."

## getBroadcastShape(mshape, element_count)

Given an array shape `mshape` and a desired `element_count`, returns a numpy-broadcast-compatible shape (a suffix of `mshape`'s dimensions) that contains exactly `element_count` elements, or `None` if `mshape` already matches `element_count` or no compatible broadcast shape can be found. `element_count == 1` always returns `[1]`. Used to support assigning a smaller-than-selection value that numpy can broadcast up to the full selection shape.
