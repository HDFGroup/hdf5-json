# dset_util

Helper functions for dataset-level metadata operations in the h5json object model: reading and validating a dataset's storage layout (chunked/contiguous/compact, or a reference into an existing traditional HDF5 file), computing/guessing chunk shapes and sizes, estimating a dataset's on-disk size, resizing extensible datasets, and validating/generating a dataset creation property list (fill value, layout, filters). These functions operate purely on the JSON representations of shape/type/layout (as produced by `hdf5dtype`/`shape_util`) — none of them read or write actual dataset element data.

## getDatasetLayout(dset_json)

Returns the `layout` sub-object from a dataset's JSON representation `dset_json`. Prefers `dset_json["creationProperties"]["layout"]` (the current location), falling back to a top-level `dset_json["layout"]` key for compatibility with older HSDS-produced JSON. Returns `None` if neither is present.

## getDatasetLayoutClass(dset_json)

Returns the `"class"` field of the dataset's layout (one of the `LAYOUT_CLASSES` values, e.g. `"H5D_CHUNKED"`), or `None` if there's no layout or no `"class"` key. Built on top of `getDatasetLayout`.

## estimateDatasetSize(shape_json, item_size, chunk_min=CHUNK_MIN)

Estimates a dataset's size in bytes from its shape and element size. Returns `0` for `H5S_NULL` shapes and `item_size` for `H5S_SCALAR` shapes. For a fixed-size simple dataspace (no `"maxdims"`), returns `item_size * getNumElements(shape_json)`. For an extensible dataspace, sizes it using only the *bounded* dimensions (extents that aren't `0`/`"H5S_UNLIMITED"` in `maxdims`), then — if the result is under `chunk_min` — rounds it up to just over `chunk_min` (rounded to a multiple of `item_size`), since an unlimited dimension can't be sized exactly and the caller needs a reasonable non-trivial guess.

## resize_dataset(dset_json, shape)

Validates and applies a new shape `shape` to a chunked, extensible dataset's JSON in-place, mutating `dset_json["shape"]["dims"]`. Raises `TypeError` if the dataset isn't `H5D_CHUNKED` or its shape class isn't `H5S_SIMPLE`, and `ValueError` if `shape`'s rank doesn't match the current rank, the dataset isn't extensible (`isExtensible`), any new extent is negative, or any new extent exceeds the corresponding `maxdims` bound (bounds of `0`/`"H5S_UNLIMITED"` allow any extent). Returns `None` immediately (without mutating anything) if `shape` equals the current dims — this is purely a metadata update; it does not touch or resize any underlying stored chunk data.

## getContiguousLayout(shape_json, item_size, chunk_min=None, chunk_max=None)

Computes a chunk-shaped layout tuple for a dataset stored contiguously (used as an internal I/O granularity even for non-chunked storage). Requires `item_size` to be a fixed (`int`) size — raises `ValueError` for variable-length types — and requires both `chunk_min`/`chunk_max` to be supplied. Returns `None` for `H5S_NULL` shapes, `(1,)` for `H5S_SCALAR` shapes, and the shape's `dims` directly if any dimension extent is `0` (empty dataset). Otherwise it starts from the last (fastest-varying) dimension and greedily includes full extents into the layout while the accumulated byte size stays under `chunk_max`, halving (rounding up) the extent of the first dimension that would overflow and leaving all lower-indexed dimensions as `1`. Raises `ValueError` if `chunk_max < chunk_min`, or if any dim is negative, or if rank is 0.

## getChunkSize(chunk_dims, type_size: int = 1)

Returns the number of elements (or bytes, if `type_size` is the item size) that a chunk of shape `chunk_dims` holds — the product of `chunk_dims` times `type_size`. Raises `ValueError` if any dimension in `chunk_dims` is `<= 0`.

## getChunkDims(dset_json)

Returns the chunk shape (as a tuple) for a dataset's JSON representation. Returns `None` for `H5S_NULL`, `(1,)` for `H5S_SCALAR`. For any non-chunked layout class (or no layout class set), returns the dataset's full shape dims (i.e., the entire dataspace is treated as one chunk). For a chunked layout, reads `layout["dims"]` from `getDatasetLayout`, raising `KeyError` if that key is missing.

## validateLayout(shape_json, type_json, layout)

Validates a proposed `layout` dict against a dataset's shape and type, raising `ValueError` (or, indirectly, letting `KeyError` propagate) on any inconsistency. If `layout` specifies `"dims"` (a chunk shape), checks its rank matches the dataspace rank, each extent is a positive integer, and each extent doesn't exceed the corresponding dataspace/`maxdims` bound (unbounded dimensions allow any positive chunk extent). Then dispatches on `layout["class"]` (required; missing raises `ValueError`): `H5D_CONTIGUOUS_REF`, `H5D_CHUNKED_REF`, and `H5D_CHUNKED_REF_INDIRECT` (references into an existing traditional HDF5 file) each require their own specific keys (`file_uri`/`offset`/`size`, `file_uri`/`dims`/`chunks`, or `dims`/`chunk_table` respectively) and disallow variable-length item sizes; `H5D_CHUNKED` requires `"dims"` and a `H5S_SIMPLE` shape class; `H5D_CONTIGUOUS` and `H5D_COMPACT` disallow `"dims"`/`"maxdims"`. Any other `layout["class"]` value raises `ValueError`.

## validateDatasetCreationProps(creation_props, type_json=None, shape=None)

Validates a dataset creation property list dict `creation_props` against a required `type_json`/`shape` (raises `ValueError` if either is missing/falsy). If `"fillValue"` is present, validates it against the dataset's dtype: if `"fillValue_encoding"` is set, it must be `"None"` or `"base64"` and the fill value must then be a string; otherwise the fill value is checked by attempting `getNumpyValue` (from `array_util`), raising `ValueError` on failure. If `"layout"` is present, delegates to `validateLayout`. If `"filters"` is present, delegates to `filters.validateFilters` (wrapping `KeyError`/`TypeError`/`ValueError` into a `ValueError`), and additionally requires the layout class to start with `"H5D_CHUNKED"` if a layout class was set — filters are only valid with chunked storage.

## expandChunk(layout, typesize, shape_json, chunk_min=CHUNK_MIN)

Grows a chunk shape `layout` (list/tuple of extents) so its byte size exceeds `chunk_min`, given per-element `typesize` and the dataset's `shape_json`. Returns `None`/`(1,)` immediately for `H5S_NULL`/`H5S_SCALAR` shapes. If the whole dataset already fits under `chunk_min` and has no extendable dimensions, returns the full dataset shape as a single chunk. Otherwise, iteratively doubles extents — preferring extendable dimensions first (unlimited dimensions can double indefinitely; bounded-but-extendable dimensions double up to their `maxdims` cap, then are excluded from further extension), falling back to non-extendable dimensions bounded by the current dataspace extent — stopping once the chunk's byte size exceeds `chunk_min` or no further growth is possible in a pass (to avoid an infinite loop).

## shrinkChunk(layout, typesize, chunk_max=CHUNK_MAX)

Shrinks a chunk shape `layout` so its byte size (given `typesize`) is at most `chunk_max`, by repeatedly halving (round-up) whichever dimensions are still `> 1`, one pass over all dimensions at a time, stopping when the size target is met or no dimension can shrink further (guarding against an infinite loop if no progress is made in a pass).

## guessChunk(shape, typesize, chunk_min=None, chunk_max=None)

Heuristically picks a chunk shape for a dataset given its `shape` (a plain shape tuple/list, or a shape JSON dict) and `typesize` (element byte size; `"H5T_VARIABLE"` is treated as an assumed 128 bytes). Returns `None`/`(1,)` for null/scalar shapes. Unbounded (`0`/`"H5S_UNLIMITED"`) dimensions are stood in for with a placeholder extent of 1024 for sizing purposes. Computes the naive chunk size (full shape as one chunk) and calls `expandChunk`/`shrinkChunk` as needed to bring it within `[chunk_min, chunk_max]`. The docstring notes this behavior is "undocumented and subject to change without warning."

## generateLayout(shape_json, type_json, chunks=None, chunk_min=CHUNK_MIN, chunk_max=CHUNK_MAX, max_chunks_per_folder=0)

Builds a full layout dict (suitable for a creation property list) for a new dataset given its shape and type. Returns `{}` for `H5S_NULL` shapes (raising `ValueError` if `chunks` was requested, since null-space datasets can't be chunked) and `{"class": "H5D_CONTIGUOUS"}` for `H5S_SCALAR` shapes (same chunk restriction). Raises `ValueError` if `chunk_min > chunk_max`. Otherwise estimates the dataset's size (`estimateDatasetSize`); if it's small and neither extensible nor explicitly chunked, returns a plain `H5D_CONTIGUOUS` layout. Otherwise builds an `H5D_CHUNKED` layout: uses caller-supplied `chunks` dims if given (validated against `maxdims`), otherwise calls `guessChunk`. If `max_chunks_per_folder > 0`, also computes and sets a `"partition_count"` (estimating unbounded dimensions' eventual extent via a fixed guess of 10^6 total elements spread evenly across unlimited dimensions) when the projected number of chunks would exceed that per-folder limit. Finally re-validates the constructed layout via `validateLayout` before returning it.

## generate_dcpl(shape_json, type_json, chunks=None, filters=[], chunk_min=CHUNK_MIN, chunk_max=CHUNK_MAX, max_chunks_per_folder=None, initializer=None, initializer_opts=None)

Generates a full dataset creation property list (`plist` dict) for a new dataset. For non-`H5S_SIMPLE` shape classes, raises `TypeError` if `chunks`/`filters` were requested (unsupported for scalar/null datasets) and otherwise returns an empty plist. Validates `filters` via `filters.validateFilters`, then sets `plist["layout"]` via `generateLayout` and `plist["filters"]` if any filters were given. If `initializer` is provided, packages it (plus any `initializer_opts`) into `plist["initializer"]` — the module comment notes this option still needs to be documented in the JSON spec. Note: the default `filters=[]` is a mutable default argument, though the function only reads from it (`validateFilters(filters)`, `len(filters)`) and never mutates it in place, so this is not currently a bug in practice.

## getFillValue(obj_json)

Returns the fill value from an object's JSON (`obj_json["creationProperties"]["fillValue"]`, or directly from `obj_json` if no `"creationProperties"` key is present, treating `obj_json` itself as the creation property list), or `None` if not set. Note: this function checks for the misspelled key `"filLValue"` rather than `"fillValue"`, so as written it will always return `None` even when a correctly-named `"fillValue"` key is present — likely a bug worth flagging rather than documenting as intended behavior.
