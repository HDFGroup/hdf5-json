# filters

Handling of HDF5 compression/transformation filters (gzip/deflate, szip, shuffle, fletcher32,
scale-offset, nbit, and third-party filters like lzf, blosc, snappy, lz4, zstd). This module maps
between the h5json JSON filter descriptor format (`{"class": ..., "id": ..., "name": ..., ...options}`)
and the filter registry defined by `FILTER_DEFS`, validating filter options and extracting filter lists
from a dataset's `creationProperties`.

## Module data

- `FILTER_DEFS` — tuple of `(class_key, filter_id, name, option_names)` for all recognized filters
  (including `H5Z_FILTER_NONE`, `H5Z_FILTER_DEFLATE`, `H5Z_FILTER_SHUFFLE`, `H5Z_FILTER_FLETCHER32`,
  `H5Z_FILTER_SZIP`, `H5Z_FILTER_NBIT`, `H5Z_FILTER_SCALEOFFSET`, and various third-party filters).
- `HDF_FILTER_OPTION_ENUMS` — maps h5py enum option values (e.g. szip `coding`, scale-offset
  `scaleType`) to their HDF5 constant names.
- `COMPRESSION_FILTER_IDS` / `COMPRESSION_FILTER_NAMES` — the subset of filter classes/names considered
  compression filters.
- `DEFAULT_GZIP`, `DEFAULT_SZIP`, `DEFAULT_LZ4`, `SO_INT_MINBITS_DEFAULT` — default option values.

## getAllFilterNames()

Returns a sorted tuple of all recognized filter names (excluding the id-0 "none" filter and any entry
with an empty name), derived from `FILTER_DEFS`.

## getFilterItem(name, options={})

Resolves a filter identifier (a filter class key, alias like `"deflate"`/`"zlib"`, a numeric filter id,
or an existing filter JSON dict) to a full filter JSON dict, validating and filling in `options` along
the way. `name="deflate"`/`"zlib"` is normalized to `"gzip"`. Numeric ids greater than 32000 not found in
`FILTER_DEFS` are treated as a generic `H5Z_FILTER_USER` filter. Each option is checked against the
option set registered for that filter class (raising `KeyError` for an unsupported option) and against
type/range rules specific to the filter (e.g. deflate `level` must be an int 0-9; szip `coding` must be
a recognized enum value; scale-offset `scaleOffset` must be a non-negative int). Missing deflate `level`
defaults to `DEFAULT_GZIP`. Raises `KeyError` if the filter is unrecognized.

## validateFilter(filter_json)

Validates that `filter_json` is a well-formed filter descriptor: must be a dict with `"class"`, `"id"`,
and `"name"` keys; the `"id"` must match the id registered for `"class"` in `FILTER_DEFS` (or, for
`H5Z_FILTER_USER`, must be greater than 32000); and any extra keys are validated as options via
`getFilterItem`. Raises `TypeError`, `KeyError`, or `ValueError` on any mismatch.

## validateFilters(filters, supported_filters=None)

Validates every filter JSON dict in the `filters` list by calling `validateFilter` on each. If
`supported_filters` is given, also raises `ValueError` when a filter's class is not among the supported
set.

## getFilters(dset_json)

Returns the list of filter JSON dicts from `dset_json["creationProperties"]["filters"]`, or an empty
list if either key is absent.

## isCompressionFilter(filter)

Returns `True` if the resolved filter's class (via `getFilterItem`) is one of `COMPRESSION_FILTER_IDS`.

## getCompressionFilter(filters)

Returns the first filter JSON dict in `filters` whose `"class"` is in `COMPRESSION_FILTER_IDS`, or
`None` if none match.

## getShuffleFilter(filters)

Returns the shuffle filter JSON dict (`"class" == "H5Z_FILTER_SHUFFLE"`) from `filters`, or `None` if
not present.
