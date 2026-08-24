# shape_util

Helpers for working with h5json dataspace ("shape") descriptors. These functions read and construct
the `shape` JSON objects used by datasets and attributes (`H5S_NULL`, `H5S_SCALAR`, `H5S_SIMPLE`), and
derive properties such as dims, max dims, rank, element count, and extensibility. Most functions accept
either a bare shape JSON dict (`{"class": ..., "dims": ...}`) or a containing dataset/attribute JSON dict
that has a `"shape"` key, and dispatch accordingly.

## getShapeClass(obj_json)

Returns the shape class string (`"H5S_NULL"`, `"H5S_SCALAR"`, or `"H5S_SIMPLE"`) for the given shape.
`obj_json` may be a shape JSON dict itself or a dataset/attribute JSON dict containing a `"shape"` key.
Raises `TypeError` if `obj_json` is not a dict, `ValueError` if no shape can be identified, and
`KeyError` if the resolved shape JSON has no `"class"` key.

## getShapeJson(dims, maxdims=None)

Builds a shape JSON dict from `dims` and an optional `maxdims`. `dims` may be an `int` (treated as a
1-tuple), the string `"H5S_NULL"`, `None` (null shape), or a sequence of non-negative integers
(simple shape). When `maxdims` is given, the shape must be simple, ranks must match, and each extent of
0 or `None` is converted to `"H5S_UNLIMITED"`. Returns a dict with `"class"` and, as applicable, `"dims"`
and `"maxdims"` keys. Raises `TypeError`/`ValueError` on malformed dims/maxdims.

## getShapeDims(shape)

Extracts the dims tuple from `shape`, which may be an `int`, `list`, `tuple`, the string `"H5S_NULL"`,
or a shape/dataset JSON dict. Returns `None` for a null dataspace, `()` for scalar, or a tuple of extents
for a simple dataspace. Raises `ValueError` for unrecognized string or dict input.

## getNumElements(obj_json)

Returns the number of elements implied by the shape: `0` for a null shape, `1` for scalar, or the
product of the dims otherwise (computed via `numpy.prod`). Internally calls `getShapeDims`.

## getRank(shape)

Returns the rank (number of dimensions) of `shape`: `0` for a null shape, otherwise `len(dims)`.

## isNullSpace(shape)

Returns `True` if the shape class is `"H5S_NULL"`, `False` otherwise.

## isScalar(shape)

Returns `True` if the shape class is `"H5S_SCALAR"`, `False` otherwise.

## getDataSize(shape, type_size: int = 1)

Returns the byte size of the dataspace given a per-element `type_size`. Returns `0` for a null shape,
`type_size` for a scalar shape, and `type_size * prod(dims)` for a simple shape. Any unlimited dimension
is effectively treated as extent 1 (via whatever value is currently in `dims`), so the returned size is
a lower bound, not the maximum possible size.

## isExtensible(obj_json)

Returns `True` if the dataset/shape is extensible: shape class must be `"H5S_SIMPLE"` and a `"maxdims"`
key must be present. Raises `ValueError` if `maxdims` rank doesn't match `dims` rank. `obj_json` may be
a dataset/attribute JSON dict (with `"shape"`) or a bare shape JSON dict.

## getMaxDims(obj_json)

Returns the max dims tuple for a shape. Returns `None` for a null shape, `()` for scalar, and for a
simple shape returns `"maxdims"` if present, otherwise falls back to `"dims"`. Raises `TypeError` if
`obj_json` is not a dict, and `KeyError`/`ValueError` if the shape JSON is malformed.
