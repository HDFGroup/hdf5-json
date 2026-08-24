# hdf5dtype

This module implements the bidirectional mapping between HDF5 datatypes and their two other representations used throughout `h5json`: numpy `dtype` objects (used internally and by the h5py storage backend) and h5json's JSON type descriptors (`{"class": "H5T_...", ...}` dictionaries, used by the JSON storage backend and the schema). It covers all HDF5 datatype classes handled by the library — integer, float, string (fixed and variable length), compound, array (`H5T_ARRAY`, mapped to numpy subarray dtypes), enum (including the h5py boolean-as-enum convention), opaque (`H5T_OPAQUE`, mapped to numpy void `Vnnn` dtypes), variable-length (`H5T_VLEN`), and object/region references (`H5T_REFERENCE`). It also defines `Reference` and `RegionReference`, the Python objects stored in reference-typed numpy arrays, along with h5py-compatible helpers (`special_dtype`, `check_dtype`) for building and introspecting the "special" object dtypes used to carry vlen, enum, and reference metadata on top of plain numpy dtypes.

## Reference

Represents an HDF5 object reference (`H5T_STD_REF_OBJ`): the id of a referenced group, dataset, or committed datatype, stored as a fixed-size (`S48`) special dtype value. The id is normalized through `getHashTagForId` on construction and accepted either bare or prefixed with `groups/`, `datasets/`, or `datatypes/`.

### Reference.id

Property returning the object's low-level identifier (the hashtag-style id string, or `None` if unbound).

### Reference.tolist()

Returns the reference wrapped as a one-element list of strings suitable for JSON/array serialization: `[("groups/<id>",)]`, `[("datasets/<id>",)]`, or `[("datatypes/<id>",)]` depending on the id's type-code prefix character (`g`, `d`, or `t`). For a null reference (empty `_id`), returns `[("",)]`. Raises `TypeError` if `_id` is not a string or has an unrecognized type code.

## RegionReference

Represents an HDF5 region reference (`H5T_STD_REF_DSETREG`): the id of a referenced dataset plus a binary-serialized `selections.Selection` describing the selected region. Unlike `Reference`, it is a variable-length (`O`) special dtype value, since its serialized size depends on the bound selection rather than being fixed. Binding is optional at construction and can be deferred to `bind()`.

### RegionReference.id

Property returning the low-level identifier of the referenced dataset, or `None` if unbound.

### RegionReference.selection_bytes

Property returning the serialized selection (the output of `selections.Selection.tobytes()`), or `None` if no selection is bound (meaning the whole dataset, or a reference recovered from HDF5 with no selection information).

### RegionReference.bind(objid, selection=None)

Binds the reference to a dataset id and (optionally) a selection. `objid` may be a uuid string/bytes (optionally prefixed with `datasets/`) or any object exposing an `_id` attribute (e.g. a dataset object); it is normalized via `getHashTagForId`, and a non-`datasets/`-prefixed path raises `TypeError`. `selection` may be a `selections.Selection` instance (serialized via `.tobytes()`), raw bytes/bytearray already in serialized form, or `None` (no selection, i.e. whole dataset). Returns `self` for chaining.

### RegionReference.tobytes()

Serializes the dataset id and selection bytes into a single flat `bytes` blob (magic + version + id length/bytes + selection length/bytes + a fixed non-zero trailer byte), suitable for storing as a raw `H5T_REFERENCE` dataset or attribute value. The trailer byte guards against numpy's fixed-length `"S<n>"` dtype silently stripping trailing NUL bytes.

### RegionReference.frombytes(data)

Class method that reconstructs a `RegionReference` from a blob produced by `tobytes()`. Validates the magic bytes and serialization version, raising `ValueError` if either is invalid/unsupported.

### RegionReference.to_json()

Converts the reference to the h5json JSON representation: `{"id": <uuid>, "select_type": ..., "selection": [...]}` for a point or hyperslab selection, or just `{"id": <uuid>}` if no selection is bound. Raises `ValueError` if the reference itself is null (`id` is `None`). Selections that have no direct HDF5 equivalent (`H5S_SEL_FANCY` or stepped hyperslabs — artifacts of this project's own `Selection` model) are instead embedded under a `"selection_dict"` key via `Selection.to_dict()`.

### RegionReference.from_json(d)

Class method reconstructing a `RegionReference` from the JSON produced by `to_json()`, or from `None` (yielding an unbound reference). Requires an `"id"` key (raises `KeyError` otherwise); dispatches to `from_dict()` if `"selection_dict"` is present, to `from_region_json()` if `"select_type"` is present, otherwise binds with no selection.

## special_dtype(**kwds)

Creates an h5py-compatible "special" numpy dtype carrying extra metadata. Exactly one keyword must be given: `vlen=basetype` (Python `str`/`bytes` or an `np.dtype`) produces an `"O"` dtype tagged with `metadata={"vlen": basetype}`; `enum=(basetype, values_dict)` produces an integer dtype tagged with `metadata={"enum": values_dict}` (raises `TypeError` if the base type isn't integer); `ref=Reference` produces a fixed `"S48"` dtype tagged `metadata={"ref": Reference}`, and `ref=RegionReference` produces an `"O"` dtype tagged `metadata={"ref": RegionReference}`. Raises `TypeError` for any other keyword or an unsupported `ref` value.

## find_item_type(data)

Finds the common item type of a (possibly nested) Python list/tuple, or of a numpy object array whose dtype isn't itself a vlen special dtype. Returns `None` if the items are not all the same type (or if the input is a non-object numpy array or something other than a list/tuple/array), otherwise returns that common `type`. Used to decide whether a plain Python collection should be treated as strings/bytes for dtype-guessing purposes.

## guess_dtype(data)

Attempts to guess an h5py-style special dtype for `data` based on its item type: returns `special_dtype(vlen=bytes)` if all items are `bytes`, `special_dtype(vlen=str)` if all items are `str`, otherwise `None` (leaving dtype inference to numpy's array constructor). Does not currently handle `Reference`/`RegionReference` guessing (noted as a TODO in the code).

## is_float16_dtype(dt)

Returns `True` if `dt` (normalized via `np.dtype(dt)`) is a floating-point kind with `itemsize == 2` (i.e. `float16`); returns `False` for `dt is None` or any other dtype.

## check_dtype(**kwds)

Inspects a dtype for h5py special-type metadata attached by `special_dtype()`. Exactly one keyword must be given: `vlen=dtype` returns the vlen base class/dtype or `None`; `enum=dtype` returns the enum's name-to-value mapping dict or `None`; `ref=dtype` returns `Reference` or `RegionReference` or `None`. Raises `TypeError` for any other keyword. Returns `None` (rather than raising) if `dtype.metadata` is absent or lacks the requested key.

## getTypeResponse(typeItem)

Converts a full JSON type item into an abbreviated response form: committed (shared) types are reduced to their `"datatypes/<uuid>"` reference; `H5T_INTEGER`/`H5T_FLOAT` types are reduced to just `class`/`base`; `H5T_OPAQUE` to `class`/`size`; `H5T_REFERENCE` to `class`/`base`; `H5T_COMPOUND` recurses into each field's type via the same reduction. All other classes are passed through as-is except `size`/`base_size` keys are dropped and a dict-valued `base` (e.g. for arrays) is recursively reduced too.

## getTypeItem(dt, metadata=None)

Converts a numpy `dtype` (or dtype-like value) into an h5json JSON type descriptor dict. Handles: compound types (recurses per-field, `H5T_COMPOUND`); array/subarray types (`H5T_ARRAY`, recurses on `dt.base`, raises `TypeError` if the base type equals the parent); object (`"O"`) kind, which is disambiguated via `vlenBaseType()`/`check_dtype()` into vlen ASCII/UTF-8 strings, vlen data (`H5T_VLEN`), or object/region references (`H5T_REFERENCE`), raising `TypeError` for anything unrecognized; void (`"V"`) kind, mapped to `H5T_OPAQUE` (only `size` is recorded — no HDF5 "tag" equivalent exists in numpy); fixed-length byte (`"S"`) and unicode (`"U"`) kinds, mapped to fixed-length `H5T_STRING` (or `H5T_REFERENCE` if the `"S"` dtype carries reference metadata); boolean kind, mapped to the h5py boolean-as-`H5T_ENUM` convention over an `H5T_STD_I8` base; float kind, mapped to `H5T_FLOAT` (raises `TypeError` for non-predefined float types); and integer kind, mapped to `H5T_ENUM` if `enum` metadata is present, otherwise `H5T_INTEGER` (raises `TypeError` for non-predefined integer types). `metadata` defaults to `dt.metadata` when not explicitly supplied. Raises `TypeError` for any unrecognized dtype kind.

## isVlen(dt)

Recursively checks whether `dt` (a compound or primitive dtype) contains any variable-length element, by looking for `"vlen"` in `dt.base.metadata` (or, for compound types, in any field).

## vlenBaseType(dt)

Returns the base type of a vlen dtype — `bytes`, `str`, or an `np.dtype` — or `None` if `dt` is not a vlen special dtype. Raises `TypeError` if called on a compound dtype (`len(dt) != 0`).

## isOpaqueDtype(dt)

Returns `True` if `dt` is a plain numpy void (`"V"`) dtype with no subfields/shape, or if its metadata has a truthy `h5py_opaque` key; otherwise `False`.

## getItemSize(typeItem)

Computes the per-item size in bytes of an h5json type descriptor, or the string `"H5T_VARIABLE"` for variable-length types. Accepts either a predefined type name string (e.g. `"H5T_STD_I32LE"`, parsed to `32 // 8` bytes) or a full type dict. Handles all type classes: `H5T_INTEGER`/`H5T_FLOAT`/`H5T_ENUM`/`H5T_ARRAY` recurse into `base`/`base` (multiplying by the product of `dims` for arrays); `H5T_STRING` uses `length` directly; `H5T_VLEN` and opaque region references (`H5T_REFERENCE` with `base == H5T_STD_REF_DSETREG`) return `"H5T_VARIABLE"`; `H5T_OPAQUE` uses `size`; `H5T_REFERENCE` with `H5T_STD_REF_OBJ` returns a fixed `48`, otherwise a fallback guess of `80`; `H5T_COMPOUND` sums each field's size (short-circuiting to `"H5T_VARIABLE"` if any field is variable). Raises `KeyError`/`TypeError` for malformed input.

## getDtypeItemSize(dtype)

Computes the per-item size in bytes of a numpy `dtype` directly (as opposed to a JSON type item, which `getItemSize` handles), or `"H5T_VARIABLE"` if the dtype or any of its compound fields/subarray base is a vlen special dtype. Recurses through compound fields and subarray shapes (multiplying by `np.prod(dtype.shape)`).

## getNumpyTypename(hdf5TypeName, typeClass=None)

Maps a predefined HDF5 type name string (e.g. `"H5T_STD_I32LE"`, `"H5T_IEEE_F64BE"`) to a numpy dtype-name string (e.g. `"<i4"`, `">f8"`), applying `LE`/`BE` suffixes to little/big-endian numpy byte-order prefixes. `typeClass`, if given, restricts matching to `"H5T_INTEGER"` or `"H5T_FLOAT"`. Raises `Exception` if the name is too short, or `TypeError` if it doesn't match a known predefined type (optionally constrained by `typeClass`).

## createBaseDataType(typeItem)

Creates a numpy dtype from a non-compound h5json type descriptor (or a predefined type name string, handled directly). Supports `H5T_INTEGER`, `H5T_FLOAT` (via `getNumpyTypename`), `H5T_STRING` (fixed-length `"S"`-coded, or variable-length via `special_dtype(vlen=...)` — array dims not supported for variable-length strings), `H5T_VLEN` (via `special_dtype(vlen=...)` over a recursively-created base type), `H5T_OPAQUE` (`"V<size>"`, size must be positive), `H5T_ARRAY` (recurses via `createDataType` on the base type — restricted to integer/float/string/compound/array base classes — and wraps it as `np.dtype((baseType, dims))`), `H5T_REFERENCE` (`special_dtype(ref=Reference)` or `special_dtype(ref=RegionReference)` depending on `base`), and `H5T_ENUM` (accepts either a `"mapping"` or legacy `"members"` key; collapses the h5py 2-value `{TRUE, FALSE}` int8 enum convention back to a numpy `bool` dtype, otherwise uses `special_dtype(enum=...)`). Raises `KeyError`/`TypeError` for missing fields or unsupported combinations (e.g. array dims combined with vlen or opaque types).

## createDataType(typeItem)

Creates a numpy dtype from any h5json type descriptor, including compound types. For `H5T_COMPOUND`, validates and recursively converts each field (name must be a string and ASCII-encodable) into a list of `(name, dtype)` tuples and builds a structured `np.dtype`; all other classes delegate to `createBaseDataType`. Accepts predefined type name strings/bytes directly. Raises `KeyError`/`TypeError` for malformed field lists or names.

## validateTypeItem(typeItem)

Validates a JSON type descriptor by calling `createDataType` on it and discarding the result; a `KeyError`, `TypeError`, or `ValueError` propagating out indicates an invalid type. Returns nothing on success.

## getBaseTypeJson(type_name)

Converts a predefined HDF5 type name string (must start with `H5T_` and end in `LE`/`BE`) into a minimal JSON type dict with `class` (`H5T_INTEGER` or `H5T_FLOAT`) and `base` set to the name itself. Raises `TypeError` if the name isn't well-formed or doesn't match a known predefined integer/float type.

## getSubType(dt_parent, fields)

Builds a compound numpy dtype containing only the named fields from `dt_parent`, preserving each field's original sub-dtype. `fields` may be a single field name (string) or an iterable of names. Raises `TypeError` if `dt_parent` is not a compound type, if `fields` is empty/`None`, or if a requested field name is not present in `dt_parent`.
