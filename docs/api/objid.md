# objid

UUID generation and classification for HDF5 object ids, plus mapping between object ids and their S3
storage keys. Two id "schemas" are supported: Schema 1 ids are bare, standard-format UUIDs (`8-4-4-4-12`
hex groups), optionally combined with a one-character collection prefix (`g-`/`d-`/`t-`/`c-` for
groups/datasets/datatypes/chunks) and/or an md5 hash-tag prefix used for S3 key distribution. Schema 2
ids use a non-standard `8-8-4-6-6` grouping and encode extra structure — root group ids are recognizable
because their last 16 hex chars are a fixed rotation of their first 16, and non-root ids embed their
root id's hex prefix so children can be traced back to the domain's root group. The module also handles
domain ids (path-like strings identifying an HDF5 file/domain rather than an in-file object) and chunk
ids (which append a `_i_j_k` chunk coordinate suffix).

## getIdHash(id)

Returns the first 5 hex characters of the MD5 digest of `id` (encoded as UTF-8). Used to build the
distribution-hash prefix for Schema 1 S3 keys.

## isSchema2Id(id)

Returns `True` if `id` is a valid UUID (per `isValidUuid`) formatted as a Schema 2 id — distinguished by
its third hyphen-separated segment being 8 hex characters long (vs. 4 for Schema 1). Returns `False` if
`id` isn't a valid UUID at all. Raises `ValueError` if a valid UUID doesn't split into exactly 6 parts.

## getIdHexChars(id)

Returns the concatenated hex characters of `id` (the 5 UUID segments after the leading collection-prefix
segment), with hyphens removed. For chunk ids (leading `c`), the chunk-coordinate suffix (after `_`) is
stripped first. Raises `ValueError` if the id doesn't split into 6 parts.

## hexRot(ch)

Returns a single hex character (`0`-`f`) equal to `(int(ch, 16) + 8) % 16`, formatted as lowercase hex.
Used to derive/verify the rotated-prefix relationship between a Schema 2 root id and its non-root
descendants.

## getCollectionForId(obj_id)

Returns `"groups"`, `"datasets"`, or `"datatypes"` based on `obj_id`'s prefix (`g-`/`groups/`,
`d-`/`datasets/`, or `t-`/`datatypes`). Raises `ValueError` if `obj_id` is not a string or has no
recognized collection prefix.

## getHashTagForId(id)

Returns the canonical `<collection_char>-<uuid>` form of `id`. If `id` contains a `/` (e.g.
`"collection/uuid"`), splits off the last path segment as the tag and, if it lacks a one-character
collection prefix, derives one from the collection name. Raises `ValueError` if the tag portion is
shorter than a UUID or the input has an unexpected number of path segments.

## isRootObjId(id)

Returns `True` if `id` is a Schema 2 root group id — determined by checking that the id is a group id
whose last 16 hex characters equal `hexRot` applied to its first 16 hex characters. Requires `id` to
already be a Schema 2 id (raises `ValueError` otherwise, via `isSchema2Id`) and a valid UUID (raises via
`validateUuid`). Returns `False` (rather than raising) if the id is not a group id.

## getRootObjId(id)

Returns the root group id that `id` belongs to (Schema 2 only). If `id` is already a root id, returns it
unchanged. Otherwise takes the id's hex token, overwrites its last 16 characters with the `hexRot` of its
first 16, and reassembles it as a `g-`-prefixed Schema 2 id.

## createObjId(obj_type=None, root_id=None)

Creates a new object id. If `obj_type` is `None`, returns a plain `uuid.uuid4()` string (Schema 1 style).
Otherwise `obj_type` must be one of `"groups"`, `"datasets"`, `"datatypes"`, `"chunks"`, and a Schema 2 id
is built from a SHA-256 hash of a random UUID salt. If `root_id` is given, the new id's first 16 hex
characters are replaced with the root id's first 16 hex characters (linking the new id to that root); if
`root_id` is omitted, `obj_type` must be `"groups"` and the new id is itself constructed as a root id
(last 16 chars set to the `hexRot` of the first 16). Returns the id string, prefixed appropriately (e.g.
`g-`, `d-`, `t-`).

## getS3Key(id)

Returns the S3 storage key for `id`. Domain ids (containing `/`) map to a `<path>.domain.json` key
(directory-suffixed if not already ending in a filename). Schema 2 object ids map to
`db/<root_prefix>/<collection_char>/<suffix>` plus a type-specific filename suffix (`.group.json`,
`.dataset.json`, `.datatype.json`), with chunk ids using collection `"d"` and appending an optional
partition segment and the chunk coordinate. Schema 1 ids map to `<md5hash>-<id>`. Raises `ValueError` for
unrecognized prefixes.

## getObjId(s3key)

Inverse of `getS3Key`: reconstructs the object id from an S3 key string. Handles Schema 1 keys (5
alphanumeric hash chars + `-` + prefix char), domain keys (`.../.domain.json`), and Schema 2 keys
(`db/...` with 3, 5, or 6 `/`-separated segments, covering root, group/dataset/datatype/chunk, and
partitioned-chunk forms respectively). Raises `ValueError` for any key that doesn't match a recognized
pattern.

## isS3ObjKey(s3key)

Returns `True` if `getObjId(s3key)` succeeds and yields a non-empty id, `False` if it raises `KeyError`
or `ValueError`.

## validateUuid(id, obj_class=None)

Validates that `id` is a well-formed object id, raising `ValueError` on any problem. Accepts a bare
36-character UUID (only if `obj_class` is not given), or a prefixed/collection-pathed id which is
normalized (stripping any Schema 1 hash tag, expanding a `"collection/uuid"` form, and re-checking the
one-character prefix). If `obj_class` (`"groups"`/`"datasets"`/`"datatypes"`) is given, checks that the
id's prefix matches. Chunk ids have their coordinate suffix stripped before the final length/character
checks. Every remaining character must be alphanumeric or `-`.

## isValidUuid(id, obj_class=None)

Returns `True` if `validateUuid(id, obj_class)` succeeds, `False` if it raises `ValueError`.

## isValidChunkId(id)

Returns `True` if `id` is a valid uuid (per `isValidUuid`) and its prefix character is `"c"`.

## getClassForObjId(id)

Returns `"domains"` if `id` starts with `/`, `"chunks"` if it's a valid chunk id, otherwise delegates to
`getCollectionForId` (`"groups"`/`"datasets"`/`"datatypes"`). Raises `ValueError` for a non-string or
empty `id`.

## isObjId(id)

Returns `True` if `id` is a non-empty string that is either a domain-style path (contains `/` not at
position 0) or a valid UUID per `isValidUuid`. Returns `False` for non-strings or empty strings.

## getUuidFromId(id)

Strips any collection path prefix (`"collection/uuid"`) and any one-character type prefix (`g-`/`d-`/
`t-`) from `id`, returning the bare 36-character UUID. Raises `ValueError` if the resulting length isn't
UUID length (with or without a 2-character prefix) or if the path form has more than 2 segments.
