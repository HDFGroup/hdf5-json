# link_util

Helpers for working with h5json link JSON objects representing HDF5 group links — hard links (target
`id`), soft links (in-file `h5path`), and external links (`h5path` plus a `file`/`h5domain` reference to
another file). No user-defined link support is present; link class is either inferred from the JSON
shape or validated against an explicitly given `"class"` value.

## validateLinkName(name)

Verifies `name` is a string containing no `/` character. Raises `ValueError` otherwise (a link name must
be a single path component).

## getLinkClass(link_json)

Determines and returns the link class (`"H5L_TYPE_HARD"`, `"H5L_TYPE_SOFT"`, or `"H5L_TYPE_EXTERNAL"`)
for `link_json`. A hard link is identified by an `"id"` key (validated as a UUID via
`h5json.objid.isValidUuid`); a soft or external link is identified by an `"h5path"` key, external if
`"h5domain"` or `"file"` is also present. If `link_json` already has a `"class"` key, it must agree with
the inferred class. Raises `ValueError` if both `"h5path"` and `"id"` are set, if neither is set, if the
target id is not a valid UUID, or if the explicit `"class"` conflicts with the inferred class.

## getLinkId(link_json)

Returns `link_json["id"]` for a hard link. Raises `TypeError` if `getLinkClass` does not return
`"H5L_TYPE_HARD"`.

## getLinkPath(link_json)

Returns `link_json["h5path"]` for a soft or external link. Raises `TypeError` if the link is not soft or
external.

## getLinkFilePath(link_json)

Returns the external file reference for an external link — `link_json["file"]`, or the deprecated
`link_json["h5domain"]` key for backward compatibility. Raises `TypeError` if the link is not external,
and `KeyError` if neither key is present.

## isEqualLink(link1, link2)

Returns `True` if `link1` and `link2` represent the same link. Both arguments must be dicts with a
`"class"` key. Links of different classes are never equal; hard links compare by target id; soft links
compare by `h5path`; external links compare by both `h5path` and file path. Raises `TypeError` for
non-dict input, a missing `"class"` key, or an unrecognized link class.

## h5Join(path, paths)

Joins `path` with one or more additional path components in `paths` (a single string or a sequence of
strings), inserting `/` separators as needed, and returns the combined path. Returns `path` unchanged if
`paths` is empty/`None`.
