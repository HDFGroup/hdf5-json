# hdf5db

This module defines `Hdf5db`, the in-memory object store that sits between the h5json library's storage backends (`storage_plugin.StoragePlugin` implementations) and everything else that manipulates HDF5 metadata and data. `Hdf5db` keeps a dictionary of object JSON (`_db`, keyed by object id) plus sets tracking which objects are new, dirty, resized, or deleted since the last flush, and a per-dataset list of pending, not-yet-flushed value updates. Reads (`getObjectById`, `getDatasetValues`, etc.) transparently overlay these in-memory changes on top of whatever the plugin last supplied, and `flush()`/`close()` push everything through that same plugin. The module also defines the `ChunkIterator` helper class used to walk a dataset's values chunk by chunk without loading the whole dataset into memory.

## ChunkIterator

Iterates through the chunks of a dataset, yielding each chunk's data as an ndarray. It is modeled on h5py's chunk iterator, but fetches each chunk's data through `Hdf5db.getDatasetValues()` rather than by slicing an `h5py.Dataset`, so it works uniformly across storage backends and reflects any pending in-memory updates. Only hyperslab (or "select all") selections without a step other than 1 are supported. Instances should be obtained via `Hdf5db.getChunkIterator()` rather than constructed directly.

### ChunkIterator.sel

Property. Returns the `Selection` (within the full dataset space) corresponding to the chunk most recently returned by `__next__`, or `None` before iteration has started.

## Hdf5db

Central in-memory store for HDF5 groups, datasets, and committed datatypes. Object state (new/dirty/deleted/resized) is tracked in memory and only written to persistent storage on `flush()` or `close()`; reads consult this in-memory state first so that unflushed changes are visible immediately. I/O is delegated to a single pluggable `StoragePlugin` supplied at construction (or later via the `plugin` property) that serves as both reader and writer, so a read always reflects whatever that plugin instance has most recently flushed; if none is supplied, `open()` installs a no-op `NullPlugin` instance.

### Hdf5db.getVersionInfo()

Static method. Returns a dict with a single key, `"hdf5-json-version"`, giving the package's API version string (`_apiver`).

### Hdf5db.db

Property. Returns the internal object dictionary (`_db`) mapping object id to that object's JSON representation (or `None` for a deleted object).

### Hdf5db.plugin

Property. Returns the current `StoragePlugin` instance, or `None` if none is set. The setter flushes and closes the current plugin (if set and not already closed) before installing the new plugin and calling `set_db()` on it so it can access this `Hdf5db` instance.

### Hdf5db.root_id

Property. Returns the root group's object id, as established by `open()`.

### Hdf5db.is_new(obj_id)

Returns `True` if `obj_id` (normalized via `getHashTagForId`) refers to an object that has been created but not yet persisted (i.e. is in the new-objects set).

### Hdf5db.is_dirty(obj_id)

Returns `True` if `obj_id` has been modified since the last flush. This is `True` for new objects, for datasets that have been resized, or for objects explicitly marked dirty via `make_dirty()`.

### Hdf5db.is_resized(dset_id)

Returns `True` if `dset_id` is in the set of datasets that have been resized since the last flush.

### Hdf5db.new_objects

Property. Returns the set of object ids created since the last flush.

### Hdf5db.dirty_objects

Property. Returns the set of object ids marked dirty (modified) since the last flush.

### Hdf5db.deleted_objects

Property. Returns the set of object ids deleted since the last flush.

### Hdf5db.resized_datasets

Property. Returns the set of dataset ids resized since the last flush.

### Hdf5db.make_dirty(obj_id)

Marks the object identified by `obj_id` as dirty and updates its `lastModified` timestamp. Raises `KeyError` (after logging an error) if `obj_id` is not present in the db. Does nothing further if the object has already been deleted (its db entry is `None`). New objects are not added to the dirty set, since they will be written in full on the next flush regardless.

### Hdf5db.flush()

Writes out all pending changes (new, dirty, deleted, and resized objects, plus queued dataset value updates) via the plugin. Calls `_checkPlugin()` first, which raises `IOError` if no plugin is set or the plugin is closed. Returns `False` without clearing any tracking sets if `plugin.flush()` reports failure; otherwise clears the new/dirty/deleted/resized sets and the pending dataset-updates map, and returns `True`.

### Hdf5db.readAll()

Recursively walks the object hierarchy starting at the root, fetching every reachable group, dataset, and committed datatype into the db via `getObjectById` (following hard links only). Raises `IOError` if the db is closed.

### Hdf5db.copy(other_db)

Writes this db's current content into `other_db`, which must already be open with its own plugin; raises `IOError` if it isn't. Used to convert between storage formats (e.g. the `h5tojson`/`jsontoh5` CLI apps) without a single `Hdf5db` juggling two different plugin types at once: open a source db and a destination db, each with its own plugin, then call `source_db.copy(dest_db)`.

Object ids are minted fresh in `other_db`; an internal id-translation map is built up as objects are copied so that any embedded object/region reference value (which otherwise still refers to this db's ids) is rewritten to the corresponding destination id. If a source dataset uses a committed (named) datatype, the destination dataset gets an equivalent inline (non-shared) dtype rather than a reference to a copied committed type — so datasets that shared one committed type in the source will each get their own independent (but structurally identical) type in the destination.

The copy proceeds in three passes over the object hierarchy reachable from `root_id` (following hard links only): first, every reachable group/dataset/datatype is created "empty" (a shell with no attributes, links, or values yet, but a real destination id) via `createGroup`/`createCommittedType`/`createDataset`, so that any reference to it can already be translated regardless of processing order, and so a hard link's target is always already known (this also correctly handles circular group references); second, each object's attributes and (for datasets) values are copied into its destination shell (chunk by chunk for a dataset, via `getChunkIterator`, so the whole dataset is never loaded into memory at once), with any reference-valued data translated via the id map, which is only safe to do once every object has a destination id; third, every group's links are recreated in the destination (`createHardLink`/`createSoftLink`/`createExternalLink`/`createCustomLink`) now that every possible hard-link target exists.

### Hdf5db.open()

Opens the storage plugin, installing a `NullPlugin` if none was set. Raises `IOError` if the plugin is already open (i.e. not closed). Calls `plugin.open()` to obtain the root id, checking it for consistency against any previously-set `root_id` (raising `IOError` on a mismatch) or else adopting it as `self._root_id`. If the root object isn't already cached in the db (e.g. a brand-new, empty store such as a fresh `H5JsonPlugin`), synthesizes a bare root group directly rather than fetching it; otherwise the root, like any other object, is left to be loaded lazily via `getObjectById()` on first access. Returns the root object id.

### Hdf5db.close()

Flushes and closes the plugin, unless it is a no-op `NullPlugin` (which can never persist anything, so flushing it would only log a spurious failure) — in that case just closes it.

### Hdf5db.closed

Property. Returns `True` if the plugin reports closed, or if no plugin is set but a `root_id` has been established; returns `False` if no plugin or root id is present.

### Hdf5db.getObjectById(obj_id, refresh=False)

Returns the JSON representation of the object identified by `obj_id`. Calls `_checkPlugin()` first (raises `IOError` if no plugin is set or the plugin is closed). If the object is not already cached in the db, or `refresh=True` and the object is neither new nor dirty, fetches it fresh from the plugin and stores it in the db. Returns the (possibly in-memory-modified) object JSON, or `None` if the object was deleted.

### Hdf5db.getObjectIdByPath(h5path, parent_id=None)

Resolves a slash-separated path to an object id, starting the walk from `parent_id` (or the root id if not given). Returns `root_id` unchanged for `h5path == "/"`. Only hard links are followed; soft and external links along the path cause the lookup to fail. Raises `KeyError` if any path component cannot be resolved.

### Hdf5db.getObjectByPath(path)

Convenience wrapper that resolves `path` via `getObjectIdByPath` and returns the corresponding object JSON via `getObjectById`.

### Hdf5db.getPathsForObjectId(obj_id, parent_id=None, path_prefix="", _visited=None)

Recursively searches the hierarchy below `parent_id` (default root) for all hard-link paths that resolve to `obj_id`, returning a list of path strings. Guards against link cycles using the `_visited` set (logging a warning and stopping recursion at a repeat). Only hard links are traversed; soft and external links are skipped with a warning. Noted in the code as potentially slow for domains with many objects, since it walks the entire hierarchy.

### Hdf5db.getDtype(obj_json)

Returns the numpy `dtype` for a dataset, committed datatype, or attribute JSON object. If the object's `"type"` field references a committed datatype (a valid UUID belonging to the `"datatypes"` collection), the datatype object is fetched and used; otherwise the type is built directly from the inline type JSON via `createDataType`. Raises `TypeError` if `obj_json` has no `"type"` key, or `KeyError` if a referenced committed datatype cannot be found.

### Hdf5db.getAttributes(obj_id)

Returns a list of attribute names defined on the object `obj_id`, excluding any attributes marked as deleted.

### Hdf5db.getAttribute(obj_id, name, includeData=True)

Returns the JSON representation of attribute `name` on object `obj_id`, or `None` if the attribute does not exist or has been deleted. (The `includeData` parameter is accepted but not used to filter the returned JSON.)

### Hdf5db.getAttributeValue(obj_id, name)

Returns the attribute's value as a numpy array (or scalar-shaped array), converted from its JSON representation via `jsonToArray` using the attribute's shape and dtype. Returns `None` for attributes with an `H5S_NULL` dataspace. Raises `KeyError` if the attribute is not found.

### Hdf5db.createAttribute(obj_id, name, value, shape=None, dtype=None)

Creates (or overwrites) attribute `name` on object `obj_id` with the given `value`, `shape`, and `dtype`. Handles committed-datatype references passed as a `"datatypes/<id>"` string, `Reference` values, null-space attributes (`shape="H5S_NULL"`), and array/sub-array dtypes. Converts `value` to a numpy array (falling back to `jsonToArray` for compound/vlen types), validates shape compatibility, and serializes the value back to JSON via `bytesArrayToList`. Sets `"encoding": "base64"` for opaque dtypes. Marks the object dirty. Raises `ValueError` or `TypeError` for shape/dtype mismatches.

### Hdf5db.deleteAttribute(obj_id, name)

Marks attribute `name` on object `obj_id` as deleted (sets a `"DELETED"` timestamp rather than removing it outright) and marks the object dirty. Raises `KeyError` if the attribute is not present.

### Hdf5db.getDatasetValues(dset_id, sel, query=None)

Reads values from dataset `dset_id` for the given `Selection` `sel`, returning a numpy array (or `None` for a null-space dataset). If `query` is given, delegates to the query path (equivalent to filtering the selection by a boolean expression) and returns a 1-D array of the matching full-record values rather than the values of the raw selection. Otherwise: for scalar datasets, returns the pending update value if one exists, an initialized (fill-value) array if the dataset is new, or the plugin's value; for simple datasets, starts from the plugin's data (or a fill-initialized array if the plugin has no relevant data, e.g. the dataset is new or fully covered by pending updates) and overlays any pending `setDatasetValues` updates that intersect `sel`, correctly handling hyperslab, point, and paired-point/fancy selections and compound field restrictions. Raises `TypeError` if `sel` is not a `Selection`, or `ValueError` if `sel`'s shape does not match the dataset's shape.

### Hdf5db.getChunkIterator(dset_id, sel=None)

Returns a `ChunkIterator` over dataset `dset_id`, restricted to `sel` if given (otherwise the whole dataset), allowing large datasets to be read chunk by chunk without loading everything into memory.

### Hdf5db.queryDataset(dset_id, query, sel=None, limit=0, update_value=None)

Evaluates the boolean expression `query` against dataset `dset_id`, restricted to selection `sel` (defaults to the whole dataset), and returns a numpy array of coordinate indices (shape `(N, rank)`) for the matching elements, capped at `limit` elements if `limit > 0`. Delegates to `plugin.queryDataset()` when the plugin can answer for the requested region and no pending update supersedes it, falling back to iterating the selection chunk by chunk (via `ChunkIterator`) and evaluating `arrayQuery` on each chunk, or on the whole selection at once if a `ChunkIterator` cannot be constructed for it (e.g. point/fancy selections). Pending in-memory updates are queried directly and merged into (or subtracted from, where they invalidate stale plugin results) the match mask. If `update_value` is provided, the matching elements (up to `limit`) are overwritten with `update_value` (which may be a dict mapping field names to values for a compound dtype, updating only those fields) by first flushing the db and delegating to `plugin.queryDataset()`/`setDatasetValues()`. Raises `TypeError` for a non-string `query`, non-`Selection` `sel`, or negative `limit`, and `ValueError` for null-space datasets.

### Hdf5db.setDatasetValues(dset_id, sel, arr)

Writes ndarray `arr` into dataset `dset_id` at selection `sel` by queueing the (selection, array) pair as a pending update and marking the dataset dirty; the update is only materialized in persistent storage on the next `flush()`. Validates that `sel` is a `Selection`, `arr` is an ndarray, `arr`'s dtype matches the dataset's (or selected fields') dtype, and that `arr`'s shape is compatible with the selection type (hyperslab, points, fancy, or select-all), reshaping/broadcasting as needed. If the write fully covers the dataset with no field restriction, prior pending updates are discarded since this write supersedes them. Raises `ValueError` for null-space datasets or scalar/shape mismatches, and `TypeError` for dtype or shape mismatches.

### Hdf5db.resizeDataset(dset_id, shape)

Resizes dataset `dset_id`'s dataspace to `shape` via `resize_dataset()`, adds the dataset to the resized set (unless it is a new, not-yet-persisted object), and trims any pending hyperslab updates whose bounds now fall outside the new dimensions. If any dimension shrank, immediately calls `flush()` so the resize is applied before further operations.

### Hdf5db.deleteObject(obj_id)

Deletes object `obj_id` by setting its db entry to `None`, removing it from the new/dirty/resized sets if present, and adding it to the deleted-objects set. Raises `KeyError` if the object is not found or is the root group.

### Hdf5db.getLinks(grp_id)

Returns a list of link names defined in group `grp_id`, excluding links marked as deleted. Raises `KeyError` if `grp_id`'s JSON has no `"links"` key.

### Hdf5db.getLink(grp_id, name)

Returns the JSON for link `name` in group `grp_id`, or `None` if the link does not exist or has been deleted.

### Hdf5db.createHardLink(grp_id, name, tgt_id)

Creates a hard link named `name` in group `grp_id` pointing to object `tgt_id`, stamped with a creation timestamp, and marks `grp_id` dirty.

### Hdf5db.createSoftLink(grp_id, name, h5path)

Creates a soft link named `name` in group `grp_id` targeting path `h5path`, stamped with a creation timestamp, and marks `grp_id` dirty.

### Hdf5db.createCustomLink(grp_id, name, link_json)

Adds a user-defined link named `name` in group `grp_id` from the supplied `link_json`, forcing its `"class"` to `"H5L_TYPE_USER_DEFINED"` if not already set, stamping a creation timestamp, and marking `grp_id` dirty.

### Hdf5db.createExternalLink(grp_id, name, h5path, filepath)

Creates an external link named `name` in group `grp_id` referencing path `h5path` in external file `filepath`, stamped with a creation timestamp, and marks `grp_id` dirty.

### Hdf5db.deleteLink(grp_id, name)

Marks link `name` in group `grp_id` as deleted (sets a `"DELETED"` timestamp) and marks `grp_id` dirty. Raises `KeyError` if `grp_id` has no `"links"` key or `name` is not present.

### Hdf5db.createGroup(cpl=None)

Creates a new, empty group (no links or attributes) with a freshly generated object id, optionally recording creation properties `cpl`, adds it to the db and the new-objects set, and returns its id. Raises `ValueError` if the db is closed.

### Hdf5db.createCommittedType(datatype, cpl=None)

Creates a new named (committed) datatype from `datatype` (a numpy `dtype` or a type description convertible via `createDataType`), optionally with creation properties `cpl`, adds it to the db and the new-objects set, and returns its object id. Raises `ValueError` if the db is closed.

### Hdf5db.createDataset(shape=None, maxdims=None, dtype=None, cpl=None)

Creates a new dataset with the given `shape`, `maxdims` (for resizable/extensible dimensions), `dtype`, and creation properties `cpl`. Validates and normalizes any `"filters"` entry in `cpl` against the plugin's supported filters, and validates the `"fillValue"` entry for compatibility with `dtype` (including compound-type fill values). Raises `ValueError` if `maxdims` is given but the dataset's layout is not `H5D_CHUNKED`, or if the db is closed, or for an invalid fill value. Adds the new dataset to the db and the new-objects set, and returns its object id.

### Hdf5db.getCollection(col_type=None)

Returns a list of object ids currently in the db (excluding deleted objects), optionally filtered to only those belonging to collection `col_type` (e.g. `"groups"`, `"datasets"`, `"datatypes"`).
