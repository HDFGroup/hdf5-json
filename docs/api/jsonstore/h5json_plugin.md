# h5json_plugin

Implements the `StoragePlugin` interface (`h5json.storage_plugin.StoragePlugin`) for h5json's own `.json` file format. A single in-memory `self.json` dict is the source of truth for both reading and writing — it starts either loaded from an existing file (`append=True` or `read_only=True`, and the file already exists) or empty (a fresh file), and is progressively updated by `flush()`'s dump methods, so a read always sees whatever this same instance has most recently loaded or flushed. Since the on-disk representation is already close to the in-memory model `Hdf5db` operates on, reading is largely a matter of reshaping the document's `groups`/`datasets`/`datatypes` collections into the object dicts the rest of the codebase expects, rather than translating between two different data models as the h5py-backed plugin must. The `dump*` methods each build the JSON dict for one kind of object (group, dataset, datatype, attribute, link) from the current `Hdf5db` state, mirroring the on-disk schema — for example propagating `"encoding": "base64"` onto a dataset's or attribute's dict whenever its type is `H5T_OPAQUE`.

## H5JsonPlugin

Concrete `StoragePlugin` implementation that holds the document in memory as `self.json` and writes it to `self.filepath` (or prints it) when flushed. `append` and `read_only` both cause `open()` to load an existing file at `filepath` into `self.json` (rather than starting from an empty document) if one is present. An optional `data_limit` caps how large a dataset's element data can be (in bytes, estimated) before its `"value"` is omitted from the dump; passing `data_limit=0` is equivalent to `no_data=True` on the base class. Unlike the h5py-backed plugin, there is no incremental/dirty-tracking write path for the file itself: the whole document (or the parts affected since the previous flush) is rebuilt and written out in one pass by `dumpFile()`, driven by an `_init` flag that is `True` until the first `flush()` completes (unless constructed with `append=True` or `read_only=True`), forcing every group/dataset/datatype to be (re-)dumped on that first flush even if not marked dirty.

### H5JsonPlugin.open()

Opens the json store: if `append` or `read_only` is set and the file already exists, loads and parses it into `self.json`, requiring a `"root"` key (raises `Exception` if absent) and deriving the root object id as `"g-" + json["root"]` — raising `IOError` if that doesn't match an already-set `Hdf5db.root_id`. Otherwise starts from an empty `self.json`, establishing the root id from `self.db.root_id` if set or minting a new one via `createObjId`. A no-op (returning the existing root id) if already open.

### H5JsonPlugin.close()

Clears `self._root_id` and resets `self.json` to `{}`. Does not flush — `Hdf5db.close()` (the only caller) always calls `Hdf5db.flush()` immediately beforehand, which itself calls this plugin's `flush()`; re-flushing here would be redundant (and, for a stdout-destined plugin, would print the dump a second time).

### H5JsonPlugin.isClosed()

Returns `True` if `self._root_id` is not set (i.e. `open()` hasn't been called, or `close()` has), `False` otherwise.

### H5JsonPlugin.get_root_id()

Returns the root object id established by `open()`.

### H5JsonPlugin.getObjectById(obj_id, include_attrs=True, include_links=True, include_values=False)

Looks up the JSON object for `obj_id` by resolving its collection (`groups`/`datasets`/`datatypes` via `getCollectionForId`) and uuid, returning `None` (with a warning logged) if the collection or uuid isn't found in `self.json`. Builds a response dict by copying whichever of `shape`, `type`, `cpl`, `dcpl`, `creationProperties`, `encoding` are present on the source object. If `include_attrs` is true and the object has an `"attributes"` list, converts it from the on-disk list-of-dicts form (each with a `"name"` key) into a dict keyed by attribute name, each entry carrying `type`, `shape`, and optionally `value`/`encoding`. If `include_links` is true and the object has a `"links"` list, similarly converts it to a dict keyed by link `"title"`, additionally resolving any `"collection"` + `"id"` pair on a hard-link item into a combined h5json object id (`"g-"`, `"d-"`, or `"t-"` prefixed) stored back into the item's `"id"` key. If `include_values` is true and the object is a dataset with a `"value"` key, that value is included in the response as `"value"`.

### H5JsonPlugin.getAttribute(obj_id, name, includeData=True)

Returns the JSON dict for a single named attribute of `obj_id`, obtained via `getObjectById`. Returns `None` (with a log message) if the object doesn't exist, has no attributes collection, or has no attribute of that name. If `includeData` is false, returns a copy of the attribute dict with `"value"`/`"encoding"` removed.

### H5JsonPlugin.getDtype(obj_json)

Derives a numpy dtype from an object's JSON `type` item via `createDataType`. Raises `KeyError` if `obj_json` has no `"type"` key. If the type item is a string reference to a committed datatype (starts with `"datatypes/"`), resolves the referenced datatype object via `getObjectById` and uses its `type` item instead, raising `KeyError` if that lookup comes back without a `"type"` key.

### H5JsonPlugin.getDatasetValues(obj_id, sel=None, dtype=None, query=None)

Reads dataset values for `obj_id`, applying selection `sel`. Returns `None` if the object or its `"value"` key isn't found, or (with a warning) if the dataset has a null-space shape. Determines `dims` from the shape JSON (`()` for scalar, `shape["dims"]` otherwise), converts the raw JSON value into a numpy array via `jsonToArray(dims, dtype, json_value)`, then applies the selection: unselected/`H5S_SEL_ALL` returns the whole array; a `SimpleSelection` indexes it with `arr[sel.slices]`; any other selection type raises `NotImplementedError`. Raises `NotImplementedError` if `query` is given, since the JSON store has no query pushdown (the caller falls back to evaluating the query itself over the fetched values).

### H5JsonPlugin.getAliasList(obj_id)

Returns the list of HDF5 paths (aliases) at which `obj_id` is reachable, delegating to `self.db.getPathsForObjectId(obj_id)`.

### H5JsonPlugin.dumpAttribute(obj_id, attr_name)

Builds the JSON dict for one attribute, fetched via `self.db.getAttribute(obj_id, attr_name)`. Includes `"name"`, `"type"`, and `"shape"` always; includes `"value"` (and `"encoding"` if present) when the attribute has a value, logging a warning if it doesn't.

### H5JsonPlugin.dumpAttributes(obj_id)

Returns a list of attribute dicts (via `dumpAttribute`) for every attribute on `obj_id`, as reported by `self.db.getAttributes(obj_id)`.

### H5JsonPlugin.dumpLink(obj_id, name)

Builds the JSON dict for one link named `name` on `obj_id`, fetched via `self.db.getLink(obj_id, name)`. Always includes `"class"`; for a hard link (one with an `"id"` key), adds `"collection"` (derived from the target id via `getCollectionForId`) and `"id"` (the target's bare uuid via `getUuidFromId`). Copies through any other keys on the source item except `id`, `created`, and `modified`, and sets `"title"` to `name`.

### H5JsonPlugin.dumpLinks(obj_id)

Returns a list of link dicts (via `dumpLink`) for every link on `obj_id`, as reported by `self.db.getLinks(obj_id)`.

### H5JsonPlugin.dumpGroup(obj_id)

Builds the JSON dict for a group: `"alias"` (via `getAliasList`), `"creationProperties"` (from the source's `"cpl"` or `"creationProperties"` key if present), `"attributes"` (via `dumpAttributes`, included only if non-empty), and `"links"` (via `dumpLinks`, included only if non-empty).

### H5JsonPlugin.dumpGroups()

Updates the `"groups"` collection in `self.json`: (re-)dumps the root group and every other group returned by `self.db.getCollection("groups")` whenever `self._init` is set, the group isn't already present in `self.json["groups"]`, or `self.db.is_dirty(obj_id)`; leaves any other existing entry untouched (so a value already written by an earlier flush in the same session isn't needlessly recomputed). Removes entries for any group no longer present in the db (i.e. deleted since the last flush).

### H5JsonPlugin.dumpDataset(obj_id)

Builds the JSON dict for a dataset: `"alias"`, `"type"`, and a `"shape"` dict (`class`, and `dims`/`maxdims` as applicable — `maxdims` entries of `0` are rewritten to the string `"H5S_UNLIMITED"`). Includes `"creationProperties"` from the source `"cpl"` key if present, and `"attributes"` if non-empty. If `self._data_limit` is set, estimates the dataset's total byte size (`getItemSize(type_item)` times element count, treating a variable-length item size as 1024 bytes for the estimate) and skips writing `"value"` (logging that it's being skipped) when the estimate exceeds the limit; otherwise — or when no limit is configured — reads the full dataset value via `self.db.getDatasetValues` with an all-selection and serializes it with `bytesArrayToList` into `"value"`, adding `"encoding": "base64"` when the type is `H5T_OPAQUE`. A null-space dataset (`num_elements == 0`) never gets a `"value"` key.

### H5JsonPlugin.dumpDatasets()

Updates the `"datasets"` collection in `self.json` from every id returned by `self.db.getCollection("datasets")`, (re-)dumping an entry only when `self._init` is set, it isn't already present, or `self.db.is_dirty(obj_id)`. Removes entries for any dataset deleted since the last flush, and removes the `"datasets"` key entirely if it ends up empty.

### H5JsonPlugin.dumpDatatype(obj_id)

Builds the JSON dict for a committed datatype: `"alias"`, `"type"`, `"creationProperties"` (from `"cpl"` if present), and `"attributes"` (if non-empty).

### H5JsonPlugin.dumpDatatypes()

Updates the `"datatypes"` collection in `self.json` from every id returned by `self.db.getCollection("datatypes")`, (re-)dumping an entry only when `self._init` is set, it isn't already present, or `self.db.is_dirty(obj_id)`. Removes entries for any datatype deleted since the last flush, and removes the `"datatypes"` key entirely if it ends up empty.

### H5JsonPlugin.dumpFile()

Top-level entry point that (re-)builds the h5json document in `self.json`: resolves the root object's uuid via `self.db.getObjectIdByPath("/")`, sets `"apiVersion"` from `self.db.getVersionInfo()["hdf5-json-version"]` and `"root"` to the root's bare uuid, then calls `dumpGroups()`, `dumpDatasets()`, and `dumpDatatypes()` in turn. If `self._filepath` is set, writes the document to that path as indented JSON (`json.dump`, ASCII-escaped, indent from `self._indent`); otherwise prints the document (sorted keys) to stdout. Updates `self._lastModified` to the current time when done.

### H5JsonPlugin.flush()

Writes pending changes. May be called more than once per session (e.g. `Hdf5db`'s periodic auto-flush, or `close()` flushing before its own final flush); each call re-dumps only the objects that are new/dirty/resized (or removes deleted ones) since the previous flush — see the `_init` handling on the `dumpGroups`/`dumpDatasets`/`dumpDatatypes` methods — since recomputing an unchanged dataset's value via `Hdf5db.getDatasetValues()` would incorrectly return a zero/fill-value array once its pending update has already been cleared by an earlier flush. Raises `IOError` if called before `open()`. If `read_only` is set, never writes: returns `True` if there's nothing pending to persist, or logs a warning and returns `False` if there is (in-memory-only edits are left un-flushed rather than raising). Otherwise calls `dumpFile()` and clears `self._init`. Always returns `True` when it doesn't raise.

### H5JsonPlugin.getStats()

Returns a dict with `created`, `lastModified`, and `owner` keys derived from `os.stat()` on the underlying file path (`st_ctime`, `st_mtime`, `st_uid` respectively). `owner` is the raw numeric uid, not resolved to a username.

### H5JsonPlugin.getFilters(compressors_only=False)

Always returns an empty tuple — the JSON store format has no notion of HDF5 compression filters.
