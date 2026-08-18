# h5json_reader

Implements the `H5Reader` interface (`h5json.h5reader.H5Reader`) for h5json's own `.json` file format. Since the on-disk representation is already close to the in-memory model `Hdf5db` operates on, this reader is largely a matter of loading the JSON document once and reshaping its `groups`/`datasets`/`datatypes` collections into the object dicts the rest of the codebase expects, rather than translating between two different data models as the h5py-backed reader must.

## H5JsonReader

Concrete `H5Reader` implementation backed by a parsed h5json JSON document held in memory as `self._h5json`.

### H5JsonReader.open()

Reads and parses the JSON file at `self.filepath` if not already loaded. Requires a `"root"` key in the document (raises a plain `Exception` if absent) and derives the root object id as `"g-" + h5json["root"]`. If the owning `Hdf5db` already has a `root_id` that doesn't match, logs a warning and raises `IOError`. Returns the root object id. A no-op if the JSON has already been read.

### H5JsonReader.close()

Discards the in-memory parsed JSON (`self._h5json = None`).

### H5JsonReader.isClosed()

Returns `True` if no JSON document is currently loaded, `False` otherwise.

### H5JsonReader.get_root_id()

Returns the root object id established by `open()`.

### H5JsonReader.getObjectById(obj_id, include_attrs=True, include_links=True, include_values=False)

Looks up the JSON object for `obj_id` by resolving its collection (`groups`/`datasets`/`datatypes` via `getCollectionForId`) and uuid, returning `None` (with a warning logged) if the collection or uuid isn't found in the document. Builds a response dict by copying whichever of `shape`, `type`, `cpl`, `dcpl`, `encoding` are present on the source object. If `include_attrs` is true and the object has an `"attributes"` list, converts it from the on-disk list-of-dicts form (each with a `"name"` key) into a dict keyed by attribute name, each entry carrying `type`, `shape`, and optionally `value`/`encoding`. If `include_links` is true and the object has a `"links"` list, similarly converts it to a dict keyed by link `"title"`, additionally resolving any `"collection"` + `"id"` pair on a hard-link item into a combined h5json object id (`"g-"`, `"d-"`, or `"t-"` prefixed) stored back into the item's `"id"` key. If `include_values` is true and the object is a dataset with a `"value"` key, that value is included in the response as `"value"`.

### H5JsonReader.getAttribute(obj_id, name, includeData=True)

Returns the JSON dict for a single named attribute of `obj_id`, obtained via `getObjectById`. Returns `None` (with a log message) if the object doesn't exist, has no attributes collection, or has no attribute of that name. Note: despite the `includeData` parameter being accepted, it is not actually used to filter out the value in this implementation — the value is always present as returned by `getObjectById`.

### H5JsonReader.getDtype(obj_json)

Derives a numpy dtype from an object's JSON `type` item via `createDataType`. Raises `KeyError` if `obj_json` has no `"type"` key. If the type item is a string reference to a committed datatype (starts with `"datatypes/"`), resolves the referenced datatype object via `getObjectById` and uses its `type` item instead, raising `KeyError` if that lookup comes back without a `"type"` key.

### H5JsonReader.getDatasetValues(obj_id, sel=None, dtype=None, query=None)

Reads dataset values for `obj_id`, applying selection `sel`. Returns `None` if the object or its `"value"` key isn't found, or (with a warning) if the dataset has a null-space shape. Determines `dims` from the shape JSON (`()` for scalar, `shape["dims"]` otherwise), converts the raw JSON value into a numpy array via `jsonToArray(dims, dtype, json_value)`, then applies the selection: unselected/`H5S_SEL_ALL` returns the whole array; a `SimpleSelection` indexes it with `arr[sel.slices]`; any other selection type raises `NotImplementedError`. Raises `NotImplementedError` if `query` is given, since the JSON store has no query pushdown (the caller falls back to evaluating the query itself over the fetched values).

### H5JsonReader.getStats()

Returns a dict with `created`, `lastModified`, and `owner` keys derived from `os.stat()` on the underlying file path (`st_ctime`, `st_mtime`, `st_uid` respectively). `owner` is the raw numeric uid, not resolved to a username.
