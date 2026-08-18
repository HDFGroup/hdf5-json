# h5py_reader

Implements the `H5Reader` interface (`h5json.h5reader.H5Reader`) for real HDF5 `.h5` files, using the h5py library to pull groups, datasets, attributes, datatypes, and links out of an on-disk file and translate them into the h5json in-memory/JSON representation. Two areas need special handling beyond a straight h5py-to-h5json copy: opaque-typed data, which real HDF5 tags with an arbitrary tag string that h5py's high-level API requires the memory type to match (so opaque reads go through the low-level `h5t`/`h5d`/`h5a` API instead), and region references, where a real HDF5 region reference only supports point or hyperslab selections and so cannot capture the broader "FANCY" (multi-dimension coordinate list) selection concept this codebase otherwise supports for its own JSON format — reading a region reference here resolves only the target dataset, with no selection bound.

## H5pyReader

Concrete `H5Reader` implementation backed by an h5py `File` handle. It maintains two internal maps built up as objects are discovered: `_id_map` (h5json object id -> live h5py object) and `_addr_map` (HDF5 object address -> h5json object id), the latter used to detect when multiple hard links/paths resolve to the same underlying object.

### H5pyReader.open()

Opens the underlying HDF5 file with `h5py.File(self.filepath)` (read-only, default mode) if not already open. Establishes the root id — reusing `self.db.root_id` if the owning `Hdf5db` already has one, otherwise minting a new one via `createObjId` — and registers the root group's object in `_id_map`/`_addr_map`. Returns the root object id. A no-op if the file handle is already open or objects have already been loaded.

### H5pyReader.close()

Clears `_id_map` and closes the underlying h5py file handle, setting it to `None`. Safe to call when already closed.

### H5pyReader.isClosed()

Returns `True` if there is no open h5py file handle, `False` otherwise.

### H5pyReader.get_root_id()

Returns the root object id established by `open()`.

### H5pyReader.getObjIdByAddress(addr)

Looks up the h5json object id previously associated with an HDF5 object address (`addr`, as returned by `h5py.h5o.get_info(...).addr`). Returns `None` if no id has been registered for that address yet — used to resolve committed-datatype and hard-link targets that may not have been visited yet.

### H5pyReader.getAttribute(obj_id, name, include_data=True)

Reads a single attribute named `name` off the object identified by `obj_id` and returns it as an h5json attribute dict (`type`, `shape`, optionally `value`/`encoding`, and `created`). Detects whether the attribute uses a committed datatype (via `h5py.h5t.TypeID.committed`) and resolves it to the corresponding datatype object's id; otherwise derives the type item from `attrObj.dtype` via `getTypeItem`. Shape is reported as `H5S_NULL` (also inferred when storage size is 0, working around an h5py quirk where null-space attributes report no shape), `H5S_SCALAR`, or `H5S_SIMPLE` with `dims`. When `include_data` is true and the attribute is opaque-typed, data is read via the low-level path (`_readOpaqueAttribute`) instead of `obj.attrs[name]`; the result is passed through `_copy_array` to translate h5py reference/vlen values into h5json equivalents, then flattened to a JSON-serializable value with `bytesArrayToList`. Opaque attributes get `item["encoding"] = "base64"`. Returns `None` if the named attribute doesn't exist.

### H5pyReader.getAttributes(obj_id, include_data=True)

Returns an ordered dict of all attributes on the object identified by `obj_id`, keyed by attribute name, each built via `getAttribute`. Relies on Python 3.7+ dict ordering to preserve the attribute order reported by h5py.

### H5pyReader.getObjectById(obj_id, include_attrs=True, include_links=True)

Central per-object read: looks up the live h5py object for `obj_id` in `_id_map` (raising `KeyError` if not found) and dispatches on its type — `h5py.Group` (also registers ids for any not-yet-seen hard-linked children via the internal hard-link scan before building the group JSON), `h5py.Dataset`, or `h5py.Datatype` — to build the object's JSON representation. Raises `TypeError` for any other object type. If `include_attrs` is true, attaches an `"attributes"` key built via `getAttributes`.

### H5pyReader.getDatasetValues(dset_id, sel, dtype=None, query=None)

Reads dataset values for the dataset identified by `dset_id`, applying the given selection `sel`. Returns `None` immediately for a null-space dataset. Raises `NotImplementedError` if `query` is given, since h5py has no server-side query support (the caller, `Hdf5db`, falls back to evaluating the query in Python over the fetched values). For opaque-typed datasets, the whole dataset is read tag-matched via `_readOpaqueDataset` and then the selection is applied with plain numpy indexing (which, unlike h5py's own dataspace selection, tolerates a paired-coordinate multi-dimension list selection). For an all-selection or `None` selection, the whole dataset is read with `dset[...]`. For a `SimpleSelection` with more than one list-valued dimension (h5py only supports one coordinate array per read), the read is decomposed into one indexed read per paired coordinate and the results are stacked with `np.stack`; otherwise a single `dset[slices]` read is used. Other selection types raise `NotImplementedError`. The resulting array is passed through `_copy_array` to convert any h5py references into h5json reference values before being returned.

### H5pyReader.getStats()

Returns a dict with `created`, `lastModified`, and `owner` keys derived from `os.stat()` on the underlying file path (`st_ctime`, `st_mtime`, `st_uid` respectively). `owner` is the raw numeric uid, not resolved to a username.
