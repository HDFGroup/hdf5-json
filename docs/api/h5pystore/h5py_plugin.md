# h5py_plugin

Implements the `StoragePlugin` interface (`h5json.storage_plugin.StoragePlugin`) for real HDF5 `.h5` files, using the h5py library to read and write groups, datasets, attributes, datatypes, and links, translating between real HDF5 and the h5json in-memory/JSON representation. A single instance is both the reader and the writer for a given file, so a read always sees whatever this same instance has most recently written — including changes made earlier in the same session to objects that existed before it opened. Three areas need special handling beyond a straight h5py-to-h5json translation: opaque-typed data, which real HDF5 tags with an arbitrary tag string that h5py's high-level API requires the memory type to match (so opaque reads/writes go through the low-level `h5t`/`h5d`/`h5a` API instead); region references, where a real HDF5 region reference only supports point or hyperslab selections, so writing one requires building the dataspace selection via low-level `h5s`/`h5r` calls (h5py's high-level slicing supports only a single fancy-index array and cannot express a paired-coordinate point selection at all) and reading one back resolves only the target dataset, with no selection bound (since the broader "FANCY" multi-dimension coordinate list selection this codebase otherwise supports for its own JSON format has no HDF5 region-reference equivalent); and object-dtype (vlen/reference) array writes, which go through `write_direct`/`attrs.create` rather than plain assignment, to avoid numpy silently homogenizing same-length object-array elements into a regular N-d array during a high-level write (which then makes h5py reject the resulting shape).

## H5pyPlugin

Concrete `StoragePlugin` implementation backed by an h5py `File` handle opened for reading and writing (or appending). Maintains `_id_map` (h5json object id -> live h5py object) and `_addr_map` (HDF5 object address -> h5json object id) — built up as objects are discovered on read and extended as objects are created on write — the latter used to detect when multiple hard links/paths resolve to the same underlying object and to resolve reference targets. An `_init` flag is `True` until the first `flush()` completes (unless constructed with `append=True` or `read_only=True`), indicating that every object needs its values (not just deltas) written.

### H5pyPlugin.resizeDataset(dset_id, dset)

Resizes the live h5py dataset `dset` to match the current dims recorded for `dset_id` in the db (via `dset.resize(new_dims)`), used when a dataset's extent has grown/shrunk since it was created.

### H5pyPlugin.updateDatasetValues(dset_id, dset)

Applies pending (dirty) writes for `dset_id`, fetched from `self.db._getDatasetUpdates(dset_id)` as a list of `(selection, value)` pairs, onto the live h5py dataset `dset`. Dispatches on selection type: `H5S_SEL_NONE` is a no-op; `H5S_SEL_ALL` writes the whole array; `H5S_SEL_HYPERSLABS` builds a tuple of Python `slice` objects from the selection's start/count/step and writes the region; `H5S_SEL_POINTS` writes one point at a time in a loop; `H5S_SEL_FANCY` with more than one coordinate-list dimension decomposes into one indexed write per paired coordinate (h5py can only handle a single fancy-index array per assignment), otherwise assigns directly. Raises `TypeError` for an unrecognized selection type.

### H5pyPlugin.createAttribute(obj, name, attr_json)

Creates attribute `name` on the live h5py object `obj` from its h5json representation `attr_json`. Handles a null-space attribute specially, assigning `h5py.Empty(...)` with the converted dtype. For scalar/simple shapes, builds the source numpy array with `jsonToArray` and converts it to an h5py-compatible array, then writes it via `obj.attrs.create(name, data=tgt_arr, dtype=tgt_arr.dtype)` rather than plain `obj.attrs[name] = tgt_arr` — passing the dtype explicitly preserves the actual object dtype (e.g. vlen-of-reference) instead of letting h5py re-infer a possibly-wrong dtype from the array's runtime contents.

### H5pyPlugin.updateAttributes(obj_id, obj)

Walks the `"attributes"` dict on `obj_id`'s current JSON (from `self.db.getObjectById`) and, for each attribute: deletes it from the live h5py object `obj` if marked `"DELETED"` (ignoring the case where it's already absent); skips it if it was already flushed (its `"created"` timestamp predates `self._flush_time`); otherwise creates/overwrites it via `createAttribute`. Does nothing if the object has no `"attributes"` key.

### H5pyPlugin.getAttribute(obj_id, name, include_data=True)

Reads a single attribute named `name` off the object identified by `obj_id` and returns it as an h5json attribute dict (`type`, `shape`, optionally `value`/`encoding`, and `created`). Detects whether the attribute uses a committed datatype (via `h5py.h5t.TypeID.committed`) and resolves it to the corresponding datatype object's id; otherwise derives the type item from `attrObj.dtype` via `getTypeItem`. Shape is reported as `H5S_NULL` (also inferred when storage size is 0, working around an h5py quirk where null-space attributes report no shape), `H5S_SCALAR`, or `H5S_SIMPLE` with `dims`. When `include_data` is true and the attribute is opaque-typed, data is read via the low-level path instead of `obj.attrs[name]`; the result is converted to h5json equivalents (translating any h5py reference/vlen values) then flattened to a JSON-serializable value with `bytesArrayToList`. Opaque attributes get `item["encoding"] = "base64"`. Returns `None` if the named attribute doesn't exist.

### H5pyPlugin.getAttributes(obj_id, include_data=True)

Returns an ordered dict of all attributes on the object identified by `obj_id`, keyed by attribute name, each built via `getAttribute`. Relies on Python 3.7+ dict ordering to preserve the attribute order reported by h5py.

### H5pyPlugin.getObjectById(obj_id, include_attrs=True, include_links=True)

Central per-object read: looks up the live h5py object for `obj_id` in `_id_map` (raising `KeyError` if not found) and dispatches on its type — `h5py.Group` (also registers ids for any not-yet-seen hard-linked children before building the group JSON), `h5py.Dataset`, or `h5py.Datatype` — to build the object's JSON representation. Raises `TypeError` for any other object type. If `include_attrs` is true, attaches an `"attributes"` key built via `getAttributes`.

### H5pyPlugin.getDatasetValues(dset_id, sel, dtype=None, query=None)

Reads dataset values for the dataset identified by `dset_id`, applying the given selection `sel`. Returns `None` immediately for a null-space dataset. Raises `NotImplementedError` if `query` is given, since h5py has no server-side query support (the caller, `Hdf5db`, falls back to evaluating the query in Python over the fetched values). For opaque-typed datasets, the whole dataset is read tag-matched via the low-level API and then the selection is applied with plain numpy indexing (which, unlike h5py's own dataspace selection, tolerates a paired-coordinate multi-dimension list selection). For an all-selection or `None` selection, the whole dataset is read with `dset[...]`. For a `SimpleSelection` with more than one list-valued dimension (h5py only supports one coordinate array per read), the read is decomposed into one indexed read per paired coordinate and the results are stacked with `np.stack`; otherwise a single `dset[slices]` read is used. Other selection types raise `NotImplementedError`. The resulting array is converted to translate any h5py references into h5json reference values before being returned.

### H5pyPlugin.getObjIdByAddress(addr)

Looks up the h5json object id previously associated with an HDF5 object address (`addr`, as returned by `h5py.h5o.get_info(...).addr`). Returns `None` if no id has been registered for that address yet.

### H5pyPlugin.get_root_id()

Returns the root object id established by `open()`.

### H5pyPlugin.flush()

Writes all pending (dirty) state to the HDF5 file. Returns `False` without writing if the plugin is closed; raises `IOError` if the file handle isn't open. If `read_only` is set, never writes: returns `True` if there's nothing pending to persist, or logs a warning and returns `False` if there is (in-memory-only edits are left un-flushed rather than raising). Otherwise, whenever there's anything pending (new/dirty/resized objects, or on the first flush), walks the root group's links and creates any new objects via an internal recursion, reconnecting any pre-existing objects discovered this way into `_id_map`. Then, for every object this plugin has ever seen (created this session or discovered by reading) that is dirty (or during initial write), updates its attributes and, for datasets, resizes them if needed and — unless `no_data` is set — writes pending values via `updateDatasetValues`. Records `self._flush_time` and clears `self._init` at the end. Always returns `True` on success.

### H5pyPlugin.open()

Opens (or creates) the underlying HDF5 file with h5py, in `'r'` mode if `read_only`, `'a'` (append) if `append`, or `'w'` (overwrite) otherwise; a subsequent `open()` call after a non-read-only open switches to append mode. Establishes the root object id, reusing `self.db.root_id` if set or minting a new one otherwise. Raises `ValueError` if no db has been attached yet. A no-op (returning the existing root id) if the file is already open.

### H5pyPlugin.close()

Closes the underlying h5py file handle and clears `_id_map`/`_addr_map`. Does not flush — `Hdf5db.close()` (the only caller) always calls `Hdf5db.flush()` immediately beforehand, which itself calls this plugin's `flush()`. A no-op if the file is not open.

### H5pyPlugin.isClosed()

Returns `True` if there is no open h5py file handle, `False` otherwise.

### H5pyPlugin.getStats()

Returns a dict with `created`, `lastModified`, and `owner` keys derived from `os.stat()` on the underlying file path (`st_ctime`, `st_mtime`, `st_uid` respectively). `owner` is the raw numeric uid, not resolved to a username.

### H5pyPlugin.getFilters(compressors_only=False)

Returns a tuple naming the HDF5 filters h5py supports for dataset creation. Always includes `H5Z_FILTER_DEFLATE`; unless `compressors_only` is true, also includes `H5Z_FILTER_SHUFFLE`, `H5Z_FILTER_FLETCHER32`, `H5Z_FILTER_SZIP`, `H5Z_FILTER_NBIT`, and `H5Z_FILTER_SCALEOFFSET`.
