# h5py_writer

Implements the `H5Writer` interface (`h5json.h5writer.H5Writer`) for real HDF5 `.h5` files, using h5py to create groups, datasets, attributes, datatypes, and links from the in-memory h5json model maintained by `Hdf5db`. Three non-obvious areas are handled at the low level rather than through h5py's high-level API: building dataspace selections for HDF5 region references (h5py's high-level slicing supports only a single fancy-index array and cannot express a paired-coordinate point selection at all, so `h5s`/`h5r` low-level calls are used directly); converting between h5json's own tagged numpy dtypes (carrying `ref`/`vlen` metadata) and h5py's special dtype representations, since h5py's type layer identity-checks special dtype metadata against its own `Reference`/`RegionReference` classes; and writing object-dtype (vlen/reference) arrays via `write_direct`/`attrs.create` rather than plain assignment, to avoid numpy silently homogenizing same-length object-array elements into a regular N-d array during a high-level write (which then makes h5py reject the resulting shape).

## H5pyWriter

Concrete `H5Writer` implementation backed by an h5py `File` handle opened for writing (or appending). Maintains `_id_map`, mapping h5json object ids to the HDF5 path at which each has been created, and an `_init` flag that is `True` until the first `flush()` completes, indicating that every object needs its values (not just deltas) written.

### H5pyWriter.resizeDataset(dset_id, dset)

Resizes the live h5py dataset `dset` to match the current dims recorded for `dset_id` in the db (via `dset.resize(new_dims)`), used when a dataset's extent has grown/shrunk since it was created.

### H5pyWriter.updateDatasetValues(dset_id, dset)

Applies pending (dirty) writes for `dset_id`, fetched from `self.db._getDatasetUpdates(dset_id)` as a list of `(selection, value)` pairs, onto the live h5py dataset `dset`. Dispatches on selection type: `H5S_SEL_NONE` is a no-op; `H5S_SEL_ALL` writes the whole array via `_writeDatasetFull`; `H5S_SEL_HYPERSLABS` builds a tuple of Python `slice` objects from the selection's start/count/step and assigns via `dset[slices] = val`; `H5S_SEL_POINTS` writes one point at a time in a loop; `H5S_SEL_FANCY` with more than one coordinate-list dimension decomposes into one indexed write per paired coordinate (h5py can only handle a single fancy-index array per assignment), otherwise assigns directly. Raises `TypeError` for an unrecognized selection type.

### H5pyWriter.initializeDatasetValues(dset_id, dset)

Writes the entire initial contents of a newly created dataset. Returns immediately for a null-space dataset (`dset.shape is None`). Otherwise builds an all-selection with `selections.select(dset.shape, ...)`, fetches the full array from `self.db.getDatasetValues`, converts it to an h5py-compatible array via `_copy_array`, and writes it with `_writeDatasetFull`.

### H5pyWriter.createAttribute(obj, name, attr_json)

Creates attribute `name` on the live h5py object `obj` from its h5json representation `attr_json`. Handles a null-space attribute specially, assigning `h5py.Empty(...)` with the converted dtype. For scalar/simple shapes, builds the source numpy array with `jsonToArray` and converts it to an h5py-compatible array with `_copy_array`, then writes it via `obj.attrs.create(name, data=tgt_arr, dtype=tgt_arr.dtype)` rather than plain `obj.attrs[name] = tgt_arr` — passing the dtype explicitly preserves the actual object dtype (e.g. vlen-of-reference) instead of letting h5py re-infer a possibly-wrong dtype from the array's runtime contents.

### H5pyWriter.updateAttributes(obj_id, obj)

Walks the `"attributes"` dict on `obj_id`'s current JSON (from `self.db.getObjectById`) and, for each attribute: deletes it from the live h5py object `obj` if marked `"DELETED"` (ignoring the case where it's already absent); skips it if it was already flushed (its `"created"` timestamp predates `self._flush_time`); otherwise creates/overwrites it via `createAttribute`. Does nothing if the object has no `"attributes"` key.

### H5pyWriter.flush()

Writes all pending (dirty) state to the HDF5 file. Returns `False` without writing if the writer is closed (`self.closed`); raises `IOError` if the file handle isn't open. On the first flush (or whenever `self.db.new_objects` is set), walks the root group's links and creates any new objects via the internal `_createObjects` recursion. Then, for every object tracked in `_id_map` that is dirty (or during initial write), updates its attributes and, for datasets, resizes them if needed (`self.db.is_resized`) and — unless `self.no_data` is set — writes values via `initializeDatasetValues` (first flush) or `updateDatasetValues` (subsequent flushes). Records `self._flush_time` and clears `self._init` at the end. Always returns `True` on success.

### H5pyWriter.open()

Opens (or creates) the underlying HDF5 file with h5py in `'a'` (append) or `'w'` (overwrite) mode depending on the writer's `append` setting, then switches `_append` to `True` so a subsequent `open()` call would append rather than overwrite. Establishes the root object id, reusing `self.db.root_id` if set or minting a new one otherwise. Raises `ValueError` if no db has been attached yet.

### H5pyWriter.close()

Flushes any pending writes (via `flush()`) and closes the underlying h5py file handle. A no-op if the file is not open.

### H5pyWriter.isClosed()

Returns `True` if there is no open h5py file handle, `False` otherwise.

### H5pyWriter.getStats()

Returns a dict with `created`, `lastModified`, and `owner` keys derived from `os.stat()` on the underlying file path (`st_ctime`, `st_mtime`, `st_uid` respectively). `owner` is the raw numeric uid, not resolved to a username.

### H5pyWriter.getFilters(compressors_only=False)

Returns a tuple naming the HDF5 filters h5py supports for dataset creation. Always includes `H5Z_FILTER_DEFLATE`; unless `compressors_only` is true, also includes `H5Z_FILTER_SHUFFLE`, `H5Z_FILTER_FLETCHER32`, `H5Z_FILTER_SZIP`, `H5Z_FILTER_NBIT`, and `H5Z_FILTER_SCALEOFFSET`.
