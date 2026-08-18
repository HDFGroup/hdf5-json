# h5reader

This module defines `H5Reader`, the abstract base class that `Hdf5db` delegates all read I/O to, and `H5NullReader`, a trivial no-op implementation used as a default/placeholder reader. Concrete, storage-backed implementations are `h5pystore.h5py_reader.H5PyReader` (reads real `.h5` files via h5py) and `jsonstore.h5json_reader.H5JsonReader` (reads h5json `.json` files); both subclass `H5Reader` and are documented separately.

## H5Reader

Abstract base class (`abc.ABC`) declaring the interface `Hdf5db` uses to read objects, attributes, and dataset values from a storage medium. Subclasses must implement the abstract methods below; `queryDataset` has a default (non-abstract) implementation that simply signals it isn't supported.

### H5Reader.set_db(db)

Stores a weak reference to the owning `Hdf5db` instance (via `weakref.ref`), later exposed through the `db` property. Called by `Hdf5db` when it attaches this reader.

### H5Reader.db

Property returning the `Hdf5db` instance previously registered via `set_db()` (dereferencing the weak reference). Raises `ValueError` if no db reference has been set.

### H5Reader.filepath

Property returning the filepath the reader was constructed with.

### H5Reader.closed

Property returning `True` if the reader handle is closed or was never opened; implemented by calling `isClosed()`.

### H5Reader.get_root_id()

Abstract method. Subclasses must return the id of the root group of the storage medium.

### H5Reader.getObjectById(obj_id, include_attrs=True, include_links=True)

Abstract method. Subclasses must return the JSON representation of the object (group, dataset, or committed datatype) identified by `obj_id`, optionally including its attributes and/or links.

### H5Reader.getAttribute(obj_id, name, includeData=True)

Abstract method. Subclasses must return the JSON representation of the named attribute on the object identified by `obj_id`, optionally including its value data.

### H5Reader.getDatasetValues(obj_id, sel, dtype=None, query=None)

Abstract method. Subclasses must return the values of the dataset identified by `obj_id`, restricted to the given selection `sel` (a slices list/tuple with one element per dimension of the dataset) and optionally cast/interpreted according to `dtype`, and optionally filtered by a `query` expression.

### H5Reader.queryDataset(obj_id, query, sel=None, limit=0)

Query the dataset identified by `obj_id` using a query expression and an optional selection, intended to return a numpy array of indices of matching elements. Not required to be implemented by subclasses — the base implementation always raises `NotImplementedError`, in which case `Hdf5db` falls back to evaluating the query itself against values obtained via `getDatasetValues`. Backends should override this only when they can evaluate the query more efficiently than that fallback (e.g. by pushing it down into storage).

### H5Reader.open()

Abstract method. Subclasses must open the data source for reading.

### H5Reader.close()

Abstract method. Subclasses must close any open handles to the storage.

### H5Reader.isClosed()

Abstract method. Subclasses must return `True` if the reader's storage handle is closed.

### H5Reader.getStats()

Abstract method. Subclasses must return a dict with at least the keys `'created'` (creation time), `'lastModified'` (modification time), and `'owner'` (owner name).

## H5NullReader

A no-op `H5Reader` implementation used by `Hdf5db` as a default reader when no real storage backend is attached (e.g. for building an in-memory object graph from scratch). It returns/creates a bare root group with no links or attributes, and never actually reads from any external medium.

### H5NullReader.get_root_id()

Returns the reader's root id, which was assigned during `open()` — either taken from the associated `Hdf5db.root_id` if one already exists, or a freshly generated group id otherwise.

### H5NullReader.getObjectById(obj_id, include_attrs=True, include_links=True)

Returns a minimal group JSON object (`{"links": {}, "attributes": {}, "cpl": {}, "created": <time>}`) if `obj_id` matches the reader's root id; raises `KeyError` for any other id, since this reader has no other objects to return.

### H5NullReader.getAttribute(obj_id, name, includeData=True)

Always returns `None` — this reader has no attributes to return.

### H5NullReader.getDatasetValues(obj_id, sel=None, dtype=None)

Always returns `None` — this reader has no dataset data to return. Note its signature omits the `query` parameter present in the base class's abstract signature.

### H5NullReader.open()

Opens the reader: if not already open, assigns `_root_id` from the associated `Hdf5db.root_id` if set, otherwise generates a new group id via `createObjId`. Raises `ValueError` if no `Hdf5db` has been registered via `set_db()`. Returns the root id.

### H5NullReader.close()

Marks the reader as closed. Performs no other action since there is no real handle to release.

### H5NullReader.isClosed()

Returns whether the reader is currently marked closed.

### H5NullReader.getStats()

Returns a stats dict with `created`, `lastModified` set to `0` and `owner` set to `""`.
