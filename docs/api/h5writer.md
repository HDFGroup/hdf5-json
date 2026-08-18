# h5writer

This module defines `H5Writer`, the abstract base class that `Hdf5db` delegates all write I/O (flushing dirty objects/attributes/dataset data) to, and `H5NullWriter`, a trivial no-op implementation used as a default/placeholder writer that cannot actually persist anything. Concrete, storage-backed implementations are `h5pystore.h5py_writer.H5PyWriter` (writes real `.h5` files via h5py) and `jsonstore.h5json_writer.H5JsonWriter` (writes h5json `.json` files); both subclass `H5Writer` and are documented separately.

## H5Writer

Abstract base class (`abc.ABC`) declaring the interface `Hdf5db` uses to write objects, attributes, and dataset values to a storage medium, and to query the writer for compression filter support. Subclasses must implement the abstract methods below; `queryDataset` has a default (non-abstract) implementation that simply signals it isn't supported.

### H5Writer.set_db(db)

Stores a weak reference to the owning `Hdf5db` instance (via `weakref.ref`), later exposed through the `db` property. Called by `Hdf5db` when it attaches this writer.

### H5Writer.filepath

Property returning the filepath the writer was constructed with.

### H5Writer.closed

Property returning `True` if the writer handle is closed or was never opened; implemented by calling `isClosed()`.

### H5Writer.lastModified

Property returning the last-modified timestamp recorded for the storage medium (`None` until set by the implementation).

### H5Writer.db

Property returning the `Hdf5db` instance previously registered via `set_db()` (dereferencing the weak reference), or `None` (logged at debug level) if no db reference has been set — unlike the equivalent `H5Reader.db` property, this does not raise.

### H5Writer.append

Property returning whether the writer was constructed in append mode.

### H5Writer.no_data

Property returning whether the writer was constructed with `no_data` set (i.e. write structure/metadata only, without dataset values).

### H5Writer.queryDataset(obj_id, query, sel=None, limit=0, update_value=None)

Query the dataset identified by `obj_id` using a query expression and optional selection, and (per `update_value`) replace matching elements; intended to return a numpy array of indices of matching elements. Not required to be implemented by subclasses — the base implementation always raises `NotImplementedError`, in which case `Hdf5db` falls back to evaluating the query itself against values obtained via `getDatasetValues`. Backends should override this only when they can evaluate/update the query more efficiently than that fallback (e.g. by pushing it down into storage).

### H5Writer.open()

Abstract method. Subclasses must open the storage handle for writing and return the root id.

### H5Writer.flush()

Abstract method. Subclasses must write any dirty (new/modified) items to storage. The base docstring/comment notes a conceptual default of returning `False` to indicate nothing could be persisted, though as an abstract method each concrete subclass supplies its own implementation.

### H5Writer.close()

Abstract method. Subclasses must close any open handles to the storage.

### H5Writer.isClosed()

Abstract method. Subclasses must return `True` if the writer's storage handle is closed.

### H5Writer.getStats()

Abstract method. Subclasses must return a dict with at least the keys `'created'` (creation time), `'lastModified'` (modification time), and `'owner'` (owner name).

### H5Writer.getFilters(compressors_only=False)

Abstract method. Subclasses must return a list of compression/filter identifiers supported by the writer's storage backend; `compressors_only`, if `True`, restricts the list to compression filters.

## H5NullWriter

A no-op `H5Writer` implementation used by `Hdf5db` as a default writer when no real storage backend is attached. It can open (assigning/reusing a root id) and close a handle and track closed state, but its `flush()` never actually persists anything, and it does not support `append` mode (raises `IOError` if constructed with `append=True`).

### H5NullWriter.open()

Opens the writer: if already open, returns the existing root id unchanged. Otherwise, requires an `Hdf5db` to have been registered via `set_db()` (raises `ValueError` if not), then assigns `_root_id` from the associated `Hdf5db.root_id` if set, otherwise generates a new group id via `createObjId`. Returns the root id.

### H5NullWriter.flush()

Logs a debug message and always returns `False`, since this writer is unable to actually persist anything.

### H5NullWriter.close()

Marks the writer as closed. Performs no other action since there is no real handle to release.

### H5NullWriter.isClosed()

Returns whether the writer is currently marked closed.

### H5NullWriter.getStats()

Returns a stats dict with `created`, `lastModified` set to `0` and `owner` set to `""`.

### H5NullWriter.getFilters(compressors_only=False)

Always returns an empty tuple — this writer supports no compression filters.
