# storage_plugin

This module defines `StoragePlugin`, the abstract base class that `Hdf5db` delegates all storage I/O (both reads and writes) to, and `NullPlugin`, a trivial no-op implementation used as a default/placeholder plugin. A single plugin instance serves as both the reader and the writer for a given `Hdf5db` — there is no separate reader object with its own view of the store, so a read always reflects whatever that same plugin instance has most recently flushed. Concrete, storage-backed implementations are `h5pystore.h5py_plugin.H5pyPlugin` (reads/writes real `.h5` files via h5py) and `jsonstore.h5json_plugin.H5JsonPlugin` (reads/writes h5json `.json` files); both subclass `StoragePlugin` and are documented separately.

## StoragePlugin

Abstract base class (`abc.ABC`) declaring the interface `Hdf5db` uses to read and write objects, attributes, dataset values, and compression filter support, to/from a storage medium. Constructed with `filepath`, `append` (preserve existing state vs. reset), `no_data` (write/report structure and metadata only, without dataset values), `read_only`, and an optional `app_logger`. Subclasses must implement the abstract methods below; `queryDataset` has a default (non-abstract) implementation that simply signals it isn't supported.

### StoragePlugin.set_db(db)

Stores a weak reference to the owning `Hdf5db` instance (via `weakref.ref`), later exposed through the `db` property. Called by `Hdf5db` when it attaches this plugin.

### StoragePlugin.db

Property returning the `Hdf5db` instance previously registered via `set_db()` (dereferencing the weak reference). Raises `ValueError` if no db reference has been set.

### StoragePlugin.filepath

Property returning the filepath the plugin was constructed with.

### StoragePlugin.closed

Property returning `True` if the plugin's storage handle is closed or was never opened; implemented by calling `isClosed()`.

### StoragePlugin.lastModified

Property returning the last-modified timestamp recorded for the storage medium (`None` until set by the implementation).

### StoragePlugin.append

Property returning whether the plugin was constructed in append mode.

### StoragePlugin.no_data

Property returning whether the plugin was constructed with `no_data` set (i.e. write/report structure and metadata only, without dataset values).

### StoragePlugin.read_only

Property returning whether the plugin was constructed with `read_only` set. This is distinct from `append`: it guarantees the plugin will never write to its storage at all, regardless of what the owning `Hdf5db` does — useful for a source db in a format-conversion tool, which should never risk modifying its input. Concrete plugins make `flush()` a safe no-op under this flag (only raising if there's actually something pending to write) and have `open()` use the least-privileged access mode the backend supports (e.g. read-only file mode).

### StoragePlugin.get_root_id()

Abstract method. Subclasses must return the id of the root group of the storage medium.

### StoragePlugin.getObjectById(obj_id, include_attrs=True, include_links=True)

Abstract method. Subclasses must return the JSON representation of the object (group, dataset, or committed datatype) identified by `obj_id`, optionally including its attributes and/or links.

### StoragePlugin.getAttribute(obj_id, name, includeData=True)

Abstract method. Subclasses must return the JSON representation of the named attribute on the object identified by `obj_id`, optionally including its value data.

### StoragePlugin.getDatasetValues(obj_id, sel, dtype=None, query=None)

Abstract method. Subclasses must return the values of the dataset identified by `obj_id`, restricted to the given selection `sel` (a slices list/tuple with one element per dimension of the dataset) and optionally cast/interpreted according to `dtype`, and optionally filtered by a `query` expression.

### StoragePlugin.queryDataset(obj_id, query, sel=None, limit=0, update_value=None)

Query the dataset identified by `obj_id` using a query expression and an optional selection, and (per `update_value`) replace matching elements; intended to return a numpy array of indices of matching elements. Not required to be implemented by subclasses — the base implementation always raises `NotImplementedError`, in which case `Hdf5db` falls back to evaluating the query itself against values obtained via `getDatasetValues`. Backends should override this only when they can evaluate/update the query more efficiently than that fallback (e.g. by pushing it down into storage).

### StoragePlugin.open()

Abstract method. Subclasses must open the storage handle for reading/writing and return the root id.

### StoragePlugin.flush()

Abstract method. Subclasses must write any dirty (new/modified) items to storage.

### StoragePlugin.close()

Abstract method. Subclasses must close any open handles to the storage.

### StoragePlugin.isClosed()

Abstract method. Subclasses must return `True` if the plugin's storage handle is closed.

### StoragePlugin.getStats()

Abstract method. Subclasses must return a dict with at least the keys `'created'` (creation time), `'lastModified'` (modification time), and `'owner'` (owner name).

### StoragePlugin.getFilters(compressors_only=False)

Abstract method. Subclasses must return a list of compression/filter identifiers supported by the plugin's storage backend; `compressors_only`, if `True`, restricts the list to compression filters.

## NullPlugin

A no-op `StoragePlugin` implementation used by `Hdf5db` as a default plugin when no real storage backend is attached (e.g. for building an in-memory object graph from scratch). It returns/creates a bare root group with no links or attributes, and never actually reads from or writes to any external medium. Its constructor does not accept a `read_only` flag (it is always constructed with `read_only=False`, though this has no practical effect since it never writes regardless).

### NullPlugin.get_root_id()

Returns the plugin's root id, which was assigned during `open()` — either taken from the associated `Hdf5db.root_id` if one already exists, or a freshly generated group id otherwise.

### NullPlugin.getObjectById(obj_id, include_attrs=True, include_links=True)

Returns a minimal group JSON object (`{"links": {}, "attributes": {}, "cpl": {}, "created": <time>}`) if `obj_id` matches the plugin's root id; raises `KeyError` for any other id, since this plugin has no other objects to return.

### NullPlugin.getAttribute(obj_id, name, includeData=True)

Always returns `None` — this plugin has no attributes to return.

### NullPlugin.getDatasetValues(obj_id, sel=None, dtype=None, query=None)

Always returns `None` — this plugin has no dataset data to return.

### NullPlugin.open()

Opens the plugin: if not already open, assigns `_root_id` from the associated `Hdf5db.root_id` if set, otherwise generates a new group id via `createObjId`. Raises `ValueError` if no `Hdf5db` has been registered via `set_db()`. Returns the root id.

### NullPlugin.flush()

Logs a debug message and always returns `False`, since this plugin is unable to actually persist anything.

### NullPlugin.close()

Marks the plugin as closed. Performs no other action since there is no real handle to release.

### NullPlugin.isClosed()

Returns whether the plugin is currently marked closed.

### NullPlugin.getStats()

Returns a stats dict with `created`, `lastModified` set to `0` and `owner` set to `""`.

### NullPlugin.getFilters(compressors_only=False)

Always returns an empty tuple — this plugin supports no compression filters.
