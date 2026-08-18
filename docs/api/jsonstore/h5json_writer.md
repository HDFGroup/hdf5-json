# h5json_writer

Implements the `H5Writer` interface (`h5json.h5writer.H5Writer`) for h5json's own `.json` file format, by dumping the in-memory model held by `Hdf5db` back out as a single h5json JSON document. The `dump*` methods each build the JSON dict for one kind of object (group, dataset, datatype, attribute, link), mirroring the on-disk schema — for example propagating `"encoding": "base64"` onto a dataset's or attribute's dict whenever its type is `H5T_OPAQUE`. Unlike the h5py-backed writer, there is no incremental/dirty-tracking write path: the entire document is (re)built and written out in one pass by `dumpFile`, which `flush()` calls at most once per writer instance.

## H5JsonWriter

Concrete `H5Writer` implementation that accumulates the output document in `self.json` and writes it to `self.filepath` (or prints it) when flushed. Does not support append mode — the constructor raises `ValueError` if `append=True`. An optional `data_limit` caps how large a dataset's element data can be (in bytes, estimated) before its `"value"` is omitted from the dump; passing `data_limit=0` is equivalent to `no_data=True` on the base class.

### H5JsonWriter.flush()

Writes the JSON document if it hasn't been written yet. Raises `IOError` if called before `open()` (i.e. before `self._root_id` is set). If the file has already been dumped in this writer's lifetime (`self._file_dumped`), this is a no-op; otherwise calls `dumpFile()` and marks the file as dumped. Always returns `True` when it doesn't raise.

### H5JsonWriter.open()

Establishes the root object id — reusing `self.db.root_id` if the `Hdf5db` already has one, otherwise minting a new one via `createObjId` — and returns it. Since this writer has no incremental-update path, this just records the root id rather than opening any file handle.

### H5JsonWriter.close()

Calls `flush()` to ensure the document has been written, then clears `self._root_id`.

### H5JsonWriter.isClosed()

Returns `True` if `self._root_id` is not set (i.e. `open()` hasn't been called, or `close()` has), `False` otherwise.

### H5JsonWriter.getAliasList(obj_id)

Returns the list of HDF5 paths (aliases) at which `obj_id` is reachable, delegating to `self.db.getPathsForObjectId(obj_id)`.

### H5JsonWriter.dumpAttribute(obj_id, attr_name)

Builds the JSON dict for one attribute, fetched via `self.db.getAttribute(obj_id, attr_name)`. Includes `"name"`, `"type"`, and `"shape"` always; includes `"value"` (and `"encoding"` if present) when the attribute has a value, logging a warning if it doesn't.

### H5JsonWriter.dumpAttributes(obj_id)

Returns a list of attribute dicts (via `dumpAttribute`) for every attribute on `obj_id`, as reported by `self.db.getAttributes(obj_id)`.

### H5JsonWriter.dumpLink(obj_id, name)

Builds the JSON dict for one link named `name` on `obj_id`, fetched via `self.db.getLink(obj_id, name)`. Always includes `"class"`; for a hard link (one with an `"id"` key), adds `"collection"` (derived from the target id via `getCollectionForId`) and `"id"` (the target's bare uuid via `getUuidFromId`). Copies through any other keys on the source item except `id`, `created`, and `modified`, and sets `"title"` to `name`.

### H5JsonWriter.dumpLinks(obj_id)

Returns a list of link dicts (via `dumpLink`) for every link on `obj_id`, as reported by `self.db.getLinks(obj_id)`.

### H5JsonWriter.dumpGroup(obj_id)

Builds the JSON dict for a group: `"alias"` (via `getAliasList`), `"attributes"` (via `dumpAttributes`, included only if non-empty), and `"links"` (via `dumpLinks`, included only if non-empty). Note the `"cpl"` handling here mutates the source `item` dict in place (renaming it to `"creationProperties"` on that dict) but that key is never copied into `response`, so creation properties are not actually included in a group's dumped output.

### H5JsonWriter.dumpGroups()

Builds the `"groups"` collection for the whole document: dumps the root group plus every other group returned by `self.db.getCollection("groups")` (skipping the root if it reappears in that list), keyed by each group's bare uuid, and stores the result at `self.json["groups"]`.

### H5JsonWriter.dumpDataset(obj_id)

Builds the JSON dict for a dataset: `"alias"`, `"type"`, and a `"shape"` dict (`class`, and `dims`/`maxdims` as applicable — `maxdims` entries of `0` are rewritten to the string `"H5S_UNLIMITED"`). Includes `"creationProperties"` from the source `"cpl"` key if present, and `"attributes"` if non-empty. If `self._data_limit` is set, estimates the dataset's total byte size (`getItemSize(type_item)` times element count, treating a variable-length item size as 1024 bytes for the estimate) and skips writing `"value"` (logging that it's being skipped) when the estimate exceeds the limit; otherwise — or when no limit is configured — reads the full dataset value via `self.db.getDatasetValues` with an all-selection and serializes it with `bytesArrayToList` into `"value"`, adding `"encoding": "base64"` when the type is `H5T_OPAQUE`. A null-space dataset (`num_elements == 0`) never gets a `"value"` key.

### H5JsonWriter.dumpDatasets()

Builds the `"datasets"` collection for the whole document from every id returned by `self.db.getCollection("datasets")`, keyed by bare uuid, via `dumpDataset`; stores the result at `self.json["datasets"]` only if there is at least one dataset.

### H5JsonWriter.dumpDatatype(obj_id)

Builds the JSON dict for a committed datatype: `"alias"`, `"type"`, `"creationProperties"` (from `"cpl"` if present), and `"attributes"` (if non-empty).

### H5JsonWriter.dumpDatatypes()

Builds the `"datatypes"` collection for the whole document from every id returned by `self.db.getCollection("datatypes")`, keyed by bare uuid, via `dumpDatatype`; stores the result at `self.json["datatypes"]` only if there is at least one committed datatype.

### H5JsonWriter.dumpFile()

Top-level entry point that assembles the complete h5json document: resolves the root object's uuid via `self.db.getObjectIdByPath("/")`, sets `"apiVersion"` from `self.db.getVersionInfo()["hdf5-json-version"]` and `"root"` to the root's bare uuid, then calls `dumpGroups()`, `dumpDatasets()`, and `dumpDatatypes()` in turn to populate the rest of `self.json`. If `self._filepath` is set, writes the document to that path as indented JSON (`json.dump`, ASCII-escaped, indent from `self._indent`); otherwise prints the document (sorted keys) to stdout. Updates `self._lastModified` to the current time when done.

### H5JsonWriter.getStats()

Returns a dict with `created`, `lastModified`, and `owner` keys derived from `os.stat()` on the underlying file path (`st_ctime`, `st_mtime`, `st_uid` respectively). `owner` is the raw numeric uid, not resolved to a username.

### H5JsonWriter.getFilters(compressors_only=False)

Always returns an empty tuple — the JSON store format has no notion of HDF5 compression filters.
