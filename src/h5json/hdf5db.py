##############################################################################
# Copyright by The HDF Group.                                                #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of H5Serv (HDF5 REST Server) Service, Libraries and      #
# Utilities.  The full HDF5 REST Server copyright notice, including          #
# terms governing use, modification, and redistribution, is contained in     #
# the file COPYING, which can be found at the root of the source code        #
# distribution tree.  If you do not have access to this file, you may        #
# request a copy from help@hdfgroup.org.                                     #
##############################################################################

import numpy as np
import logging
from .hdf5dtype import getTypeItem, createDataType, Reference, special_dtype, isOpaqueDtype
from .hdf5dtype import numpy_integer_types, numpy_float_types
from .hdf5dtype import RegionReference, is_reference, is_regionreference, has_reference
from .array_util import jsonToArray, bytesArrayToList
from .query_util import arrayQuery
from .dset_util import resize_dataset, getDatasetLayoutClass, getChunkDims
from .shape_util import getShapeClass, getShapeDims, getShapeJson
from .filters import validateFilters, FILTER_DEFS
from .objid import createObjId, getCollectionForId, isValidUuid, getUuidFromId, getHashTagForId
from . import selections
from .time_util import getNow
from .apiversion import _apiver
from .storage_plugin import StoragePlugin, NullPlugin

# Default auto-flush thresholds (see Hdf5db.__init__) - pending dataset value
# updates are held in memory and normally only written out when flush()/close()
# is called explicitly.  These defaults bound how much can accumulate before
# an automatic flush is triggered, similar to a disk cache's write-back policy.
DEFAULT_AUTO_FLUSH_MEMORY = 128 * 1024 * 1024  # 128 MiB
DEFAULT_AUTO_FLUSH_INTERVAL = 30  # seconds


def _query_rel_to_abs(x_sel, rel_indices, rank):
    """Map arrayQuery relative indices (within a sub-array) to absolute dataset indices.

    x_sel: SimpleSelection whose sub-array was queried
    rel_indices: arrayQuery result, ndarray of shape (N, rank)
    """
    slices = x_sel.slices
    if len(rel_indices) == 0:
        return np.zeros((0, rank), dtype='int64')
    abs_result = np.zeros((len(rel_indices), rank), dtype='int64')
    for d in range(rank):
        s = slices[d]
        if isinstance(s, slice):
            start = s.start if s.start is not None else 0
            step = s.step if s.step is not None else 1
            abs_result[:, d] = rel_indices[:, d] * step + start
        elif isinstance(s, list):
            abs_result[:, d] = [s[i] for i in rel_indices[:, d].tolist()]
        else:
            abs_result[:, d] = int(s)
    return abs_result


def _withFilterName(filter_json):
    """ Return filter_json unchanged if it already has a "name" key, otherwise a copy with
    "name" filled in from FILTER_DEFS based on its "class" (if recognized). """
    if not isinstance(filter_json, dict) or "name" in filter_json or "class" not in filter_json:
        return filter_json
    for filter_class, _, filter_name, _ in FILTER_DEFS:
        if filter_class == filter_json["class"]:
            filter_json = dict(filter_json)
            filter_json["name"] = filter_name
            break
    return filter_json


def _dtypesStructurallyEqual(dt1, dt2):
    """ Compare two dtypes by field names/types/shape rather than exact byte layout.

    A dataset's expected dtype (built from its h5json type descriptor via createDataType(),
    which always produces a packed, no-padding layout) can otherwise fail a strict numpy
    dtype equality check against an array read from an actual HDF5 file, whose compound
    types may carry C-struct alignment padding between fields - same logical type, different
    field offsets/itemsize. """
    if dt1.names is not None or dt2.names is not None:
        if dt1.names is None or dt2.names is None or dt1.names != dt2.names:
            return False
        return all(
            _dtypesStructurallyEqual(dt1.fields[name][0], dt2.fields[name][0])
            for name in dt1.names
        )
    if dt1.subdtype is not None or dt2.subdtype is not None:
        if dt1.subdtype is None or dt2.subdtype is None:
            return False
        base1, shape1 = dt1.subdtype
        base2, shape2 = dt2.subdtype
        return shape1 == shape2 and _dtypesStructurallyEqual(base1, base2)
    return dt1 == dt2


def _decode(item, encoding="ascii"):
    """
    decode any byte items to python 3 strings
    """
    ret_val = None
    if type(item) is bytes:
        ret_val = item.decode(encoding)
    elif type(item) is list:
        ret_val = []
        for x in item:
            ret_val.append(_decode(x, encoding))
    elif type(item) is tuple:
        ret_val = []
        for x in item:
            ret_val.append(_decode(x, encoding))
        ret_val = tuple(ret_val)
    elif type(item) is dict:
        ret_val = {}
        for k in dict:
            ret_val[k] = _decode(item[k], encoding)
    elif type(item) is np.ndarray:
        x = item.tolist()
        ret_val = []
        for x in item:
            ret_val.append(_decode(x, encoding))
    elif type(item) in numpy_integer_types:
        ret_val = int(item)
    elif type(item) in numpy_float_types:
        ret_val = float(item)
    else:
        ret_val = item
    return ret_val


class ChunkIterator:
    """
    Iterate through the chunks of a dataset, yielding the chunk's data as an
    ndarray on each iteration. This lets a caller read through a large,
    chunked dataset one chunk at a time without loading the whole dataset
    into memory.

    Modeled on h5py's chunk iterator (h5py.Dataset.iter_chunks() /
    h5py._hl.dataset.ChunkIterator), but each chunk's data is fetched via
    Hdf5db.getDatasetValues() rather than by slicing an h5py.Dataset, so it
    works uniformly across storage backends and picks up any not-yet-flushed
    in-memory updates.

    Use Hdf5db.getChunkIterator() rather than constructing this directly.
    """

    def __init__(self, db, dset_id, sel=None):
        dset_json = db.getObjectById(dset_id)
        shape_json = dset_json["shape"]
        dims = getShapeDims(shape_json)
        rank = len(dims)
        if rank == 0:
            raise ValueError("ChunkIterator can't be used with scalar datasets")

        if sel is None:
            sel = selections.select(dims, ...)
        if not isinstance(sel, selections.Selection):
            raise TypeError("Expected Selection class")
        if sel.shape != dims:
            raise TypeError("Selection shape does not match dataset shape")
        if sel.select_type not in (selections.H5S_SEL_ALL, selections.H5S_SEL_HYPERSLABS):
            raise ValueError("ChunkIterator only supports hyperslab selections")

        self._db = db
        self._dset_id = dset_id
        self._shape = dims
        self._layout = getChunkDims(dset_json)

        sel_slices = []
        for s in sel.slices:
            if s.step not in (None, 1):
                raise ValueError("ChunkIterator does not support stepped selections")
            sel_slices.append(slice(s.start, s.stop, 1))
        self._sel = tuple(sel_slices)

        # a 0-sized dimension (e.g. a not-yet-extended unlimited dataset) means
        # the selection is legitimately empty - nothing to validate or iterate
        self._empty = any(d == 0 for d in self._shape)

        self._chunk_index = []
        if not self._empty:
            for dim in range(rank):
                s = self._sel[dim]
                if s.start < 0 or s.stop > self._shape[dim] or s.stop <= s.start:
                    raise ValueError("Invalid selection - selection region must be within dataset space")
                self._chunk_index.append(s.start // self._layout[dim])

        self._current_sel = None

    @property
    def sel(self):
        """ Selection (within the full dataset) of the chunk most recently returned by __next__ """
        return self._current_sel

    def __iter__(self):
        return self

    def __next__(self):
        if self._empty:
            raise StopIteration()
        rank = len(self._shape)
        if self._chunk_index[0] * self._layout[0] >= self._sel[0].stop:
            # ran past the last chunk, end iteration
            raise StopIteration()

        slices = []
        for dim in range(rank):
            s = self._sel[dim]
            start = self._chunk_index[dim] * self._layout[dim]
            stop = (self._chunk_index[dim] + 1) * self._layout[dim]
            # adjust the start if this is an edge chunk
            if start < s.start:
                start = s.start
            if stop > s.stop:
                stop = s.stop  # trim to end of the selection
            slices.append(slice(start, stop, 1))
        slices = tuple(slices)

        # bump up the last index and carry forward if we run outside the selection
        dim = rank - 1
        while dim >= 0:
            s = self._sel[dim]
            self._chunk_index[dim] += 1

            chunk_end = self._chunk_index[dim] * self._layout[dim]
            if chunk_end < s.stop:
                # we still have room to extend along this dimension
                break

            if dim > 0:
                # reset to the start and continue iterating with higher dimension
                self._chunk_index[dim] = s.start // self._layout[dim]
            dim -= 1

        self._current_sel = selections.select(self._shape, slices)
        return self._db.getDatasetValues(self._dset_id, self._current_sel)


class Hdf5db:
    """
    This class is used to manage id lookup tables for primary HDF objects (Groups, Datasets,
    and Datatypes).  By default all data is held in-memory.  Initialize with a StoragePlugin
    (or set the `plugin` property later) to read from and write to a storage medium - the same
    plugin instance is used for both, so a read always reflects whatever that plugin has most
    recently flushed.
    """

    @staticmethod
    def getVersionInfo():
        versionInfo = {}
        versionInfo["hdf5-json-version"] = _apiver
        return versionInfo

    def __init__(
        self,
        plugin: StoragePlugin = None,
        app_logger=None,
        auto_flush_memory=DEFAULT_AUTO_FLUSH_MEMORY,
        auto_flush_interval=DEFAULT_AUTO_FLUSH_INTERVAL,
    ):
        if app_logger:
            self.log = app_logger
        else:
            self.log = logging.getLogger()

        self._db = {}

        self._new_objects = set()       # set of for newly created objects
        self._dirty_objects = set()     # set of modified objects
        self._deleted_objects = set()   # set of deleted objects
        self._resized_datasets = set()  # set of dataset ids that have been resized
        self._dataset_updates = {}      # list of dataset values updates keyed by dset_id

        # auto-flush policy: pending changes are flushed automatically once
        # either threshold is crossed. Pass None for either to disable that
        # trigger (e.g. auto_flush_interval=None to only auto-flush on
        # memory pressure).
        self._auto_flush_memory = auto_flush_memory
        self._auto_flush_interval = auto_flush_interval
        self._last_flush_time = getNow()

        self._root_id = None

        if plugin:
            self._plugin = plugin
            self._plugin.set_db(self)
        else:
            self._plugin = None

    def _getDatasetUpdates(self, dset_id):
        """ Return list of updates for the given dataset id """

        if dset_id not in self._dataset_updates:
            self._dataset_updates[dset_id] = []

        return self._dataset_updates[dset_id]

    @property
    def db(self):
        """ return object db dictionary """
        return self._db

    @property
    def plugin(self):
        """ return the storage plugin instance """
        return self._plugin

    @plugin.setter
    def plugin(self, value: StoragePlugin):
        """ set the storage plugin """
        if self._plugin and not self._plugin.isClosed():
            self.flush()
            self._plugin.close()
        self._plugin = value
        if self._plugin:
            self._plugin.set_db(self)

    @property
    def root_id(self):
        """ return root uuid """
        return self._root_id

    def is_new(self, obj_id):
        """ return true if this is a new object (has not been persisted) """
        obj_id = getHashTagForId(obj_id)
        return obj_id in self._new_objects

    def is_dirty(self, obj_id):
        """ return true if this object has been modified """
        obj_id = getHashTagForId(obj_id)
        if self.is_new(obj_id):
            return True
        if obj_id in self._resized_datasets:
            return True
        return obj_id in self._dirty_objects

    def is_resized(self, dset_id):
        """ return true if this dataset has been resized """
        dset_id = getHashTagForId(dset_id)

        if dset_id in self._resized_datasets:
            return True
        else:
            return False

    @property
    def new_objects(self):
        return self._new_objects

    @property
    def dirty_objects(self):
        return self._dirty_objects

    @property
    def deleted_objects(self):
        return self._deleted_objects

    @property
    def resized_datasets(self):
        return self._resized_datasets

    @property
    def memory_usage(self):
        """ Approximate number of bytes currently held in memory for pending
        (not yet flushed) dataset value updates - the dominant contributor to
        memory growth, since dataset spaces can be arbitrarily large.
        New/dirty/deleted object metadata (group, dataset, attribute JSON) is
        comparatively negligible and not included. """
        total = 0
        for updates in self._dataset_updates.values():
            for (_, arr) in updates:
                total += arr.nbytes
        return total

    @property
    def last_flush_time(self):
        """ Time (per time_util.getNow()) that flush() last completed successfully,
        or of __init__() if there hasn't been one yet """
        return self._last_flush_time

    @property
    def auto_flush_memory(self):
        """ memory_usage threshold (bytes) that triggers an automatic flush,
        or None if the memory-based trigger is disabled """
        return self._auto_flush_memory

    @property
    def auto_flush_interval(self):
        """ Number of seconds since the last flush that triggers an automatic
        flush, or None if the time-based trigger is disabled """
        return self._auto_flush_interval

    def _maybeAutoFlush(self):
        """ Flush pending changes if either auto-flush threshold has been
        crossed. No-op if no plugin is set/open, or the plugin is the
        no-op NullPlugin (which can't actually persist anything). """
        if self._plugin is None or self._plugin.isClosed():
            return
        if isinstance(self._plugin, NullPlugin):
            return
        if self._auto_flush_memory is not None and self.memory_usage >= self._auto_flush_memory:
            self.log.debug(f"auto-flush: memory_usage {self.memory_usage} >= {self._auto_flush_memory}")
            self.flush()
        elif self._auto_flush_interval is not None and \
                (getNow() - self._last_flush_time) >= self._auto_flush_interval:
            self.log.debug(f"auto-flush: {getNow() - self._last_flush_time}s since last flush")
            self.flush()

    def make_dirty(self, obj_id):
        """ Mark the object as dirty and update the lastModified timestamp """
        obj_id = getHashTagForId(obj_id)
        if obj_id not in self.db:
            self.log.error("make dirty called on deleted object")
            raise KeyError(f"obj_id: {obj_id} not found")
        if self.db[obj_id] is None:
            # object deleted, just return
            return
        obj_json = self.db[obj_id]
        obj_json["lastModified"] = getNow()
        if not self.is_new(obj_id):
            # object hasn't been initially written yet, add to dirty_object set
            self._dirty_objects.add(obj_id)
        self._maybeAutoFlush()

    def flush(self):
        """ write out any changes """
        self.log.debug("db.flush()")
        self._checkPlugin()
        if not self.plugin.flush():
            # flush not successful, don't clear dirty set
            self.log.error("plugin flush failed")
            return False
        self.log.debug("clearing new, dirty, deleted sets")
        # reset new, dirty and deleted sets
        self._new_objects.clear()
        self._dirty_objects.clear()
        self._deleted_objects.clear()
        self._resized_datasets.clear()
        self._dataset_updates.clear()
        self._last_flush_time = getNow()

        return True

    def readAll(self):
        """ read all meta data objects from the plugin's storage and save to db """

        self.log.debug("readAll")
        if self.closed:
            raise IOError("database is not open")

        obj_ids = set()
        obj_ids.add(self.root_id)
        while obj_ids:
            obj_id = obj_ids.pop()
            self.log.debug(f"readAll, get {obj_id}")
            obj_json = self.getObjectById(obj_id)  # will add obj_id to db if not already present
            if getCollectionForId(obj_id) == "groups":
                # add any hard links to the set
                links = obj_json["links"]
                for title in links:
                    link_json = links[title]
                    if "id" in link_json:
                        link_id = link_json["id"]
                        obj_ids.add(link_id)

    def copy(self, other_db):
        """ Write this db's current content into other_db, which must already be open with its
        own plugin.  Used to convert between storage formats (e.g. h5tojson, jsontoh5) without a
        single Hdf5db juggling two different plugin types at once: open a source db and a
        destination db, each with its own plugin, then call source_db.copy(dest_db).

        Object ids are minted fresh in other_db, so an internal id_map is used to translate any
        embedded object/region reference values (which otherwise still refer to this db's ids)
        as they're copied. If a source dataset uses a committed (named) datatype, the destination
        dataset gets an equivalent inline (non-shared) dtype rather than a reference to a copied
        committed type - so datasets that shared one committed type in the source will each get
        their own independent (but structurally identical) type in the destination.
        """
        if other_db.closed:
            raise IOError("other_db is not open")

        id_map = {self.root_id: other_db.root_id}

        def translate_id(old_id):
            # normalize first: a reference element's embedded id may be a
            # bare Schema 1 uuid (getHashTagForId() needs the "<collection>/"
            # prefix intact to know which "g-"/"d-"/"t-" letter to apply),
            # while id_map's keys are always the fully-qualified form.
            old_id = getHashTagForId(old_id)
            return id_map.get(old_id, old_id)

        def translate_ref_element(val, ref):
            if not val:
                return val
            if is_reference(ref):
                is_bytes = isinstance(val, bytes)
                text = val.decode("ascii") if is_bytes else val
                if not text or text == "null":
                    # "null" is the on-the-wire sentinel for an unset object
                    # reference (distinct from Reference.tolist()'s own
                    # empty-string convention) - nothing to translate
                    return val
                if "/" in text:
                    collection = text.split("/", 1)[0]
                    new_text = f"{collection}/{translate_id(text)}"
                else:
                    new_text = translate_id(text)
                return new_text.encode("ascii") if is_bytes else new_text
            elif is_regionreference(ref):
                raw = val.item() if isinstance(val, np.ndarray) else val
                if not raw:
                    return val
                region_ref = raw if isinstance(raw, RegionReference) else RegionReference.frombytes(raw)
                if region_ref.id is None:
                    return val
                new_ref = RegionReference("datasets/" + translate_id(region_ref.id), region_ref.selection_bytes)
                return new_ref.tobytes()
            else:
                raise TypeError(f"Unexpected ref type: {ref}")

        def translate_refs(arr, dtype):
            """ Return a copy of arr with any embedded object/region reference element rewritten
            from this db's object ids to other_db's (via id_map), leaving non-reference data
            unchanged. """
            if not has_reference(dtype):
                return arr
            if len(dtype) > 0:
                out = arr.copy()
                for name in dtype.fields:
                    out[name] = translate_refs(arr[name], dtype.fields[name][0])
                return out
            if dtype.metadata and "ref" in dtype.metadata:
                ref = dtype.metadata["ref"]
                out = arr.reshape(-1).copy()
                src = arr.reshape(-1)
                for i in range(src.shape[0]):
                    out[i] = translate_ref_element(src[i], ref)
                return out.reshape(arr.shape)
            # vlen wrapping a reference (or reference-containing) base type -
            # each element is itself a nested array of the base dtype
            base_dt = dtype.metadata["vlen"]
            out = arr.reshape(-1).copy()
            src = arr.reshape(-1)
            for i in range(src.shape[0]):
                elem = src[i]
                if elem is None or not isinstance(base_dt, np.dtype):
                    continue
                out[i] = translate_refs(np.asarray(elem), base_dt)
            return out.reshape(arr.shape)

        def copy_attributes(src_id, dst_id):
            for name in self.getAttributes(src_id):
                attr_json = self.getAttribute(src_id, name)
                dtype = self.getDtype(attr_json)
                shape_json = attr_json["shape"]
                if shape_json["class"] == "H5S_NULL":
                    other_db.createAttribute(dst_id, name, None, shape="H5S_NULL", dtype=dtype)
                    continue
                shape = () if shape_json["class"] == "H5S_SCALAR" else tuple(shape_json["dims"])
                value = self.getAttributeValue(src_id, name)
                if value is not None and has_reference(dtype):
                    value = translate_refs(value, dtype)
                other_db.createAttribute(dst_id, name, value, shape=shape, dtype=dtype)

        def copy_dataset_values(src_id, dst_id, dtype, shape_json):
            if shape_json["class"] == "H5S_NULL":
                return  # nothing to copy
            if shape_json["class"] == "H5S_SCALAR":
                sel_all = selections.select((), ...)
                arr = self.getDatasetValues(src_id, sel_all)
                if arr is not None:
                    if has_reference(dtype):
                        arr = translate_refs(arr, dtype)
                    other_db.setDatasetValues(dst_id, sel_all, arr)
                return
            # copy chunk by chunk so the whole dataset is never loaded into memory at once
            chunk_iter = self.getChunkIterator(src_id)
            for chunk_arr in chunk_iter:
                if chunk_arr is None:
                    continue  # no explicit value was ever set for this dataset
                if has_reference(dtype):
                    chunk_arr = translate_refs(chunk_arr, dtype)
                other_db.setDatasetValues(dst_id, chunk_iter.sel, chunk_arr)

        def create_shell(src_id):
            """ Create the (empty, contentless) destination object for src_id and record it in
            id_map - just enough that any OTHER object's reference to src_id can already be
            translated, regardless of which order objects are processed in. """
            if src_id in id_map:
                return id_map[src_id]

            collection = getCollectionForId(src_id)
            src_json = self.getObjectById(src_id)
            cpl = src_json.get("creationProperties")
            if cpl and "filters" in cpl:
                # some older fixtures omit a filter's "name" key (deriving it
                # from "class") - validateFilters() (called by createDataset()
                # below) requires it, unlike the low-level h5py write path,
                # which tolerates and just skips a malformed filter entry
                cpl = dict(cpl)
                cpl["filters"] = [_withFilterName(f) for f in cpl["filters"]]

            if collection == "groups":
                dst_id = other_db.createGroup(cpl=cpl)
            elif collection == "datatypes":
                dtype = self.getDtype(src_json)
                dst_id = other_db.createCommittedType(dtype, cpl=cpl)
            elif collection == "datasets":
                shape_json = src_json["shape"]
                dtype = self.getDtype(src_json)
                if shape_json["class"] == "H5S_NULL":
                    shape, maxdims = None, None
                elif shape_json["class"] == "H5S_SCALAR":
                    shape, maxdims = (), None
                else:
                    shape = tuple(shape_json["dims"])
                    # createDataset() requires H5D_CHUNKED layout for any
                    # maxdims - some fixtures declare maxdims equal to dims
                    # (not usefully resizable) alongside a non-chunked
                    # layout, which would otherwise fail that check
                    if "maxdims" in shape_json and getDatasetLayoutClass(src_json) == "H5D_CHUNKED":
                        maxdims = tuple(shape_json["maxdims"])
                    else:
                        maxdims = None
                dst_id = other_db.createDataset(shape=shape, maxdims=maxdims, dtype=dtype, cpl=cpl)
            else:
                raise TypeError(f"unexpected collection: {collection}")

            id_map[src_id] = dst_id
            return dst_id

        def copy_content(src_id):
            """ Copy src_id's attributes and (for a dataset) values into its already-created
            destination counterpart. Reference-valued data can only be safely translated once
            EVERY object has a destination id, so this must run after ALL shells exist. """
            dst_id = id_map[src_id]
            copy_attributes(src_id, dst_id)
            if getCollectionForId(src_id) == "datasets":
                src_json = self.getObjectById(src_id)
                dtype = self.getDtype(src_json)
                copy_dataset_values(src_id, dst_id, dtype, src_json["shape"])

        def create_links(src_grp_id):
            dst_grp_id = id_map[src_grp_id]
            for name in self.getLinks(src_grp_id):
                link_json = self.getLink(src_grp_id, name)
                link_class = link_json["class"]
                if link_class == "H5L_TYPE_HARD":
                    tgt_dst_id = id_map[link_json["id"]]
                    other_db.createHardLink(dst_grp_id, name, tgt_dst_id)
                elif link_class == "H5L_TYPE_SOFT":
                    other_db.createSoftLink(dst_grp_id, name, link_json["h5path"])
                elif link_class == "H5L_TYPE_EXTERNAL":
                    other_db.createExternalLink(dst_grp_id, name, link_json["h5path"], link_json["file"])
                else:
                    other_db.createCustomLink(dst_grp_id, name, dict(link_json))

        # pass 1: discover and create every object reachable from root (without
        # attributes/values/links yet), so any reference to it can already be
        # translated, and so a hard link's target is always already known -
        # this also correctly handles circular group references
        visited = set()
        obj_ids = [self.root_id]
        create_shell(self.root_id)
        while obj_ids:
            src_grp_id = obj_ids.pop()
            if src_grp_id in visited:
                continue
            visited.add(src_grp_id)
            for name in self.getLinks(src_grp_id):
                link_json = self.getLink(src_grp_id, name)
                if link_json["class"] != "H5L_TYPE_HARD":
                    continue
                tgt_id = link_json["id"]
                create_shell(tgt_id)
                if getCollectionForId(tgt_id) == "groups":
                    obj_ids.append(tgt_id)

        # pass 2: copy attributes and dataset values now that every object
        # (and therefore every possible reference target) has a destination id
        for src_id in list(id_map):
            copy_content(src_id)

        # pass 3: wire up all links now that every target object exists
        for src_grp_id in visited:
            create_links(src_grp_id)

    def open(self):
        """ open the storage plugin, installing a NullPlugin if none is set """
        self.log.debug("db.open()")

        if self.plugin is None:
            self.plugin = NullPlugin(None, app_logger=self.log)

        if not self.plugin.isClosed():
            self.log.debug("db is already opened")
            raise IOError("db is already opened")

        plugin_root_id = self.plugin.open()
        self.log.debug(f"got plugin root_id: {plugin_root_id}")
        if self._root_id:
            if plugin_root_id != self._root_id:
                raise IOError("plugin root id does not match db root id")
        else:
            self._root_id = plugin_root_id

        if self._root_id not in self.db:
            # a brand new, empty store (e.g. a fresh H5JsonPlugin) has nothing
            # to report for the root yet, so synthesize one directly - but
            # otherwise leave the root object unfetched, exactly like any
            # other object, so it's loaded lazily (via getObjectById()) on
            # first access rather than always eagerly pulled in by open()
            obj_json = self.plugin.getObjectById(self._root_id)
            if obj_json is None:
                self.db[self._root_id] = {"links": {}, "attributes": {}, "cpl": {}}

        self.log.debug(f"db.open() - returning root_id: {self._root_id}")
        return self._root_id

    def close(self):
        """ close the storage plugin's handle """
        self.log.info("Hdf5db __close")

        if self.plugin:
            if not isinstance(self.plugin, NullPlugin):
                # a NullPlugin can never persist anything - flushing it always
                # "fails" by design, so skip the (spurious) error log
                self.flush()
            self.plugin.close()

    @property
    def closed(self):
        if self.plugin:
            return self.plugin.isClosed()
        elif self._root_id:
            return True
        else:
            return False

    def __enter__(self):
        """ called on package init """
        self.log.info("Hdf5db __enter")
        return self

    def __exit__(self, type, value, traceback):
        """ called on package exit """
        self.log.info("Hdf5db __exit")
        self.close()

    def _checkPlugin(self):
        """ check the storage plugin is set and open """
        if self.plugin is None:
            raise IOError("plugin not set")
        if self.plugin.isClosed():
            raise IOError("plugin is closed")

    def getObjectById(self, obj_id, refresh=False):
        """ return object with given id """
        self._checkPlugin()
        obj_id = getHashTagForId(obj_id)
        if obj_id not in self.db or (refresh and not self.is_new(obj_id) and not self.is_dirty(obj_id)):
            # load the obj from the plugin
            self.log.debug(f"getObjectById - fetching {obj_id} from plugin")
            obj_json = self.plugin.getObjectById(obj_id)
            self.db[obj_id] = obj_json
        obj_json = self.db[obj_id]

        return obj_json

    def getObjectIdByPath(self, h5path, parent_id=None):
        """ Return id for the given link path starting from parent_id if set,
        otherwise the root_id """

        if h5path == "/":
            return self.root_id  # just return root id

        if parent_id is None:
            parent_id = self.root_id
        else:
            parent_id = getHashTagForId(parent_id)

        self.log.debug(f"getObjectIdDByPath(h5path: {h5path} parent_id: {parent_id}")

        obj_json = self.getObjectById(parent_id)
        if obj_json is None:
            self.log.warning("getObjectIdDByPath - parent_id not found")
            raise KeyError("parent_id: {parent_id} not found")

        obj_id = parent_id

        link_names = h5path.split('/')
        self.log.debug(f"link_names: {link_names}")
        for link_name in link_names:
            if not link_name:
                continue
            link_tgt = None
            self.log.debug(f"link_name: {link_name}")
            if not obj_id:
                break
            if 'links' not in obj_json:
                self.log.error(f"expected to find links key in: {obj_json}")
                raise KeyError(h5path)
            links = obj_json['links']
            self.log.debug(f"links: {links}")
            if link_name not in links:
                self.log.warning(f"link: {link_name} not found in {obj_id}")
                self.log.debug(f"links: {links}")
                raise KeyError(h5path)
            link_tgt = links[link_name]
            self.log.debug(f"link_tgt: {link_tgt}")
            link_class = link_tgt['class']
            obj_id = None
            obj_json = None
            if link_class == 'H5L_TYPE_HARD':
                # hard link
                obj_id = link_tgt['id']
                obj_json = self.getObjectById(obj_id)
            elif link_class == 'H5L_TYPE_SOFT':
                self.log.warning("getObjectIdByPath can't follow soft links")
            elif link_class == 'H5L_TYPE_EXTERNAL':
                self.log.warning("getObjectIdByPath can't follow external links")
            else:
                self.log.error(f"link type: {link_class} not supported")

            if not obj_id:
                self.log.warning(f"get_bypath {h5path} not found")
                raise KeyError(h5path)
        return obj_id

    def getObjectByPath(self, path):
        """ Get Object JSON at given path """
        obj_id = self.getObjectIdByPath(path)
        obj_json = self.getObjectById(obj_id)
        return obj_json

    def getPathsForObjectId(self, obj_id, parent_id=None, path_prefix="", _visited=None):
        """ Return list of paths for the given object id starting from parent_id if set,
        otherwise the root_id """
        # TBD: this function will be rather slow for domains with a large number
        # of objects (it will search through the complete heirarchy).

        if parent_id is None:
            parent_id = self.root_id
        else:
            parent_id = getHashTagForId(parent_id)

        if _visited is None:
            _visited = set()

        if parent_id in _visited:
            self.log.warning(f"circular reference detected at path: {path_prefix}")
            return []
        _visited.add(parent_id)

        obj_json = self.getObjectById(parent_id)
        if obj_json is None:
            self.log.warning("getPathsForObjectId - parent_id not found")
            raise KeyError("parent_id: {parent_id} not found")

        paths = []
        obj_id = getHashTagForId(obj_id)

        if parent_id == obj_id:
            paths.append(path_prefix if path_prefix else "/")

        if 'links' in obj_json:
            links = obj_json['links']
            for link_name in links:
                link_tgt = links[link_name]
                link_class = link_tgt['class']
                if link_class == 'H5L_TYPE_HARD':
                    # hard link
                    tgt_obj_id = link_tgt['id']
                    kwargs = {"parent_id": tgt_obj_id, "_visited": _visited}
                    kwargs["path_prefix"] = path_prefix + "/" + link_name
                    paths.extend(self.getPathsForObjectId(obj_id, **kwargs))
                elif link_class == 'H5L_TYPE_SOFT':
                    self.log.warning("getPathsForObjectId can't follow soft links")
                elif link_class == 'H5L_TYPE_EXTERNAL':
                    self.log.warning("getPathsForObjectId can't follow external links")
                else:
                    self.log.error(f"link type: {link_class} not supported")

        return paths

    def getDtype(self, obj_json):
        """ Return numpy data type for given dataset, datatype, or attribute
        """

        if "type" not in obj_json:
            # group id?
            raise TypeError(f"{obj_json} does not have a datatype")
        type_item = obj_json["type"]
        if isValidUuid(type_item) and getCollectionForId(type_item) == "datatypes":
            ctype_id = "t-" + getUuidFromId(type_item)
            ctype_json = self.getObjectById(ctype_id)
            if ctype_json is None:
                raise KeyError(f"ctype: {ctype_id} not found")

            type_json = ctype_json["type"].copy()
            type_json["id"] = ctype_id
            dtype = createDataType(type_json)
        else:
            dtype = createDataType(type_item)

        return dtype

    def getAttributes(self, obj_id):
        """
        Get attributes given an object id and name
        returns: JSON object
        """

        obj_json = self.getObjectById(obj_id)
        # some plugins (e.g. H5JsonPlugin, reading an on-disk file that omits
        # empty attribute lists) don't always include the "attributes" key
        attrs = obj_json.get("attributes", {})
        names = []

        for name in attrs:
            attr_json = attrs[name]
            if attr_json is None:
                continue
            if "DELETED" in attr_json:
                continue  # deleted attr
            names.append(name)

        return names

    def getAttribute(self, obj_id, name, includeData=True):
        """
        Get attribute given an object id and name
        returns: JSON object
        """

        attr_names = self.getAttributes(obj_id)
        if name not in attr_names:
            return None

        obj_json = self.getObjectById(obj_id)
        attrs = obj_json["attributes"]

        attr_json = attrs[name]

        return attr_json

    def getAttributeValue(self, obj_id, name):
        """ Return NDArray of the given attribute value """
        attr_json = self.getAttribute(obj_id, name)
        if attr_json is None:
            raise KeyError(f"attribute {name} not found")
        shape_json = attr_json["shape"]
        if shape_json["class"] == "H5S_NULL":
            # no value for empty shape attributes
            return None
        elif shape_json["class"] == "H5S_SCALAR":
            dims = ()
        else:
            dims = shape_json["dims"]
        dtype = self.getDtype(attr_json)

        value = attr_json["value"]
        arr = jsonToArray(dims, dtype, value)

        return arr

    def createAttribute(self, obj_id, name, value, shape=None, dtype=None):
        """
        create an attribute - will override any existing attributes
        """

        # TBD: if dtype is a committed ref type, fetch it first
        # TBD: also, check special case for complex types

        if isinstance(dtype, str) and dtype.startswith("datatypes/"):
            ctype_id = dtype[len("datatypes/"):]
            if getCollectionForId(ctype_id) != "datatypes":
                raise TypeError(f"unexpected dtype value for createAttribute: {dtype}")
            if ctype_id not in self.db:
                raise KeyError(f"ctype: {ctype_id} not found")
            ctype_json = self.getObjectById(ctype_id)
            type_json = ctype_json["type"].copy()
            type_json["id"] = ctype_id
            dtype = createDataType(type_json)

        # First, make sure we have a NumPy array
        if isinstance(value, Reference) and dtype is None:
            dtype = special_dtype(ref=Reference)
        if shape == "H5S_NULL":
            if value:
                raise ValueError("Value can't be set for Null space attributes")
            if dtype is None:
                raise ValueError("Dtype must be set for Null space attributes")
            else:
                dtype = np.dtype(dtype)
        else:
            try:
                value = np.asarray(value, dtype=dtype, order='C')
            except ValueError:
                # some special cases for compound and vlen types are handled
                # by jsonToArray...
                if shape is None or dtype is None:
                    raise
                value = jsonToArray(shape, dtype, value)
            if dtype is None:
                dtype = value.dtype
            else:
                dtype = np.dtype(dtype)  # In case a string, e.g. 'i8' is passed

        # Where a top-level array type is requested, we have to do some
        # fiddling around to present the data as a smaller array of
        # sub-arrays.
        if value is not None:
            if dtype.subdtype is not None:
                subdtype, subshape = dtype.subdtype

                # Make sure the subshape matches the last N axes' sizes.
                if shape[-len(subshape):] != subshape:
                    raise ValueError(f"Array dtype shape {subshape} is incompatible with data shape {shape}")

                # New "advertised" shape and dtype
                shape = shape[0:len(shape) - len(subshape)]
                dtype = subdtype

            # Not an array type; make sure to check the number of elements
            # is compatible, and reshape if needed.
            else:
                if isinstance(shape, tuple):
                    if np.prod(shape) != np.prod(value.shape):
                        raise ValueError("Shape of new attribute conflicts with shape of data")

                    if shape != value.shape:
                        value = value.reshape(shape)

                # We need this to handle special string types.
                value = np.asarray(value, dtype=dtype)

            value_json = bytesArrayToList(value)

        else:
            value_json = None

        if shape is None and value is not None:
            shape = value.shape
        if shape == "H5S_NULL":
            shape_json = {"class": "H5S_NULL"}
        elif len(shape) == 0:
            shape_json = {"class": "H5S_SCALAR"}
        else:
            shape_json = {"class": "H5S_SIMPLE"}
            shape_json["dims"] = list(shape)

        obj_json = self.getObjectById(obj_id)
        attrs_json = obj_json["attributes"]
        type_json = getTypeItem(dtype)
        # finally put it all together...
        attr_json = {"shape": shape_json, "type": type_json}
        if shape != "H5S_NULL":
            attr_json["value"] = value_json
            if isOpaqueDtype(dtype):
                attr_json["encoding"] = "base64"
        attr_json["created"] = getNow()

        # slot into the obj_json["attrs"]
        attrs_json[name] = attr_json

        # mark object as dirty
        self.make_dirty(obj_id)

    def deleteAttribute(self, obj_id, name):
        """ delete the given attribute """
        obj_json = self.getObjectById(obj_id)
        attrs_json = obj_json["attributes"]
        if name not in attrs_json:
            raise KeyError(f"attribute [{name}] not found in {obj_id}")
        attr_json = attrs_json[name]
        attr_json["DELETED"] = getNow()  # mark key for deletion

        self.make_dirty(obj_id)

    def getDatasetValues(self, dset_id, sel, query=None):
        """
        Get values from dataset identified by obj_id.
        If a slices list or tuple is provided, it should have the same
        number of elements as the rank of the dataset.
        If a query is provided, it should be a string representing a boolean expression,
        and the return value will be a 1D ndarray of the (full-record) values within sel
        that satisfy the query, rather than the values of the selection itself.
        """

        if query is not None:
            return self._getDatasetValuesByQuery(dset_id, sel, query)

        def _result_dtype(base_dtype, fields):
            """Return the dtype for the result array given a field selection.

            If fields is None or the dtype is not compound, return base_dtype.
            If a single field is requested, return that field's scalar dtype.
            If multiple fields are requested, return a sub-compound dtype with
            those fields in the same order as in base_dtype.
            """
            if fields is None or len(base_dtype) == 0:
                return base_dtype
            ordered = [f for f in base_dtype.names if f in fields]
            if not ordered:
                raise ValueError(f"None of the requested fields {fields} found in dtype")
            if len(ordered) == 1:
                return base_dtype.fields[ordered[0]][0]
            return np.dtype([(f, base_dtype.fields[f][0]) for f in ordered])

        def _extract_fields(val, fields, rdtype):
            """Extract the selected fields from a compound ndarray.

            Returns val unchanged when field selection is not active.
            For a single field returns a plain array; for multiple fields
            returns a sub-compound array in dataset-dtype order.
            """
            if fields is None or len(val.dtype) == 0:
                return val
            ordered = [f for f in val.dtype.names if f in fields]
            if len(ordered) == 1:
                return val[ordered[0]]
            result = np.zeros(val.shape, dtype=rdtype)
            for f in ordered:
                result[f] = val[f]
            return result

        def _assign(arr, tgt, val, src, write_fields):
            """Copy val[src] into arr[tgt], optionally restricted to write_fields."""
            src_val = val[()] if val.ndim == 0 else val[src]
            if write_fields:
                for f in write_fields:
                    arr[tgt][f] = src_val if len(val.dtype) == 0 else src_val[f]
            else:
                arr[tgt] = src_val

        def init_arr(rdtype, cpl):
            """ create an ndarray with the given shape, dtype and fill_value
                (if the latter is found in the creation properties list) """
            if sel.select_type == selections.H5S_SEL_FANCY:
                arr_shape = sel.mshape
            elif hasattr(sel, "count"):
                arr_shape = sel.count if isinstance(sel.count, tuple) else (sel.count, )
            else:
                arr_shape = (sel.nselect,)
            arr = np.zeros(arr_shape, dtype=rdtype)
            if "fillValue" in cpl:
                fillValue = cpl["fillValue"]
                if len(rdtype) > 0 and isinstance(fillValue, list):
                    # for a compound dtype, a plain list fillValue (one value
                    # per field) must be a tuple for numpy to broadcast it as
                    # a single record value - assigning the list as-is instead
                    # broadcasts each element as a separate scalar attempt
                    fillValue = tuple(fillValue)
                arr[...] = fillValue
            return arr

        dset_id = getHashTagForId(dset_id)
        self.log.info(f"getDatasetValues dset_id: {dset_id}, sel: {sel}")

        dset_json = self.getObjectById(dset_id)
        shape_json = dset_json["shape"]
        if not isinstance(sel, selections.Selection):
            raise TypeError("Expected Selection class")

        dtype = self.getDtype(dset_json)
        rdtype = _result_dtype(dtype, sel.fields)

        if "creationProperties" in dset_json:
            cpl = dset_json["creationProperties"]
        else:
            cpl = {}

        updates = self._getDatasetUpdates(dset_id)

        shape_class = getShapeClass(shape_json)

        if shape_class == "H5S_NULL":
            # return None for selections on null space
            return None

        if sel.shape != getShapeDims(shape_json):
            raise ValueError("Selection shape does not match dataset shape")

        if shape_class == "H5S_SCALAR":
            if sel.select_type != selections.H5S_SEL_ALL:
                raise ValueError("Only SELECT_ALL selections are supported for scalar datasets")
            if sel.shape != ():
                raise ValueError("Selection shape does not match dataset shape")
            if updates:
                # for scalars the update has to be the requested value
                (update_sel, arr) = updates[-1]
            elif dset_id in self._new_objects:
                arr = init_arr(rdtype, cpl)
            else:
                # fetch from the plugin
                arr = self.plugin.getDatasetValues(dset_id, sel, dtype=dtype)
                if arr is None:
                    raise KeyError(f"Data for dataset {dset_id} not returned")
                arr = _extract_fields(arr, sel.fields, rdtype)
            # done with NULL and SCALAR cases
            return arr

        # simple dataset
        arr = None
        fetch = True

        # determine if we need to get data from the plugin
        if isinstance(self._plugin, NullPlugin) or dset_id in self._new_objects:
            fetch = False
        else:
            for (update_sel, update_val) in updates:
                sel_inter = selections.intersect(sel, update_sel)
                if sel_inter.nselect == 0:
                    continue
                if selections.contained(sel, update_sel):
                    # desired selection is wholly contained in this update
                    # TBD: determine if multiple updates would contain all the
                    # required elements
                    fetch = False
                    break
        if fetch:
            # get last saved version of the data from the plugin
            arr = self.plugin.getDatasetValues(dset_id, sel, dtype=dtype)
            arr = _extract_fields(arr, sel.fields, rdtype)
        else:
            # initialize an array with fill value if given
            arr = init_arr(rdtype, cpl)

        # apply any updates that impact this selection
        rank = len(sel.shape)
        sel_list_dims = [d for d in range(rank) if isinstance(sel.slices[d], list)]
        is_paired_read = (sel.select_type == selections.H5S_SEL_POINTS and len(sel_list_dims) == rank)
        for (update_sel, update_val) in updates:
            x_sel = selections.intersect(sel, update_sel)
            if x_sel.nselect == 0:
                continue

            # If the update has a field restriction and the read has a different
            # (or no) field restriction, check for overlap and skip when empty.
            if x_sel.fields is not None and len(x_sel.fields) == 0:
                continue  # no overlapping fields between update and read

            # Extract requested fields from compound update data.
            eff_val = _extract_fields(update_val, sel.fields, rdtype)

            # Determine which output field(s) to write to when the update is
            # field-restricted but the read selection covers all (compound) fields.
            write_fields = None  # None means write the whole element
            if update_sel.fields is not None and len(rdtype) > 0:
                write_fields = [f for f in rdtype.names if f in update_sel.fields]

            if is_paired_read:
                # Paired-coordinate read: output is 1-D, one entry per point pair.
                # Map each intersected pair back to its 1-D output index.
                n_pairs = len(sel.slices[sel_list_dims[0]])
                sel_pt_to_idx = {
                    tuple(sel.slices[d][i] for d in range(rank)): i
                    for i in range(n_pairs)
                }
                upd_list_dims = [d for d in range(rank)
                                 if isinstance(update_sel.slices[d], list)]
                is_paired_update = update_sel.select_type == selections.H5S_SEL_POINTS
                n_x = len(x_sel.slices[sel_list_dims[0]])
                for i in range(n_x):
                    pt = tuple(x_sel.slices[d][i] for d in range(rank))
                    tgt_idx = sel_pt_to_idx.get(pt)
                    if tgt_idx is None:
                        continue
                    if is_paired_update:
                        n_upd = len(update_sel.slices[upd_list_dims[0]])
                        upd_pt_to_idx = {
                            tuple(update_sel.slices[d][j] for d in range(rank)): j
                            for j in range(n_upd)
                        }
                        src_idx = upd_pt_to_idx.get(pt)
                        if src_idx is None:
                            continue
                        _assign(arr, tgt_idx, eff_val, src_idx, write_fields)
                    else:
                        src_pt = tuple(pt[d] - update_sel.start[d] for d in range(rank))
                        _assign(arr, tgt_idx, eff_val, src_pt, write_fields)
            elif update_sel.select_type == selections.H5S_SEL_POINTS:
                # Point update: eff_val is 1-D indexed by position in update_sel.
                # Iterate intersected points and copy each value individually.
                rank = len(sel.shape)
                upd_pt_to_idx = {
                    pt: j for j, pt in enumerate(selections._iter_points(update_sel))
                }
                for pt in selections._iter_points(x_sel):
                    src_idx = upd_pt_to_idx.get(pt)
                    if src_idx is None:
                        continue
                    tgt_coords = tuple(pt[d] - sel.start[d] for d in range(rank))
                    _assign(arr, tgt_coords, eff_val, src_idx, write_fields)
            else:
                src_sel = selections.translate(update_sel, x_sel)
                tgt_sel = selections.translate(sel, x_sel)
                # arr.shape == sel.mshape, which excludes:
                #   • integer-indexed (scalar) dims
                #   • all but the first of paired same-length list dims
                # Reconstruct the correct index tuple using sel.slices as a
                # guide for which dims are in mshape.
                # eff_val retains full dataset rank (reshaped on write), so
                # src_sel.slices is used as-is.
                saw_list = False
                tgt_slices_list = []
                for sel_dim, loc_dim in zip(sel.slices, tgt_sel.slices):
                    if isinstance(sel_dim, int):
                        pass  # scalar dim: excluded from mshape
                    elif isinstance(sel_dim, list):
                        if not saw_list:
                            tgt_slices_list.append(loc_dim)
                            saw_list = True
                        # else: paired list dim — shares the mshape entry with
                        # the first list dim, so skip it
                    else:  # slice
                        tgt_slices_list.append(loc_dim)
                tgt_slices = tuple(tgt_slices_list)
                _assign(arr, tgt_slices, eff_val, src_sel.slices, write_fields)

        return arr

    def _getDatasetValuesByQuery(self, dset_id, sel, query):
        """
        Return the dataset values (as a 1D ndarray) within sel that satisfy query.

        Mirrors queryDataset's reader-delegation / chunk-by-chunk strategy, but
        gathers the matching values themselves rather than their indices. This
        way a caller that wants values (rather than indices) doesn't have to
        call queryDataset and then do a separate point-selection read to fetch
        them - which would require readers that support queries to read from
        storage twice.
        """
        if not isinstance(query, str):
            raise TypeError("Expected query string")
        if not isinstance(sel, selections.Selection):
            raise TypeError("Expected Selection class")

        dset_json = self.getObjectById(dset_id)
        shape_json = dset_json["shape"]
        shape_class = getShapeClass(shape_json)
        if shape_class == "H5S_NULL":
            raise ValueError("querying null space dataset not supported")
        dims = getShapeDims(shape_json)
        if sel.shape != dims:
            raise TypeError("Selection shape does not match dataset shape")

        rank = len(dims)
        dtype = self.getDtype(dset_json)
        updates = self._getDatasetUpdates(dset_id)

        # Delegate query to the plugin when it has relevant, not-superseded data
        query_fetch = not (isinstance(self._plugin, NullPlugin) or dset_id in self._new_objects)
        if query_fetch:
            for (update_sel, _) in updates:
                if selections.contained(sel, update_sel):
                    query_fetch = False
                    break

        if query_fetch:
            try:
                result = self.plugin.getDatasetValues(dset_id, sel, dtype=dtype, query=query)
            except NotImplementedError:
                result = None
            if result is not None:
                return result

        try:
            chunk_iter = ChunkIterator(self, dset_id, sel=sel)
        except ValueError:
            # ChunkIterator doesn't support this selection (e.g. a fancy/point
            # selection, or a scalar dataset) - fall back to filtering the
            # whole selection at once
            arr = self.getDatasetValues(dset_id, sel)
            rel = arrayQuery(query, arr)
            if len(rel) == 0:
                return np.zeros((0,), dtype=arr.dtype)
            return arr[tuple(rel[:, d] for d in range(arr.ndim))]

        # walk the selection chunk by chunk so the whole selection is never
        # loaded into memory at once. Each chunk is fetched via the ordinary
        # (non-query) getDatasetValues, so it already reflects any pending
        # in-memory updates.
        hits = []
        for chunk_arr in chunk_iter:
            rel = arrayQuery(query, chunk_arr)
            if len(rel) == 0:
                continue
            hits.append(chunk_arr[tuple(rel[:, d] for d in range(rank))])

        if hits:
            return np.concatenate(hits, axis=0)
        return np.zeros((0,), dtype=dtype)

    def getChunkIterator(self, dset_id, sel=None):
        """
        Return a ChunkIterator that reads through the given dataset's values
        chunk by chunk, without loading the entire dataset into memory.
        If sel is provided, only chunks intersecting that selection are
        iterated over (each still trimmed to the selection's bounds),
        otherwise the entire dataset is iterated over.
        """
        return ChunkIterator(self, dset_id, sel=sel)

    def queryDataset(self, dset_id, query, sel=None, limit=0, update_value=None):
        """
        Query the given dataset using the selection and query expression
        If sel is provided, only the elements in the selection will be queried,
        otherwise the entire dataset will be queried.
        If limit is provided, only the first limit number of elements that match the query will be returned.
        If update_value is provied, elements matching the query (up to limit elements if limit is non-zero)
        will be updated to the given value. For a compound dtype, update_value may be a dict mapping
        field names to the value to set for that field - only those fields are modified, and the rest
        of each matching element is left unchanged.
        Return a numpy array of indices for the elements that match the query
        """

        def queryPlugin(dset_id, query, sel=None, limit=0, update_value=None):
            result = None
            try:
                result = self.plugin.queryDataset(dset_id, query, sel=sel, limit=limit, update_value=update_value)
            except NotImplementedError:
                # This plugin doesn't support queryDataset
                pass

            if result is None:
                rank = len(sel.shape)
                try:
                    chunk_iter = ChunkIterator(self, dset_id, sel=sel)
                except ValueError:
                    # ChunkIterator doesn't support this selection (e.g. a fancy/point
                    # selection, or a scalar dataset) - fall back to querying the
                    # entire selection at once
                    arr = self.getDatasetValues(dset_id, sel)
                    result = arrayQuery(query, arr, limit=limit)
                    result = _query_rel_to_abs(sel, result, rank)
                else:
                    # query the dataset chunk by chunk so the whole selection is
                    # never loaded into memory at once
                    hits = []
                    nhits = 0
                    for chunk_arr in chunk_iter:
                        chunk_rel = arrayQuery(query, chunk_arr)
                        if len(chunk_rel) == 0:
                            continue
                        hits.append(_query_rel_to_abs(chunk_iter.sel, chunk_rel, rank))
                        nhits += len(chunk_rel)
                        if limit > 0 and nhits >= limit:
                            break

                    result = np.concatenate(hits, axis=0) if hits else np.zeros((0, rank), dtype='int64')
                    if limit > 0 and len(result) > limit:
                        result = result[:limit]

            if update_value is not None and len(result) > 0:
                # update the values at the matching indices
                dtype = self.getDtype(self.getObjectById(dset_id))
                if isinstance(update_value, dict):
                    # a dict maps field names to the value to set for that field -
                    # only those fields are modified, the rest of each record is
                    # left as-is
                    if len(dtype) == 0:
                        raise TypeError("update_value dict is only supported for compound dtypes")
                    fields = [f for f in dtype.names if f in update_value]
                    if not fields:
                        raise ValueError(
                            f"None of the requested fields {list(update_value.keys())} found in dtype")
                    if len(fields) == 1:
                        value = np.asarray(update_value[fields[0]], dtype=dtype.fields[fields[0]][0])
                    else:
                        value_dtype = np.dtype([(f, dtype.fields[f][0]) for f in fields])
                        value = np.zeros((), dtype=value_dtype)
                        for f in fields:
                            value[f] = update_value[f]
                    update_sel = selections.select(sel.shape, result, fields=fields)
                else:
                    value = np.asarray(update_value, dtype=dtype)
                    update_sel = selections.select(sel.shape, result)
                self.setDatasetValues(dset_id, update_sel, value)
            return result

        #
        # start of queryDataset
        #
        if not isinstance(query, str):
            raise TypeError("Expected query string")

        if sel is not None and not isinstance(sel, selections.Selection):
            raise TypeError("Expected Selection class")
        if not isinstance(limit, int) or limit < 0:
            raise TypeError("Expected non-negative integer for limit")

        dset_json = self.getObjectById(dset_id)
        shape_json = dset_json["shape"]

        shape_class = getShapeClass(shape_json)
        if shape_class == "H5S_NULL":
            raise ValueError("querying null space dataset not supported")
        dims = getShapeDims(shape_json)
        if sel is None:
            sel = selections.select(dims, ...)

        if update_value is not None:
            # do flush so we can be sure to do an atomic operation if the plugin supports it
            self.flush()
            results = queryPlugin(dset_id, query, sel=sel, limit=limit, update_value=update_value)
            return results

        updates = self._getDatasetUpdates(dset_id)

        if sel.shape != dims:
            raise TypeError("Selection shape does not match dataset shape")

        full_shape = sel.shape
        rank = len(full_shape)

        # Delegate query to the plugin when it has relevant data
        query_fetch = not (isinstance(self._plugin, NullPlugin) or dset_id in self._new_objects)
        if query_fetch:
            for (update_sel, _) in updates:
                if selections.contained(sel, update_sel):
                    query_fetch = False
                    break

        result_mask = np.zeros(full_shape, dtype=bool)
        if query_fetch:
            fetched = queryPlugin(dset_id, query, sel=sel, limit=limit)
            if len(fetched) > 0:
                result_mask[tuple(fetched[:, d].astype(int) for d in range(rank))] = True

        for (update_sel, update_val) in updates:
            x_sel = selections.intersect(sel, update_sel)
            if x_sel.nselect == 0:
                continue

            # Invalidate reader results overwritten by this update
            inter_mask = np.zeros(full_shape, dtype=bool)
            inter_mask[x_sel.slices] = True
            result_mask &= ~inter_mask

            # Query the updated values at the intersection
            if update_sel.select_type == selections.H5S_SEL_POINTS:
                # update_val is 1-D, indexed by position in update_sel (not
                # sliceable via translate(), which only handles hyperslabs)
                x_points = list(selections._iter_points(x_sel))
                if not x_points:
                    continue
                upd_pt_to_idx = {pt: j for j, pt in enumerate(selections._iter_points(update_sel))}
                x_vals = update_val[[upd_pt_to_idx[pt] for pt in x_points]]
                x_rel = arrayQuery(query, x_vals)
                if len(x_rel) > 0:
                    abs_coords = np.array([x_points[i] for i in x_rel[:, 0]], dtype='u8')
                    result_mask[tuple(abs_coords[:, d] for d in range(rank))] = True
            else:
                local_sel = selections.translate(update_sel, x_sel)
                x_vals = update_val[local_sel.slices]
                x_rel = arrayQuery(query, x_vals)
                if len(x_rel) > 0:
                    abs_result = _query_rel_to_abs(x_sel, x_rel, rank)
                    result_mask[tuple(abs_result[:, d].astype(int) for d in range(rank))] = True

        indices = np.argwhere(result_mask)
        if limit > 0 and len(indices) > limit:
            indices = indices[:limit]

        return indices

    def setDatasetValues(self, dset_id, sel, arr):
        """
        Write the given ndarray to the dataset using the selection
        """

        if not isinstance(sel, selections.Selection):
            raise TypeError("Expected Selection class")

        if not isinstance(arr, np.ndarray):
            raise TypeError("Expected ndarray for data value")

        dset_json = self.getObjectById(dset_id)
        shape_json = dset_json["shape"]

        shape_class = getShapeClass(shape_json)
        if shape_class == "H5S_NULL":
            raise ValueError("writing to null space dataset not supported")
        dims = getShapeDims(shape_json)

        updates = self._getDatasetUpdates(dset_id)

        tgt_dt = self.getDtype(dset_json)

        if shape_class == "H5S_SCALAR":
            if sel.select_type != selections.H5S_SEL_ALL:
                # TBD: support other selection types
                raise ValueError("Only SELECT_ALL selections are supported for scalar datasets")
            if sel.shape != ():
                raise ValueError("Selection shape does not match dataset shape")

            # for an H5T_ARRAY (subarray) dtype, the subarray dims are absorbed
            # directly into arr's shape (see array_util.jsonToArray), so a
            # scalar dataset of e.g. a (3, 2) array type expects arr.shape == (3, 2)
            expected_shape = tgt_dt.subdtype[1] if tgt_dt.subdtype is not None else ()
            if arr.shape != expected_shape:
                raise ValueError("Expected scalar array for scalar dataset")

        if sel.fields is not None and len(tgt_dt) > 0:
            # Field-restricted write: check arr against the selected field dtype.
            ordered = [f for f in tgt_dt.names if f in sel.fields]
            if not ordered:
                raise ValueError(f"None of the requested fields {sel.fields} "
                                 f"found in dataset dtype")
            if len(ordered) == 1:
                expected_dt = tgt_dt.fields[ordered[0]][0]
            else:
                expected_dt = np.dtype([(f, tgt_dt.fields[f][0]) for f in ordered])
        else:
            expected_dt = tgt_dt
        src_dt = arr.dtype
        # for an H5T_ARRAY (subarray) dtype, the subarray dims are absorbed
        # directly into arr's shape (see array_util.jsonToArray), so arr's
        # own dtype is just the base scalar type, not the subarray descriptor
        cmp_dt = expected_dt.subdtype[0] if expected_dt.subdtype is not None else expected_dt
        if not _dtypesStructurallyEqual(src_dt, cmp_dt):
            raise TypeError(f"arr.dtype {src_dt} doesn't match expected dtype {expected_dt}")

        if sel.select_type == selections.H5S_SEL_POINTS:
            if arr.shape == ():
                # broadcast the scalar to match the number of selected points, so
                # the stored update value can be indexed like any other point update
                arr = np.full(sel.mshape, arr[()], dtype=arr.dtype)
            elif sel.nselect != arr.shape[0]:
                raise TypeError("Selection shape does not match number of points")
        elif sel.select_type == selections.H5S_SEL_FANCY:
            if arr.shape != sel.mshape:
                raise TypeError("Array shape does not match fancy selection shape")
        elif sel.select_type == selections.H5S_SEL_ALL:
            if sel.shape != getShapeDims(shape_json):
                raise TypeError("Selection shape does not match dataset shape")
        elif sel.select_type == selections.H5S_SEL_HYPERSLABS:
            if sel.shape != dims:
                raise TypeError("Selection shape does not match dataset shape")
            # Allow scalar arrays when writing a field-restricted selection
            # (the scalar will be broadcast to all selected positions).
            if arr.shape != ():
                # for an H5T_ARRAY (subarray) dtype, arr's shape has the
                # subarray dims appended as a suffix (see array_util.jsonToArray)
                # - compare/broadcast against just the logical (dataset-rank)
                # prefix, then restore the suffix when reshaping.
                subarray_dims = expected_dt.subdtype[1] if expected_dt.subdtype is not None else ()
                if subarray_dims:
                    if arr.shape[len(arr.shape) - len(subarray_dims):] != subarray_dims:
                        raise TypeError(
                            f"Array shape {arr.shape} doesn't match subarray dtype dims {subarray_dims}")
                    arr_shape = arr.shape[:len(arr.shape) - len(subarray_dims)]
                else:
                    arr_shape = arr.shape
                if 0 < len(arr_shape) < len(dims):
                    # arr has fewer dims than the dataset rank (e.g. a 1-D array
                    # written to a slice of a 3-D dataset).  Validate against
                    # sel.mshape (the effective shape after scalar-indexed axes
                    # are removed), then reshape to sel.count so the stored array
                    # has the full dataset rank with size-1 scalar-axis dims.
                    if arr_shape != sel.mshape:
                        raise TypeError(
                            f"Array shape {arr_shape} doesn't match "
                            f"selection mshape {sel.mshape}")
                    arr = arr.reshape(sel.count + subarray_dims)
                elif len(arr_shape) != len(dims):
                    raise TypeError("Array shape does not match dataset shape")
                else:
                    try:
                        sel.broadcast(arr_shape)
                    except TypeError:
                        raise
        else:
            raise TypeError("Unsupported selection type")

        if (sel.select_type == selections.H5S_SEL_ALL or sel.shape == sel.mshape) \
                and sel.fields is None:
            # for full-coverage writes with no field restriction, discard prior
            # updates since this one completely overwrites the dataset.
            updates.clear()

        # make a copy in case the client updates it later
        arr = arr.copy()
        updates.append((sel, arr))
        self.make_dirty(dset_id)

    def resizeDataset(self, dset_id, shape):
        """
        Resize existing Dataset
        """
        self.log.info(f"resizeDataset {dset_id}, {shape}")

        dset_json = self.getObjectById(dset_id)  # will throw exception if not found
        old_dims = getShapeDims(dset_json)
        resize_dataset(dset_json, shape)

        if dset_id not in self.new_objects:
            self._resized_datasets.add(dset_id)

        new_dims = getShapeDims(dset_json)
        rank = len(new_dims)

        # adjust any selections in the update list
        updates = self._getDatasetUpdates(dset_id)
        for i in range(len(updates)):
            (sel_update, arr) = updates[i]
            if sel_update.select_type == selections.H5S_SEL_HYPERSLABS:
                slices = list(sel_update.slices)
                for dim in range(rank):
                    s = slices[dim]
                    if s.stop > new_dims[dim]:
                        # selection outside new bounds of dataset
                        slices[dim] = slice(s.start, new_dims[dim], s.step)
                sel_update = selections.select(new_dims, tuple(slices))
                updates[i] = (sel_update, arr)

        # if the shape has shrunk in any dimension, do a flush now
        do_flush = False
        for i in range(len(new_dims)):
            if new_dims[i] < old_dims[i]:
                do_flush = True
                break

        if do_flush:
            self.flush()
        else:
            self._maybeAutoFlush()

    def deleteObject(self, obj_id):
        """ Delete the given object """
        self.log.info(f"deleteObject: {obj_id}")
        if obj_id not in self.db:
            raise KeyError(f"Object {obj_id} not found for deletion")
        if obj_id == self.root_id:
            raise KeyError("Root group cannot be deleted")
        self.db[obj_id] = None

        if obj_id in self._new_objects:
            self._new_objects.remove(obj_id)

        if obj_id in self._dirty_objects:
            self._dirty_objects.remove(obj_id)

        if obj_id in self._resized_datasets:
            self._resized_datasets.remove(obj_id)

        self._deleted_objects.add(obj_id)
        self._maybeAutoFlush()

    def getLinks(self, grp_id):
        """ Get the links for the given group """
        grp_json = self.getObjectById(grp_id)
        if "links" not in grp_json:
            # some plugins (e.g. H5JsonPlugin, reading an on-disk file that
            # omits an empty links list) don't always include the "links"
            # key for a group with zero links - only raise if this really
            # isn't a group at all
            if getCollectionForId(grp_id) != "groups":
                raise KeyError(f"No links - {grp_id} not a group?")
            return []
        links = grp_json["links"]
        names = []
        for name in links:
            link_json = links[name]
            if link_json is None:
                continue
            if "DELETED" in link_json:
                continue  # deleted link
            names.append(name)
        return names

    def getLink(self, grp_id, name):
        """ Get the given link """

        obj_json = self.getObjectById(grp_id)
        links = obj_json["links"]
        if name not in links:
            self.log.info(f"Link [{name}] not found in {grp_id}")
            return None
        link_json = links[name]
        if "DELETED" in link_json:
            self.log.info(f"Link {name} in {grp_id} has been deleted")
            return None

        return link_json

    def _addLink(self, grp_id, name, link_json):
        obj_json = self.getObjectById(grp_id)
        links = obj_json["links"]
        if name in links:
            self.log.warning(f"Link [{name}] already exists in {grp_id}")
        links[name] = link_json
        self.make_dirty(grp_id)

    def createHardLink(self, grp_id, name, tgt_id):
        """ Create a new hardlink """
        link_json = {"class": "H5L_TYPE_HARD", "id": tgt_id}
        link_json["created"] = getNow()
        self._addLink(grp_id, name, link_json)

    def createSoftLink(self, grp_id, name, h5path):
        """ Create a soft link """
        link_json = {"class": "H5L_TYPE_SOFT", "h5path": h5path}
        link_json["created"] = getNow()
        self._addLink(grp_id, name, link_json)

    def createCustomLink(self, grp_id, name, link_json):
        """ create a custom link """
        if link_json.get("class") != "H5L_TYPE_USER_DEFINED":
            link_json["class"] = "H5L_TYPE_USER_DEFINED"
        link_json["created"] = getNow()
        self._addLink(grp_id, name, link_json)

    def createExternalLink(self, grp_id, name, h5path, filepath):
        """ Create a external link link """
        link_json = {"class": "H5L_TYPE_EXTERNAL", "h5path": h5path, "file": filepath}
        link_json["created"] = getNow()
        self._addLink(grp_id, name, link_json)

    def deleteLink(self, grp_id, name):
        """ Delete the given link """
        grp_json = self.getObjectById(grp_id)
        if "links" not in grp_json:
            raise KeyError(f"No links - {grp_id} not a group?")
        links = grp_json["links"]
        if name not in links:
            raise KeyError(f"Link [{name}] not found in {grp_id}")
        link_json = links[name]
        link_json["DELETED"] = getNow()  # mark for deletion
        self.make_dirty(grp_id)
        grp_json = self.getObjectById(grp_id)
        links = grp_json["links"]

    def createGroup(self, cpl=None):
        """ Create a new group """
        if self.closed:
            raise ValueError("db is closed")
        grp_id = createObjId("groups", root_id=self.root_id)
        group_json = {"attributes": {}, "links": {}}
        if cpl:
            group_json["creationProperties"] = cpl
        else:
            group_json["creationProperties"] = {}
        group_json["created"] = getNow()
        self.db[grp_id] = group_json
        self._new_objects.add(grp_id)
        self._maybeAutoFlush()
        return grp_id

    def createCommittedType(self, datatype, cpl=None):
        """
        createCommittedType - creates new named datatype
        Returns item
        """
        if self.closed:
            raise ValueError("db is closed")
        self.log.info("createCommittedType")
        if cpl is None:
            cpl = {}

        ctype_id = createObjId(obj_type="datatypes", root_id=self.root_id)
        if isinstance(datatype, np.dtype):
            dt = datatype
        else:
            dt = createDataType(datatype)

        type_json = getTypeItem(dt)  # get canonical json description of datatype

        ctype_json = {"type": type_json, "attributes": {}, "creationProperties": cpl}
        ctype_json["created"] = getNow()
        self.db[ctype_id] = ctype_json
        self._new_objects.add(ctype_id)
        self._maybeAutoFlush()
        return ctype_id

    def createDataset(
        self,
        shape=None,
        maxdims=None,
        dtype=None,
        cpl=None,
    ):
        """
        createDataset - creates new dataset given shape and datatype
        Returns obj_id
        """
        if self.closed:
            raise ValueError("db is closed")
        type_json = getTypeItem(dtype)
        shape_json = getShapeJson(shape, maxdims=maxdims)

        dset_json = {"shape": shape_json, "type": type_json, "attributes": {}}
        if cpl:
            if "filters" in cpl:
                if self.plugin:
                    supported_filters = self.plugin.getFilters()
                else:
                    supported_filters = ()
                # validate and normalize supplied filter property list
                validateFilters(cpl["filters"], supported_filters=supported_filters)
            if cpl.get("fillValue"):
                fillvalue = cpl["fillValue"]
                # is it compatible with the array type?
                if hasattr(fillvalue, "tolist"):
                    # convert numpy object to list
                    fillvalue = fillvalue.tolist()
                fillvalue = _decode(fillvalue)
                if not isinstance(fillvalue, str) and hasattr(fillvalue, "__iter__"):
                    # fill value is a list, or similar: check that dtype is compound
                    if len(fillvalue) != len(dtype):
                        raise ValueError("Invalid fill value for non-compound type dataset")
                    fillvalue = list(fillvalue)
                    cpl["fillValue"] = fillvalue
                else:
                    if type_json["class"] == "H5T_COMPOUND":
                        raise ValueError("Invalid fill value for compound type dataset")
            dset_json["creationProperties"] = cpl
        else:
            dset_json["creationProperties"] = {}

        if maxdims and getDatasetLayoutClass(dset_json) != "H5D_CHUNKED":
            raise ValueError("Only datasets with 'H5D_CHUNKED' layout can be resizable")
        dset_json["created"] = getNow()

        dset_id = createObjId("datasets", root_id=self.root_id)
        self.db[dset_id] = dset_json
        self._new_objects.add(dset_id)
        self._maybeAutoFlush()
        return dset_id

    def getCollection(self, col_type=None):
        obj_ids = []
        for obj_id in self.db:
            if self.db[obj_id] is None:
                # skip deleted objects
                continue
            if not col_type or getCollectionForId(obj_id) == col_type:
                obj_ids.append(obj_id)
        return obj_ids

    def __len__(self):
        # return the number of objects
        count = 0
        for obj_id in self.db:
            # skip deleted objects
            if self.db[obj_id] is not None:
                count += 1
        return count

    def __iter__(self):
        """ Iterate over object ids """

        for obj_id in self.db:
            if self.db[obj_id] is None:
                # skip deleted objects
                continue
            yield obj_id

    def __contains__(self, obj_id):
        """ Test if a obj id  exists """
        return obj_id in self.db and self.db[obj_id] is not None
