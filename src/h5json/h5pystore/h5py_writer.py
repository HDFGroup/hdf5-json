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
import h5py
import numpy as np
from os import stat as os_stat
import time

from ..objid import getCollectionForId, isValidUuid, createObjId
from ..hdf5dtype import createDataType
from ..h5py_util import is_reference, is_regionreference, has_reference, convert_dtype
from ..shape_util import getShapeDims, getShapeClass, isExtensible, getMaxDims
from ..array_util import jsonToArray
from ..track_util import getTrackTimes
from ..dset_util import getDatasetLayout, getFillValue
from ..filters import isCompressionFilter, getFilters, getFilterItem
from .. import selections
from .. import filters
from ..h5writer import H5Writer


class H5pyWriter(H5Writer):
    """
    This class saves state from the Hdf5Db class into an HDF5 file.
    """

    def __init__(
        self,
        filepath,
        append=False,
        no_data=False,
        app_logger=None
    ):
        super().__init__(filepath, append=append, no_data=no_data, app_logger=app_logger)
        self._id_map = {}
        if append:
            self._init = False
        else:
            self._init = True
        self._flush_time = 0.0
        self._f = None  # h5py file handle

    def _copy_element(self, val, src_dt, tgt_dt, fout=None):
        """ convert the given dataset or attribute element to h5py equivalent """
        out = None
        if len(src_dt) > 0:
            out_fields = []
            i = 0
            for name in src_dt.fields:
                field_src_dt = src_dt.fields[name][0]
                field_tgt_dt = tgt_dt.fields[name][0]
                field_val = val[i]
                i += 1
                out_field = self._copy_element(field_val, field_src_dt, field_tgt_dt)
                out_fields.append(out_field)
            out = tuple(out_fields)
        elif src_dt.metadata and "ref" in src_dt.metadata:
            if not tgt_dt.metadata or "ref" not in tgt_dt.metadata:
                raise TypeError(f"Expected tgt dtype to be ref, but got: {tgt_dt}")
            ref = tgt_dt.metadata["ref"]
            if is_reference(ref):
                # initialize out to null ref
                out = h5py.Reference()  # null h5py ref

                if ref and val:
                    if isinstance(val, bytes):
                        val = val.decode("ascii")
                    # strip out collection prefix if present
                    parts = val.split("/")
                    obj_uuid = parts[-1]
                    if not isValidUuid(obj_uuid):
                        msg = f"invalid uuid: {obj_uuid}"
                        self.log.warning(msg)
                    elif obj_uuid not in self._id_map:
                        self.log.warning(f"ref object {obj_uuid} not found")
                    else:
                        h5path = self._id_map[obj_uuid]
                        try:
                            obj = fout[h5path]
                            out = obj.ref
                        except KeyError:
                            self.log.warning(f"referenced object: {h5path} not found")

            elif is_regionreference(ref):
                self.log.warning("region reference not supported")
                # TBD: just return a null region reference till we have support
                out = h5py.RegionReference()
            else:
                raise TypeError(f"Unexpected ref type: {type(ref)}")
        elif src_dt.metadata and "vlen" in src_dt.metadata:
            if not tgt_dt.metadata or "vlen" not in tgt_dt.metadata:
                raise TypeError(f"Expected tgt dtype to be vlen, but got: {tgt_dt}")
            src_vlen_dt = src_dt.metadata["vlen"]
            tgt_vlen_dt = tgt_dt.metadata["vlen"]

            if has_reference(src_vlen_dt):
                if isinstance(val, np.ndarray) and val.shape == ():
                    val = val[()]
                if isinstance(val, np.ndarray) or isinstance(val, list) or isinstance(val, tuple):
                    count = len(val)
                    out = np.zeros((count,), dtype=tgt_dt)
                    for i in range(count):
                        e = val[i]
                        out[i] = self._copy_element(e, src_vlen_dt, tgt_vlen_dt, fout=fout)
                else:
                    # scalar array
                    v = self._copy_element(val, src_vlen_dt, tgt_vlen_dt, fout=fout)
                    out = np.array(v, dtype=tgt_dt)
            else:
                # can just directly copy the array
                out = np.zeros(val.shape, dtype=tgt_dt)
                out[...] = val[...]
        else:
            out = val  # can just copy as is
        return out

    def _copy_array(self, src_arr, fout=None):
        """Copy the numpy array to a new array.
            Convert any reference type to point to item in the target's hierarchy.
        """
        if not isinstance(src_arr, np.ndarray):
            raise TypeError(f"Expecting ndarray, but got: {src_arr}")
        tgt_dt = convert_dtype(src_arr.dtype, to_h5py=True)
        tgt_arr = np.zeros(src_arr.shape, dtype=tgt_dt)

        if has_reference(src_arr.dtype):
            # flatten array to simplify iteration
            count = int(np.prod(src_arr.shape))
            tgt_arr_flat = tgt_arr.reshape((count,))
            src_arr_flat = src_arr.reshape((count,))
            for i in range(count):
                e = src_arr_flat[i]
                element = self._copy_element(e, src_arr.dtype, tgt_dt, fout=fout)
                tgt_arr_flat[i] = element
            tgt_arr = tgt_arr_flat.reshape(src_arr.shape)
        else:
            # can just copy the entire array
            tgt_arr[...] = src_arr[...]
        return tgt_arr

    def _createGroup(self, parent, grp_json, name=None):
        """ create the group and any links it contains """
        grp = parent.create_group(name)
        return grp

    def _createDataset(self, parent, dset_json, name=None):
        """ create a dataset object """

        dtype = self.db.getDtype(dset_json)

        kwargs = {"dtype": dtype}
        shape_class = getShapeClass(dset_json)
        if shape_class == "H5S_NULL":
            # skip the shape keyword to create a null space dataset
            pass
        elif shape_class == "H5S_SCALAR":
            kwargs["shape"] = ()
        else:
            shape = getShapeDims(dset_json)
            kwargs["shape"] = shape
            if isExtensible(dset_json):
                maxshape = list(getMaxDims(dset_json))
                # replace any 0, or H5S_UNLIMITED with None
                for dim in range(len(maxshape)):
                    if maxshape[dim] in (0, "H5S_UNLIMITED"):
                        maxshape[dim] = None
                kwargs["maxshape"] = tuple(maxshape)

        fillvalue = getFillValue(dset_json)

        if fillvalue and len(dtype) > 1 and type(fillvalue) in (list, tuple):
            # for compound types, need to convert from list to dataset compatible element

            if len(dtype) != len(fillvalue):
                msg = "fillvalue has incorrect number of elements"
                raise ValueError(msg)

            fillvalue = jsonToArray((), dtype, fillvalue)

        kwargs["fillvalue"] = fillvalue

        track_times = getTrackTimes(dset_json)
        if track_times is not None:
            kwargs["track_times"] = track_times

        layout = getDatasetLayout(dset_json)
        if layout and "dims" in layout:
            kwargs["chunks"] = tuple(layout["dims"])

        filter_props = getFilters(dset_json)

        for filter_prop in filter_props:
            try:
                getFilterItem(filter_prop)
            except (KeyError, ValueError, TypeError):
                self.log.warning(f"unknown filter: {filter_prop} ignoring")
                continue
            filter_class = filter_prop["class"]
            filter_id = filter_prop["id"]
            filter_name = filter_prop["name"]

            if not h5py.h5z.filter_avail(filter_id):
                msg = f"filter not available, filter: {filter_class}, ignoring"
                self.log.warning(msg)
                continue

            if isCompressionFilter(filter_class):
                if kwargs.get("compression"):
                    msg = f"compression filter already set for {filter_class}, ignoring"
                    self.log.info(msg)
                    continue

                kwargs["compression"] = filter_name
                self.log.info(f"setting compression filter to: {filter_class}")
                if filter_class == "H5Z_FILTER_DEFLATE":
                    kwargs["compression"] = "gzip"  # h5py doesn't recognize 'deflate' name
                    # check for an optional compression value
                    if "level" in filter_prop:
                        kwargs["compression_opts"] = filter_prop["level"]
                elif filter_class == "H5Z_FILTER_SZIP":
                    bitsPerPixel = None
                    coding = "nn"

                    if "bitsPerPixel" in filter_prop:
                        bitsPerPixel = filter_prop["bitsPerPixel"]
                    if "coding" in filter_prop:
                        if filter_prop["coding"] == "H5_SZIP_EC_OPTION_MASK":
                            coding = "ec"
                        elif filter_prop["coding"] == "H5_SZIP_NN_OPTION_MASK":
                            coding = "nn"
                        else:
                            self.log.warning("invalid szip option: 'coding'")
                        # note: pixelsPerBlock, and pixelsPerScanline not supported by h5py,
                        # so these options will be ignored
                    if "pixelsPerBlock" in filter_props:
                        self.log.info("ignoring szip option: 'pixelsPerBlock'")
                    if "pixelsPerScanline" in filter_props:
                        self.log.info("ignoring szip option: 'pixelsPerScanline'")
                    if bitsPerPixel:
                        kwargs["compression_opts"] = (coding, bitsPerPixel)
                elif filter_class == "H5Z_FILTER_SHUFFLE":
                    kwargs["shuffle"] = True
                elif filter_class == "H5Z_FILTER_FLETCHER32":
                    kwargs["fletcher32"] = True
                elif filter_class == "H5Z_FILTER_SCALEOFFSET":
                    if "scaleOffset" in filter_prop:
                        kwargs["scaleoffset"] = filter_prop["scaleOffset"]
                else:
                    self.log.warning(f"Ignoring filter: {filter_class}")

        dset = parent.create_dataset(name, **kwargs)
        return dset

    def _createDatatype(self, parent, ctype_json, name=None):
        """ create a datatype object """

        type_item = ctype_json["type"]
        dtype = createDataType(type_item)
        parent[name] = dtype
        return parent[name]

    def _createObjects(self, parent, links_json, visited=set()):
        """ create child object in the given group, recurse for any sub-groups """

        titles = list(links_json.keys())
        for title in titles:
            link_json = links_json[title]
            link_class = link_json["class"]
            if "DELETED" in link_json:
                if title in parent:
                    # delete the link
                    self.log.debug(f"deleting link {title}")
                    del parent[title]
                # update the link json
                del links_json[title]
                continue

            if link_class == "H5L_TYPE_SOFT" and title not in parent:
                h5path = link_json["h5path"]
                parent[title] = h5py.SoftLink(h5path)
            elif link_class == "H5L_TYPE_EXTERNAL" and title not in parent:
                h5path = link_json["h5path"]
                filename = link_json["file"]
                parent[title] = h5py.ExternalLink(filename, h5path)
            elif link_class == "H5L_TYPE_USER_DEFINED" and title not in parent:
                self.log.warning("unable to create user-defined link: {title}")
            elif link_class == "H5L_TYPE_HARD":
                tgt_id = link_json["id"]

                collection = getCollectionForId(tgt_id)

                obj_json = self.db.getObjectById(tgt_id)

                if tgt_id in self._id_map:
                    # object has already been created
                    tgt_path = self._id_map[tgt_id]
                    tgt_obj = parent[tgt_path]
                    if title not in parent:
                        parent[title] = tgt_obj
                    if collection == "groups" and tgt_id not in visited:
                        # recurse over sub-objects to pick up any new links
                        grp_links = obj_json["links"]
                        visited.add(tgt_id)
                        self._createObjects(tgt_obj, grp_links, visited=visited)
                else:
                    # need to create tgt_id object
                    parent_path = parent.name
                    if parent_path[-1] != '/':
                        parent_path += '/'
                    self._id_map[tgt_id] = parent_path + title
                    kwds = {"name": title}
                    if collection == "groups":
                        tgt_grp = self._createGroup(parent, obj_json, **kwds)
                        if "links" in obj_json:
                            grp_links = obj_json["links"]
                            visited.add(tgt_id)
                            self._createObjects(tgt_grp, grp_links, visited=visited)
                    elif collection == "datasets":
                        self._createDataset(parent, obj_json, **kwds)
                    elif collection == "datatypes":
                        self._createDatatype(parent, obj_json, **kwds)
                    else:
                        self.log.warning(f"unexpected collection: {collection}")
                visited.add(tgt_id)

            else:
                self.log.warning(f"unexpected link class: {link_class}")

    def resizeDataset(self, dset_id, dset):
        """ Update the datasets shape """

        dset_json = self.db.getObjectById(dset_id)
        new_dims = getShapeDims(dset_json)
        dset.resize(new_dims)

    def updateDatasetValues(self, dset_id, dset):
        """ write any pending dataset values """

        updates = self.db._getDatasetUpdates(dset_id)

        for (sel, val) in updates:
            slices = []
            for dim in range(len(sel.shape)):
                start = sel.start[dim]
                stop = start + sel.count[dim]
                step = sel.step[dim]
                slices.append(slice(start, stop, step))
            slices = tuple(slices)
            dset[slices] = val
            self.log.debug(f"h5py_writer dset {dset.name} updated")

    def initializeDatasetValues(self, dset_id, dset):
        """ write all dataset values """

        if dset.shape is None:
            return  # null space dataset

        sel_all = selections.select(dset.shape, ...)
        arr = self.db.getDatasetValues(dset_id, sel_all)
        if arr is not None:
            dset[...] = arr

    def createAttribute(self, obj, name, attr_json):
        """ add the given attribute to obj """

        src_dt = self.db.getDtype(attr_json)

        # handle special case of null space attribute here
        shape_json = attr_json["shape"]
        shape_class = shape_json["class"]
        if shape_class == "H5S_NULL":
            obj.attrs[name] = h5py.Empty(convert_dtype(src_dt, to_h5py=True))
            return

        if shape_class == "H5S_SCALAR":
            dims = ()
        else:
            dims = shape_json["dims"]
        src_arr = jsonToArray(dims, src_dt, attr_json["value"])
        if not isinstance(src_arr, np.ndarray):
            raise TypeError("Unexpected type for src_arr")
        tgt_arr = self._copy_array(src_arr, fout=obj.file)
        obj.attrs[name] = tgt_arr

    def updateAttributes(self, obj_id, obj):
        """ create/replace any modified attributes """

        obj_json = self.db.getObjectById(obj_id)

        if "attributes" not in obj_json:
            # no attributes
            return

        attrs = obj_json["attributes"]
        for name in attrs:
            attr_json = attrs[name]
            if "DELETED" in attr_json:
                if name in obj.attrs:
                    # delete the attribute
                    self.log.debug(f"h5py_writer - delete attribute {name}")
                    del obj.attrs[name]
                else:
                    pass  # already deleted or never added
                continue
            if "created" in attr_json and attr_json["created"] < self._flush_time:
                # attribute should be saved already
                continue
            self.createAttribute(obj, name, attr_json)

    def flush(self):
        """ Write dirty items """
        if self.closed:
            # no db set yet
            self.log.warning("h5py_writer - flush called but no db")
            return False
        if not self._f:
            self.log.warning("h5py_writer file not open")
            raise IOError("open not called")

        self.log.info("h5py_writer.flush()")

        root_id = self.db.root_id
        self._id_map[root_id] = "/"

        if self.db.new_objects or self._init:
            root_json = self.db.getObjectById(root_id)

            if "links" in root_json:
                root_links = root_json["links"]
                self._createObjects(self._f, root_links, visited=set((root_id,)))

        # update attributes, dataset values
        for obj_id in self._id_map:
            if self.db.is_dirty(obj_id) or self._init:
                h5path = self._id_map[obj_id]
                obj = self._f[h5path]
                self.updateAttributes(obj_id, obj)
                collection = getCollectionForId(obj_id)
                if collection == "datasets":
                    if self.db.is_resized(obj_id):
                        self.resizeDataset(obj_id, obj)
                    if not self.no_data:
                        if self._init:
                            self.initializeDatasetValues(obj_id, obj)
                        else:
                            self.updateDatasetValues(obj_id, obj)
        # mark time write is complete
        # updates before this time will not need to be written
        # TBD: possible race condition with multithreading
        self._flush_time = time.time()

        self._init = False  # done with init after first flush
        return True  # all objects written successfully

    def open(self):
        """ open HDF5 file """
        self.log.debug("h5pyWriter open")
        if self.db is None:
            # no db set yet
            self.log.warning("no self.db db_ref")
            raise ValueError("no db")
        mode = 'a' if self._append else 'w'
        self.log.info(f"creating h5py file: {self._filepath} mode: {mode}")
        self._f = h5py.File(self._filepath, mode=mode)
        self._append = True  # switch to append mode for next file open
        if self.db.root_id:
            self._root_id = self.db.root_id
        else:
            self._root_id = createObjId(obj_type="groups")
        return self._root_id

    def close(self):
        """ close storage handle """
        self.log.debug("h5py_writer.close()")
        if not self._f:
            # no open on file
            return
        self.flush()
        self._f.close()
        self._f = None

    def isClosed(self):
        """ return closed status """
        return False if self._f else True

    def getStats(self):
        """ return a dictionary object with at minimum the following keys:
            'created': creation time
            'lastModified': modificationTime
            'owner': owner name
        """
        stat_info = os_stat(self.filepath)
        stats = {}
        stats['created'] = stat_info.st_ctime
        stats["lastModified"] = stat_info.st_mtime
        stats['owner'] = stat_info.st_uid  # TBD: convert to username?
        return stats

    def getFilters(self, compressors_only=False):
        """ return list of filters supported by h5py  """

        h5py_filters = ["H5Z_FILTER_DEFLATE",]

        if not compressors_only:
            h5py_filters.append("H5Z_FILTER_SHUFFLE")
            h5py_filters.append("H5Z_FILTER_FLETCHER32")
            h5py_filters.append("H5Z_FILTER_SZIP")
            h5py_filters.append("H5Z_FILTER_NBIT")
            h5py_filters.append("H5Z_FILTER_SCALEOFFSET")

        return tuple(h5py_filters)
