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
import logging
from os import stat as os_stat

from ..objid import createObjId, getCollectionForId
from ..hdf5dtype import getTypeItem, isOpaqueDtype
from ..array_util import bytesArrayToList

from .. import selections
from .. import filters

from ..h5py_util import is_reference, is_regionreference, has_reference, convert_dtype
from ..h5reader import H5Reader


class H5pyReader(H5Reader):
    """
    This class can be used by HDF5DB to read content from an HDF5 file (using h5py)
    """

    def _copy_element(self, val, src_dt, tgt_dt, fin=None):
        """ convert the given dataset or attribute element from h5py to h5json equivalent """

        out = None
        if len(src_dt) > 0:
            out_fields = []
            i = 0
            for name in src_dt.fields:
                field_src_dt = src_dt.fields[name][0]
                field_tgt_dt = tgt_dt.fields[name][0]
                field_val = val[i]
                i += 1
                out_field = self._copy_element(field_val, field_src_dt, field_tgt_dt, fin=fin)
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
                    try:
                        fin_obj = fin[val]
                    except AttributeError as ae:
                        msg = f"Unable able to get obj for ref value: {ae}"
                        self.log.error(msg)
                        raise ValueError(msg)

                    addr = h5py.h5o.get_info(fin_obj.id).addr
                    if addr not in self._addr_map:
                        msg = f"No object found for ref object: {fin_obj.name}"
                        self.log.warning(msg)
                        out = ""
                    else:
                        obj_id = self._addr_map[addr]
                        collection = getCollectionForId(obj_id)
                        out = f"{collection}/{obj_id}"

            elif is_regionreference(ref):
                self.log.warning("region reference not supported")
                # TBD: just return a null region reference till we have support
                out = ""
            else:
                raise TypeError(f"Unexpected ref type: {type(ref)}")
        elif src_dt.metadata and "vlen" in src_dt.metadata:
            if not isinstance(val, np.ndarray):
                raise TypeError(f"Expecting ndarray or vlen element, but got: {type(val)}")
            if not tgt_dt.metadata or "vlen" not in tgt_dt.metadata:
                raise TypeError(f"Expected tgt dtype to be vlen, but got: {tgt_dt}")
            src_vlen_dt = src_dt.metadata["vlen"]
            tgt_vlen_dt = tgt_dt.metadata["vlen"]
            if has_reference(src_vlen_dt):
                if len(val.shape) == 0:
                    # scalar array
                    e = val[()]
                    v = self._copy_element(e, src_vlen_dt, tgt_vlen_dt, fin=fin)
                    out = np.array(v, dtype=tgt_dt)
                else:
                    out = np.zeros(val.shape, dtype=tgt_dt)
                    for i in range(len(out)):
                        e = val[i]
                        out[i] = self._copy_element(e, src_vlen_dt, tgt_vlen_dt, fin=fin)
            else:
                # can just directly copy the array
                out = np.zeros(val.shape, dtype=tgt_dt)
                out[...] = val[...]
        else:
            out = val  # can just copy as is
        return out

    def _copy_array(self, src_arr, fin=None):
        """Copy the numpy array to a new array.
            Convert any reference type to point to item in the target's hierarchy.
        """

        if not isinstance(src_arr, np.ndarray):
            raise TypeError(f"Expecting ndarray, but got: {src_arr}")
        tgt_dt = convert_dtype(src_arr.dtype, to_h5py=False)
        tgt_arr = np.zeros(src_arr.shape, dtype=tgt_dt)

        if has_reference(src_arr.dtype):
            # flatten array to simplify iteration
            count = int(np.prod(src_arr.shape))
            tgt_arr_flat = tgt_arr.reshape((count,))
            src_arr_flat = src_arr.reshape((count,))
            for i in range(count):
                e = src_arr_flat[i]
                element = self._copy_element(e, src_arr.dtype, tgt_dt, fin=fin)
                tgt_arr_flat[i] = element
            tgt_arr = tgt_arr_flat.reshape(src_arr.shape)
        else:
            # can just copy the entire array
            tgt_arr[...] = src_arr[...]
        return tgt_arr

    """
    def visit(self, path, obj):
        name = obj.__class__.__name__
        self.log.info(f"visit: {path} name: {name}")

        obj_id = createObjId(obj_type=name, root_id=self._root_id)  # create uuid

        self._id_map[obj_id] = obj

        addr = h5py.h5o.get_info(obj.id).addr
        self._addr_map[addr] = obj_id
    """

    def __init__(
        self,
        filepath,
        app_logger=None
    ):
        self._id_map = {}
        self._addr_map = {}
        if app_logger:
            self.log = app_logger
        else:
            self.log = logging.getLogger()
        if not h5py.is_hdf5(filepath):
            self.log.warning(f"File: {filepath} is not an HDF5 file")
            raise IOError("not an HDF5 file")
        super().__init__(filepath, app_logger=app_logger)
        self._f = None
        self._root_id = None

    def open(self):
        if self._f:
            return  # already open
        if self._id_map:
            return  # objects already loaded
        if not self._root_id:
            # get the root id from db if available
            if self.db.root_id:
                self.log.info("H5pyReader: got root_id from db")
                self._root_id = self.db.root_id
            else:
                self.log.info("H5pyReader: creating root id")
                self._root_id = createObjId(obj_type="groups")

        f = h5py.File(self.filepath)
        self._f = f
        self._id_map[self._root_id] = f
        addr = h5py.h5o.get_info(f.id).addr
        self._addr_map[addr] = self._root_id

        return self._root_id

    def close(self):
        # close h5py handles in map dict
        self._id_map = {}
        if self._f:
            self._f.close()
            self._f = None

    def isClosed(self):
        return False if self._f else True

    def get_root_id(self):
        """ Return root id """
        return self._root_id

    def getObjIdByAddress(self, addr):
        if addr in self._addr_map:
            return self._addr_map[addr]
        else:
            return None

    def getAttribute(self, obj_id, name, include_data=True):
        """ Return JSON for the given attribute """

        obj = self._id_map[obj_id]

        if name not in obj.attrs:
            msg = f"Attribute: [{name}] not found in object: {obj.name}"
            self.log.info(msg)
            return None

        # get the attribute!
        attrObj = h5py.h5a.open(obj.id, np.bytes_(name))

        item = {}

        # check if the dataset is using a committed type
        typeid = attrObj.get_type()
        type_item = None
        if h5py.h5t.TypeID.committed(typeid):
            type_uuid = None
            addr = h5py.h5o.get_info(typeid).addr
            type_uuid = self.getObjIdByAddress(addr)
            committedType = self._id_map[type_uuid]
            type_item = getTypeItem(committedType.dtype)
            type_item["id"] = type_uuid
        else:
            type_item = getTypeItem(attrObj.dtype)
        item["type"] = type_item

        shape_item = {}
        if attrObj.shape is None or attrObj.get_storage_size() == 0:
            # If storage size is 0, assume this is a null space obj
            # See: h5py issue https://github.com/h5py/h5py/issues/279
            shape_item["class"] = "H5S_NULL"
        else:
            if attrObj.shape:
                shape_item["class"] = "H5S_SIMPLE"
                shape_item["dims"] = attrObj.shape
            else:
                shape_item["class"] = "H5S_SCALAR"

        item["shape"] = shape_item
        if shape_item["class"] == "H5S_NULL":
            include_data = False
        elif isinstance(type_item, dict) and type_item["class"] == "H5T_OPAQUE":
            # TBD - don't include data for OPAQUE until JSON serialization
            # issues are addressed
            include_data = False
        else:
            pass  # use include_data parameter

        if include_data:
            try:
                data = obj.attrs[name]
                # convert from h5py to h5json
                data = self._copy_array(data, fin=obj.file)
            except TypeError:
                self.log.warning("type error reading attribute")

        if include_data and data is not None:
            value = bytesArrayToList(data)
            item["value"] = value
        else:
            pass  # no data
        stats = self.getStats()
        item['created'] = stats["lastModified"]  # use file modification time as attr creation time
        return item

    def getAttributes(self, obj_id, include_data=True):
        h5obj = self._id_map[obj_id]
        self.log.info(f"getAttributes: {obj_id} include_data={include_data}")
        items = {}  # with python 3.7+, this will maintain the attribute order we got from h5py
        attrs = h5obj.attrs
        for name in attrs:
            item = self.getAttribute(obj_id, name, include_data=include_data)
            items[name] = item

        return items

    def _getLink(self, parent, link_name):
        if link_name not in parent:
            return None

        item = {"title": link_name}
        # get the link object, one of HardLink, SoftLink, or ExternalLink
        try:
            linkObj = parent.get(link_name, None, False, True)
            linkClass = linkObj.__class__.__name__
        except TypeError:
            # UDLink? set class as 'user'
            linkClass = "UDLink"  # user defined links
            item["class"] = "H5L_TYPE_USER_DEFINED"
        if linkClass == "SoftLink":
            item["class"] = "H5L_TYPE_SOFT"
            item["h5path"] = linkObj.path
        elif linkClass == "ExternalLink":
            item["class"] = "H5L_TYPE_EXTERNAL"
            item["h5path"] = linkObj.path
            item["file"] = linkObj.filename
        elif linkClass == "HardLink":
            # Hardlink doesn't have any properties itself, just get the linked
            # object
            obj = parent[link_name]
            addr = h5py.h5o.get_info(obj.id).addr
            item["class"] = "H5L_TYPE_HARD"
            if addr not in self._addr_map:
                self.log.error(f"expected to find addr for link {link_name} in addr_map")
                item["id"] = None
            else:
                item["id"] = self._addr_map[addr]

        stats = self.getStats()
        item['created'] = stats["lastModified"]  # use file modification time as attr creation time

        return item

    def _getLinks(self, grp):
        items = {}  # with python 3.7+, this will maintain the link order we got from h5py
        for link_name in grp:
            item = self._getLink(grp, link_name)
            items[link_name] = item
        return items

    def _getGroup(self, grp, include_links=True):
        self.log.info(f"_getGroup alias: [{grp.name}]")

        item = {"alias": grp.name}

        if include_links:
            links = self._getLinks(grp)
            item["links"] = links
        return item

    def _getDatatype(self, ctype, include_attrs=True):
        self.log.info(f"getDatatype alias: ]{ctype.name}")
        item = {"alias": ctype.name}
        item["type"] = getTypeItem(ctype.dtype)

        return item

    def _getHDF5DatasetCreationProperties(self, dset, type_class):
        """ Get dataset creation properties maintained by HDF5 library """

        #
        # Fill in creation properties
        #
        creationProps = {}
        plist = h5py.h5d.DatasetID.get_create_plist(dset.id)

        # alloc time
        nAllocTime = plist.get_alloc_time()
        if nAllocTime == h5py.h5d.ALLOC_TIME_DEFAULT:
            creationProps["allocTime"] = "H5D_ALLOC_TIME_DEFAULT"
        elif nAllocTime == h5py.h5d.ALLOC_TIME_LATE:
            creationProps["allocTime"] = "H5D_ALLOC_TIME_LATE"
        elif nAllocTime == h5py.h5d.ALLOC_TIME_EARLY:
            creationProps["allocTime"] = "H5D_ALLOC_TIME_EARLY"
        elif nAllocTime == h5py.h5d.ALLOC_TIME_INCR:
            creationProps["allocTime"] = "H5D_ALLOC_TIME_INCR"
        else:
            self.log.warning(f"Unknown alloc time value: {nAllocTime}")

        # fill time
        nFillTime = plist.get_fill_time()
        if nFillTime == h5py.h5d.FILL_TIME_ALLOC:
            creationProps["fillTime"] = "H5D_FILL_TIME_ALLOC"
        elif nFillTime == h5py.h5d.FILL_TIME_NEVER:
            creationProps["fillTime"] = "H5D_FILL_TIME_NEVER"
        elif nFillTime == h5py.h5d.FILL_TIME_IFSET:
            creationProps["fillTime"] = "H5D_FILL_TIME_IFSET"
        else:
            self.log.warning(f"unknown fill time value: {nFillTime}")

        if type_class == "H5T_OPAQUE":
            # TBD: store opaque fill value as a hex string
            self.log.warning("Opaque fill value not supported")
        else:
            if plist.fill_value_defined() == h5py.h5d.FILL_VALUE_USER_DEFINED:
                creationProps["fillValue"] = bytesArrayToList(dset.fillvalue)

        # layout
        nLayout = plist.get_layout()
        if nLayout == h5py.h5d.COMPACT:
            creationProps["layout"] = {"class": "H5D_COMPACT"}
        elif nLayout == h5py.h5d.CONTIGUOUS:
            creationProps["layout"] = {"class": "H5D_CONTIGUOUS"}
        elif nLayout == h5py.h5d.CHUNKED:
            creationProps["layout"] = {"class": "H5D_CHUNKED", "dims": dset.chunks}
        else:
            self.log.warning(f"Unknown layout value: {nLayout}")

        num_filters = plist.get_nfilters()
        filter_props = []
        if num_filters:
            for n in range(num_filters):
                filter_info = plist.get_filter(n)
                opt_values = filter_info[2]
                filter_prop = {}
                filter_id = filter_info[0]
                filter_prop["id"] = filter_id
                if filter_info[3]:
                    filter_prop["name"] = bytesArrayToList(filter_info[3])
                hdf_filter = filters.getFilterItem(filter_id)
                if hdf_filter:
                    filter_prop["class"] = hdf_filter["class"]
                    if "options" in hdf_filter:
                        filter_opts = hdf_filter["options"]
                        for i in range(len(filter_opts)):
                            if len(opt_values) <= i:
                                break  # end of option values
                            opt_value = opt_values[i]
                            opt_value_enum = None
                            option_name = filter_opts[i]
                            if option_name in filters.HDF_FILTER_OPTION_ENUMS:
                                option_enums = filters.HDF_FILTER_OPTION_ENUMS[option_name]
                                if opt_value in option_enums:
                                    opt_value_enum = option_enums[opt_value]
                            if opt_value_enum:
                                filter_prop[option_name] = opt_value_enum
                            else:
                                filter_prop[option_name] = opt_value
                else:
                    # custom filter
                    filter_prop["class"] = "H5Z_FILTER_USER"
                    if opt_values:
                        filter_prop["parameters"] = opt_values
                filter_props.append(filter_prop)
            creationProps["filters"] = filter_props

        return creationProps

    def _getDataset(self, dset):
        """ return json representation of the given dataset """

        self.log.info(f"getDataset alias: [{dset.name}]")

        item = {"alias": dset.name}
        typeid = dset.id.get_type()
        if h5py.h5t.TypeID.committed(typeid):
            type_uuid = None
            addr = h5py.h5o.get_info(typeid).addr
            type_uuid = self.getObjIdByAddress(addr)
            committedType = self.getObjectById(type_uuid)
            type_item = committedType["type"]
            type_item["id"] = type_uuid
        else:
            type_item = getTypeItem(dset.dtype)
        item["type"] = type_item

        shape_item = {}
        if dset.shape is None:
            # new with h5py 2.6, null space datasets will return None for shape
            shape_item["class"] = "H5S_NULL"
        elif len(dset.shape) == 0:
            shape_item["class"] = "H5S_SCALAR"
        else:
            shape_item["class"] = "H5S_SIMPLE"
            shape_item["dims"] = list(dset.shape)
            maxshape = []
            include_maxdims = False
            for i in range(len(dset.shape)):
                extent = 0
                if len(dset.maxshape) > i:
                    extent = dset.maxshape[i]
                    if extent is None:
                        extent = 0
                    if extent > dset.shape[i] or extent == 0:
                        include_maxdims = True
                maxshape.append(extent)
            if include_maxdims:
                shape_item["maxdims"] = maxshape
        item["shape"] = shape_item

        item["cpl"] = self._getHDF5DatasetCreationProperties(dset, type_item["class"])

        return item

    def _getHardLinkIds(self, parent):
        """ create any ids for hard links of the group """

        self.log.debug(f"h5pyreader> _getHardlinkIds for {parent.name}")
        for link_name in parent:
            self.log.debug(f"h5py_reader> check link: {link_name}")

            try:
                linkObj = parent.get(link_name, None, False, True)
                linkClass = linkObj.__class__.__name__
            except TypeError:
                # UDLink? Go on to the next link
                continue
            if linkClass != "HardLink":
                self.log.debug(f"h5py_reader> ignoring {link_name} - type: {linkClass}")
            else:
                # get the linked object
                obj = parent[link_name]
                addr = h5py.h5o.get_info(obj.id).addr
                if addr not in self._addr_map:
                    name = obj.__class__.__name__
                    obj_id = createObjId(obj_type=name, root_id=self._root_id)  # create uuid
                    self.log.debug(f"h5py_reader> creating obj_id: {obj_id} for obj: {obj.name}")
                    self._id_map[obj_id] = obj
                    self._addr_map[addr] = obj_id
                else:
                    obj_id = self._addr_map[addr]
                    if obj_id not in self._id_map:
                        self.log.debug(f"h5py_reader> adding obj for {obj_id} to id_map")
                        self._id_map = obj
                    else:
                        self.log.debug("h5py_reader> obj {obj_id} already in id_map")

    def getObjectById(self, obj_id, include_attrs=True, include_links=True):
        """ return object with given id """
        if obj_id not in self._id_map:
            raise KeyError(f"{obj_id} not found")
        h5obj = self._id_map[obj_id]
        if isinstance(h5obj, h5py.Group):
            self._getHardLinkIds(h5obj)
            obj_json = self._getGroup(h5obj, include_links=include_links)
        elif isinstance(h5obj, h5py.Dataset):
            obj_json = self._getDataset(h5obj)
        elif isinstance(h5obj, h5py.Datatype):
            obj_json = self._getDatatype(h5obj)
        else:
            msg = f"unexpected object type: {type(h5obj)}"
            self.log.error(msg)
            raise TypeError(msg)

        if include_attrs:
            attributes = self.getAttributes(obj_id)
            obj_json["attributes"] = attributes

        return obj_json

    def getDatasetValues(self, dset_id, sel, dtype=None, query=None):
        """
        Get values from dataset identified by obj_id.
        If a slices list or tuple is provided, it should have the same
        number of elements as the rank of the dataset.
        """

        dset = self._id_map[dset_id]
        self.log.info(f"getDatasetValues: {dset_id}")
        if dset.shape is None:
            # TBD: return something like h5py.Empty in this case?
            return None
        if isOpaqueDtype(dset.dtype):
            # TBD: Opaque data not supported yet
            return None

        if query is not None:
            # h5py doesn't support query
            raise NotImplementedError("queryDataset not implemented for H5pyReader")

        if sel is None or sel.select_type == selections.H5S_SEL_ALL:
            arr = dset[...]
        elif isinstance(sel, selections.SimpleSelection):
            rank = len(sel.shape)
            slices = sel.slices
            list_dims = [d for d in range(rank) if isinstance(slices[d], list)]
            if len(list_dims) > 1:
                # h5py only supports one coordinate array at a time.
                # Decompose into n separate reads (one per paired-coordinate index)
                # then stack the results.
                list_dims_set = set(list_dims)
                n = len(slices[list_dims[0]])
                reads = []
                for i in range(n):
                    idx = tuple(
                        int(slices[d][i]) if d in list_dims_set else slices[d]
                        for d in range(rank)
                    )
                    reads.append(dset[idx])
                arr = np.stack(reads)
            else:
                arr = dset[slices]
        else:
            raise NotImplementedError("selection type not supported")

        # convert any h5py references to h5json references
        arr = self._copy_array(arr, fin=dset.file)
        return arr

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
