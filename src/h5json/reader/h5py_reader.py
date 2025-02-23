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

from ..objid import createObjId
from ..hdf5dtype import getTypeItem
from ..array_util import bytesArrayToList
from ..h5reader import H5Reader


class H5pyReader(H5Reader):
    """
    This class can be used by HDF5DB to read content from an HDF5 file (using h5py) 
    """

    def visit(self, path, obj):
        name = obj.__class__.__name__
        self.log.info(f"visit: {path} name: {name}")
        
        obj_id = createObjId(obj_type=name, root_id=self._root_id)  # create uuid

        self._id_map[obj_id] = obj        
        
        addr = h5py.h5o.get_info(obj.id).addr
        self._addr_map[addr] = obj_id


    def __init__(
        self,
        filepath,
        app_logger=None
    ):
        self._id_map = {}
        self._addr_map = {}
        """
        if app_logger:
            self.log = app_logger
        else:
            self.log = logging.getLogger()
        self._filepath = filepath
        """
        super().__init__(filepath, app_logger=app_logger)
        f = h5py.File(self._filepath)
        self._f = f
        self._root_id = createObjId(obj_type="groups")
        self._id_map[self._root_id] = f
        addr = h5py.h5o.get_info(f.id).addr
        self._addr_map[addr] = self._root_id
        f.visititems(self.visit)

    def close(self):
        if self._f:
            self._f.close()
            self._f = None

    def get_root_id(self):
        """ Return root id """
        return self._root_id
    
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
            committedType = self.getCommittedTypeItemByUuid(type_uuid)
            type_item = committedType["type"].copy()
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
        elif isinstance(type_item, dict) and type_item["class"] in ("H5T_OPAQUE"):
            # TBD - don't include data for OPAQUE until JSON serialization
            # issues are addressed
            include_data = False
        else:
            pass  # use include_data parameter

        if include_data:
            try:
                data = obj.attrs[name] 
            except TypeError:
                self.log.warning("type error reading attribute")

        if include_data and data is not None:
            item["value"] = bytesArrayToList(data)
             
        # timestamps will be added by getAttributeItem()
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

    
    def _getDataset(self, dset):     
        self.log.info(f"getDataset alias: [{dset.name}]")

        item = {"alias": dset.name}

        typeid = dset.id.get_type()
        if h5py.h5t.TypeID.committed(typeid):
            type_uuid = None
            addr = h5py.h5o.get_info(typeid).addr
            type_uuid = self.getObjIdByAddress(addr)
            committedType = self.getObjectByid(type_uuid)
            typeItem = committedType["type"]
            typeItem["id"] = type_uuid
        else:
            typeItem = getTypeItem(dset.dtype)
        item["type"] = typeItem
        
        shapeItem = {}
        if dset.shape is None:
            # new with h5py 2.6, null space datasets will return None for shape
            shapeItem["class"] = "H5S_NULL"
        elif len(dset.shape) == 0:
            shapeItem["class"] = "H5S_SCALAR"
        else:
            shapeItem["class"] = "H5S_SIMPLE"
            shapeItem["dims"] = list(dset.shape)
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
                shapeItem["maxdims"] = maxshape
        item["shape"] = shapeItem
        
        return item
    
    def getObjectById(self, obj_id, include_attrs=True, include_links=True):
        """ return object with given id """
        if obj_id not in self._id_map:
            raise KeyError(f"{obj_id} not found")
        h5obj = self._id_map[obj_id]
        if isinstance(h5obj, h5py.Group):
            obj_json = self._getGroup(h5obj, include_links=include_links)
        elif isinstance(h5obj, h5py.Dataset):
            obj_json = self._getDataset(h5obj)
        elif isinstance(h5obj, h5py.Datatype):
            obj_json = self._getDataType(h5obj)
        else:
            raise TypeError(f"unexpected object type: {type(h5obj)}")
        
        if include_attrs:
            attributes = self.getAttributes(obj_id)
            obj_json["attributes"] = attributes

        return obj_json


    def getDatasetValues(self, obj_id, slices=Ellipsis, format="json"):
        """
        Get values from dataset identified by obj_id.
        If a slices list or tuple is provided, it should have the same
        number of elements as the rank of the dataset.
        """
        pass

