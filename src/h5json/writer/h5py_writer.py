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

from ..objid import getCollectionForId
from ..hdf5dtype import createDataType
from ..array_util import jsonToArray

from .h5writer import H5Writer



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

        if append:
            self._mode = "a"
        else:
            self._mode = "w"

        self._f = None
        self._id_map = {}

    def _createGroup(self, parent, grp_json, name=None):
        """ create the group and any links it contains """
        grp = parent.create_group(name)
        if "links" in grp_json:
            grp_links = grp_json["links"]
            self._createLinks(grp, grp_links)
        

    def _createDataset(self, parent, dset_json, name=None):
        """ create a dataset object """

        type_item = dset_json["type"]
        dtype = createDataType(type_item)
        kwds = {"dtype": dtype}
        shape_json = dset_json["shape"]
        if shape_json["class"] == "H5S_NULL":
            # skip the shape keyword to create a null space dataset
            pass
        elif shape_json["class"] == "H5S_SCALAR":
            kwds["shape"] = ()
        else:
            kwds["shape"] = shape_json["dims"]
        parent.create_dataset(name, **kwds)


    def _createDatatype(self, parent, ctype_json, name=None):
        """ create a datatype object """

        type_item = ctype_json["type"]
        dtype = createDataType(type_item)
        parent[name] = dtype


    def _createLinks(self, parent, links_json):
        """ create links in the given group """
        for title in links_json:
            if title in parent:
                # TBD: this will do the wrong thing if the link tgt has changed
                continue
            link_json = links_json[title]
            link_class = link_json["class"]
            if link_class == "H5L_TYPE_SOFT":
                h5path = link_json["h5path"]
                parent[title] = h5py.SoftLink(h5path)
            elif link_class == "H5L_TYPE_EXTERNAL":
                h5path = link_json["h5path"]
                filename = link_json["file"]
                parent[title] = h5py.ExternalLink(filename, h5path)
            elif link_class == "H5L_TYPE_USER_DEFINED":
                self.log.warning("unable to create user-defined link: {title}")
            elif link_class == "H5L_TYPE_HARD":
                tgt_id = link_json["id"]
                if tgt_id in self._id_map:
                    tgt_path = self._id_map[tgt_id]
                    tgt_obj = parent[tgt_path]
                    parent[title] = tgt_obj
                else:
                    obj_json = self.db.getObjectById(tgt_id)
                    parent_path = parent.name
                    if parent_path[-1] != '/':
                        parent_path += '/'
                    self._id_map[tgt_id] = parent_path + title
                    collection = getCollectionForId(tgt_id)
                    kwds = {"name": title}
                    if collection == "groups":
                        tgt_obj = self._createGroup(parent, obj_json, **kwds)
                    elif collection == "datasets":
                        tgt_obj = self._createDataset(parent, obj_json, **kwds)
                    elif collection == "datatypes":
                        tgt_obj = self._createDatatype(parent, obj_json, **kwds)
                    else:
                        self.log.warning(f"unexpected collection: {collection}")
                        tgt_obj = None
                    if tgt_obj:
                        parent[title] = tgt_obj
            else:
                self.log.warning(f"unexpected link class: {link_class}")

    def createAttribute(self, obj, name, attr_json):
        """ add the given attribute to obj """

        dtype = createDataType(attr_json["type"])
        shape_json = attr_json["shape"]
        shape_class = shape_json["class"]
        if shape_class == "H5S_NULL":
            dims = None
        elif shape_class == "H5S_SCALAR":
            dims = ()
        else:
            dims = tuple(shape_json["dims"])

        if dims is None:
            obj.attrs[name] = h5py.Empty(dtype)
        else:
            json_value = attr_json["value"]
            arr = jsonToArray(dims, dtype, json_value)
            obj.attrs[name] = arr


    def createAttributes(self, obj, obj_json):
        """ create attributes """

        if "attributes" not in obj_json:
            # no attributes
            return
        
        attrs = obj_json["attributes"]
        for name in attrs:
            attr_json = attrs[name]
            self.createAttribute(obj, name, attr_json)


    def visitAttributes(self, path, obj):
        name = obj.__class__.__name__
        self.log.info(f"visit: {path} name: {name}")

        obj_json = self.db.getObjectByPath(path)
        self.createAttributes(obj, obj_json)

    def flush(self):
        """ Write dirty items """
        if not self.db:
            # no db set yet
            return
        
        root_id = self.db.root_id
        self._id_map[root_id] = "/"
        with h5py.File(self._filepath, mode=self._mode) as f:
            root_json = self.db.getObjectById(root_id)
            if "links" in root_json:
                root_links = root_json["links"]
                self._createLinks(f, root_links)
            # update attributes
            self.createAttributes(f, root_json)
            f.visititems(self.visitAttributes)
        self._mode = "a"  # use append mode for future updates

  
    def close(self):
        """ close storage handle """
        self.flush()

