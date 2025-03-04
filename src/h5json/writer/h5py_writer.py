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
from .. import filters
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
            self._createObjects(grp, grp_links)

    def _createDataset(self, parent, dset_json, name=None):
        """ create a dataset object """

        type_item = dset_json["type"]
        dtype = createDataType(type_item)
        kwargs = {"dtype": dtype}
        shape_json = dset_json["shape"]
        shape_class = shape_json["class"]
        if shape_class == "H5S_NULL":
            # skip the shape keyword to create a null space dataset
            pass
        elif shape_class == "H5S_SCALAR":
            kwargs["shape"] = ()
        else:
            kwargs["shape"] = shape_json["dims"]
        if "dcpl" in dset_json and shape_class != "H5S_NULL":
            creation_props = dset_json["dcpl"]
            if "fillValue" in creation_props:
                fillvalue = creation_props["fillValue"]
                if fillvalue and len(dtype) > 1 and type(fillvalue) in (list, tuple):
                    # for compound types, need to convert from list to dataset compatible element

                    if len(dtype) != len(fillvalue):
                        msg = "fillvalue has incorrect number of elements"
                        self.log.warning(msg)
                        raise ValueError(msg)
                    
                    fillvalue = jsonToArray((), dtype, fillvalue)

                kwargs["fillvalue"] = fillvalue

            if "trackTimes" in creation_props:
                kwargs["track_times"] = creation_props["trackTimes"]
            if "layout" in creation_props:
                layout = creation_props["layout"]
                if "dims" in layout:
                    kwargs["chunks"] = tuple(layout["dims"])
            if "filters" in creation_props:
                filter_props = creation_props["filters"]
                for filter_prop in filter_props:
                    if "id" not in filter_prop:
                        self.log.warning("filter id not provided")
                        continue
                    filter_id = filter_prop["id"]
                    if filter_id not in filters._HDF_FILTERS:
                        self.log.warning(f"unknown filter id: {filter_id} ignoring")
                        continue

                    hdf_filter = filters._HDF_FILTERS[filter_id]

                    self.log.info(f"got filter: {filter_id}")
                    if "alias" not in hdf_filter:
                        self.log.warning(f"unsupported filter id: {filter_id} ignoring")
                        continue

                    filter_alias = hdf_filter["alias"]
                    if not h5py.h5z.filter_avail(filter_id):
                        msg = "compression filter not available, filter: {filter_alias}, ignoring"
                        self.log.warning(msg)
                        continue
                    if filter_alias in filters._H5PY_COMPRESSION_FILTERS:
                        if kwargs.get("compression"):
                            msg = f"compression filter already set for {filter_alias}, ignoring"
                            self.log.info(msg)
                            continue

                        kwargs["compression"] = filter_alias
                        self.log.info("setting compression filter to: {filter_alias}")
                        if filter_alias == "gzip":
                            # check for an optional compression value
                            if "level" in filter_prop:
                                kwargs["compression_opts"] = filter_prop["level"]
                        elif filter_alias == "szip":
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
                    else:
                        if filter_alias == "shuffle":
                            kwargs["shuffle"] = True
                        elif filter_alias == "fletcher32":
                            kwargs["fletcher32"] = True
                        elif filter_alias == "scaleoffset":
                            if "scaleOffset" not in filter_prop:
                                msg = "No scale_offset provided for scale offset filter, ignoring"
                                self.log(msg)
                                continue
                            kwargs["scaleoffset"] = filter_prop["scaleOffset"]
                        else:
                            self.log.info(f"Unexpected filter name: {filter_alias}, ignoring")
                            
        parent.create_dataset(name, **kwargs)

    def _createDatatype(self, parent, ctype_json, name=None):
        """ create a datatype object """

        type_item = ctype_json["type"]
        dtype = createDataType(type_item)
        parent[name] = dtype


    def _createObjects(self, parent, links_json):
        """ create child object in the given group, recurse for any sub-groups """
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

    def updateDatasetValues(self, dset_id, dset):
        """ write any pending dataset values """
        dset_json = self.db.getObjectById(dset_id)
        if "updates" not in dset_json:
            return
        updates = dset_json["updates"]
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


    def createAttribute(self, obj, name, attr_json):
        """ add the given attribute to obj """
        print(f"h5py_writer.createAttribute {obj.name}: {name}")

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


    def updateAttributes(self, obj_id, obj):
        """ create/replace any modified attributes """

        obj_json = self.db.getObjectById(obj_id)
        
        if "attributes" not in obj_json:
            # no attributes
            return
        
        attrs = obj_json["attributes"]
        for name in attrs:
            attr_json = attrs[name]
            self.createAttribute(obj, name, attr_json)

 
    def flush(self):
        """ Write dirty items """
        if not self.db:
            # no db set yet
            return False
   
        self.log.info("h5py_writer.flush()")
        root_id = self.db.root_id
        self._id_map[root_id] = "/"
        with h5py.File(self._filepath, mode=self._mode) as f:
            root_json = self.db.getObjectById(root_id)
            if "links" in root_json:
                root_links = root_json["links"]
                self._createObjects(f, root_links)
            # update attributes, dataset values
            for obj_id in self._id_map:
                if self.db.is_dirty(obj_id):
                    h5path = self._id_map[obj_id]
                    obj = f[h5path]
                    self.updateAttributes(obj_id, obj)
                    self.updateDatasetValues(obj_id, obj)

        self._mode = "a"  # use append mode for future updates
        return True  # all objects written successfully

  
    def close(self):
        """ close storage handle """
        self.flush()

