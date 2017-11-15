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

from __future__ import absolute_import

import six

if six.PY3:
    unicode = str
    


"""
This class is used to manage UUID lookup tables for primary HDF objects (Groups, Datasets,
 and Datatypes).  For HDF5 files that are read/write, this information is managed within
 the file itself in the "__db__" group.  For read-only files, the data is managed in
 an external file (domain filename with ".db" extension).

 "___db__"  ("root" for read-only case)
    description: Group object (member of root group). Only objects below this group are used
            for UUID data
    members: "{groups}", "{datasets}", "{datatypes}", "{objects}", "{paths}"
    attrs: 'rootUUID': UUID of the root group

"{groups}"
    description: contains map of UUID->group objects
    members: hard link to each anonymous group (i.e. groups which are not
        linked to by anywhere else).  Link name is the UUID
    attrs: group reference (or path for read-only files) to the group (for non-
        anonymous groups).

"{datasets}"
    description: contains map of UUID->dataset objects
    members: hard link to each anonymous dataset (i.e. datasets which are not
        linked to by anywhere else).  Link name is the UUID
    attrs: dataset reference (or path for read-only files) to the dataset (for non-
        anonymous datasets).

"{dataset_props}:
    description contains dataset creation properties"
    members: sub-group with link name as UUID.  Sub-group attributes are the creation props

"{datatypes}"
    description: contains map of UUID->datatyped objects
    members: hard link to each anonymous datatype (i.e. datatypes which are not
        linked to by anywhere else).  Link name is the UUID
    attrs: datatype reference (or path for read-only files) to the datatype (for non-
        anonymous datatypes).

"{addr}"
    description: contains map of file offset to UUID.
    members: none
    attrs: map of file offset to UUID




"""
import errno
import time
import h5py
import numpy as np
import uuid
import os.path as op
import os
import json
import logging

from .hdf5dtype import getTypeItem, createDataType, getItemSize 

# global dictionary to direct back to the Hdf5db instance by filename
# (needed for visititems callback)
# Will break in multi-threaded context
_db = {}

UUID_LEN = 36  # length for uuid strings

# standard compress filters
_HDF_FILTERS = {
    1: {'class': 'H5Z_FILTER_DEFLATE', 'alias': 'gzip', 'options': ['level']},
    2: {'class': 'H5Z_FILTER_SHUFFLE', 'alias': 'shuffle'},
    3: {'class': 'H5Z_FILTER_FLETCHER32', 'alias': 'fletcher32'},
    4: {'class': 'H5Z_FILTER_SZIP', 'alias': 'szip', 'options': ['bitsPerPixel', 'coding', 'pixelsPerBlock', 'pixelsPerScanLine']},
    5: {'class': 'H5Z_FILTER_NBIT'},
    6: {'class': 'H5Z_FILTER_SCALEOFFSET', 'alias': 'scaleoffset', 'options': ['scaleType']},
    32000: {'class': 'H5Z_FILTER_LZF', 'alias': 'lzf'}
}

_HDF_FILTER_OPTION_ENUMS = {'coding': {h5py.h5z.SZIP_EC_OPTION_MASK: 'H5_SZIP_EC_OPTION_MASK',
                                       h5py.h5z.SZIP_NN_OPTION_MASK: 'H5_SZIP_NN_OPTION_MASK'},
                            'scaleType': {h5py.h5z.SO_FLOAT_DSCALE: 'H5Z_SO_FLOAT_DSCALE',
                                          h5py.h5z.SO_FLOAT_ESCALE: 'H5Z_SO_FLOAT_ESCALE',
                                          h5py.h5z.SO_INT: 'H5Z_SO_INT'}}

# h5py supported filters
_H5PY_FILTERS = {'gzip': 1,
                 'shuffle': 2,
                 'fletcher32': 3,
                 'szip': 4,
                 'scaleoffset': 6,
                 'lzf': 32000}

_H5PY_COMPRESSION_FILTERS = ("gzip", "lzf", "szip")


def visitObj(path, obj):
    hdf5db = _db[obj.file.filename]
    hdf5db.visit(path, obj)


class Hdf5db:

    @staticmethod
    def createHDF5File(filePath):
        # create an "empty" hdf5 file
        if op.isfile(filePath):
            raise IOError(errno.EEXIST, "Resource already exists")

        f = h5py.File(filePath, 'w')
        f.close()

    @staticmethod
    def getVersionInfo():
        versionInfo = {}
        versionInfo['hdf5-json-version'] = "1.1.1" # todo - have this auto-synch with package version
        versionInfo['h5py_version'] = h5py.version.version
        versionInfo['hdf5_version'] = h5py.version.hdf5_version
        return versionInfo

    def __init__(self, filePath, dbFilePath=None, readonly=False,
                 app_logger=None, root_uuid=None, update_timestamps=True,
                 userid=None):
        if app_logger:
            self.log = app_logger
        else:
            self.log = logging.getLogger()
        if len(filePath) == 0 or not op.isfile(filePath):
            raise IOError(errno.ENXIO, "file not found")
        if not h5py.is_hdf5(filePath):
            raise IOError(errno.EINVAL, "not an HDF5 file")

        mode = 'r'
        if readonly:
            self.readonly = True
        else:
            if not os.stat(filePath).st_mode & 0o200:
                # file is read-only
                self.readonly = True
            else:
                mode = 'r+'
                self.readonly = False
             

        self.log.info("init -- filePath: " + filePath + " mode: " + mode)

        self.update_timestamps = update_timestamps

        self.f = h5py.File(filePath, mode, libver='latest')

        self.root_uuid = root_uuid

        if self.readonly:
            # for read-only files, add a dot in front of the name to be used as
            # the db file.  This won't collide with actual data files, since
            # "." is not allowed as the first character in a domain name.
            if not dbFilePath:
                dirname = op.dirname(self.f.filename)
                basename = op.basename(self.f.filename)
                if len(dirname) > 0:
                    dbFilePath = dirname + '/.' + basename
                else:
                    dbFilePath = '.' + basename
            dbMode = 'r+'
            if not op.isfile(dbFilePath):
                dbMode = 'w'
            self.log.info("dbFilePath: " + dbFilePath + " mode: " + dbMode)
            self.dbf = h5py.File(dbFilePath, dbMode)
        else:
            self.dbf = None  # for read only
        # create a global reference to this class
        # so visitObj can call back
        _db[filePath] = self

    def __enter__(self):
        self.log.info('Hdf5db __enter')
        return self

    def __exit__(self, type, value, traceback):
        self.log.info('Hdf5db __exit')
        filename = self.f.filename
        self.f.flush()
        self.f.close()
        if self.dbf:
            self.dbf.flush()
            self.dbf.close()
        del _db[filename]

    def getTimeStampName(self, uuid, objType="object", name=None):
        ts_name = uuid
        if objType != "object":
            if len(name) == 0:
                self.log.error("empty name passed to setCreateTime")
                raise Exception("bad setCreateTimeParameter")
            if objType == "attribute":
                ts_name += "_attr:["
                ts_name += name
                ts_name += "]"
            elif objType == "link":
                ts_name += "_link:["
                ts_name += name
                ts_name += "]"
            else:
                msg = "Bad objType passed to setCreateTime"
                self.log.error(msg)
                raise IOError(errno.EIO, msg)
        return ts_name

    """
      setCreateTime - sets the create time timestamp for the
            given object.
        uuid - id of object
        objtype - one of "object", "link", "attribute"
        name - name (for attributes, links... ignored for objects)
        timestamp - time (otherwise current time will be used)

       returns - nothing

       Note - should only be called once per object
    """
    def setCreateTime(self, uuid, objType="object", name=None, timestamp=None):
        if not self.update_timestamps:
            return
        ctime_grp = self.dbGrp["{ctime}"]
        ts_name = self.getTimeStampName(uuid, objType, name)
        if timestamp is None:
            timestamp = time.time()
        if ts_name in ctime_grp.attrs:
            self.log.warning("modifying create time for object: " + ts_name)
        ctime_grp.attrs.create(ts_name, timestamp, dtype='int64')

    """
      getCreateTime - gets the create time timestamp for the
            given object.
        uuid - id of object
        objtype - one of "object", "link", "attribute"
        name - name (for attributes, links... ignored for objects)
        useRoot - if true, use the time value for root object as default

       returns - create time for object, or create time for root if not set
    """
    def getCreateTime(self, uuid, objType="object", name=None, useRoot=True):
        ctime_grp = self.dbGrp["{ctime}"]
        ts_name = self.getTimeStampName(uuid, objType, name)
        timestamp = None
        if ts_name in ctime_grp.attrs:
            timestamp = ctime_grp.attrs[ts_name]
        elif useRoot:
            # return root timestamp
            root_uuid = self.dbGrp.attrs["rootUUID"]
            if root_uuid in ctime_grp.attrs:
                timestamp = ctime_grp.attrs[root_uuid]
        return timestamp

    """
      setModifiedTime - sets the modified time timestamp for the
            given object.
        uuid - id of object
        objtype - one of "object", "link", "attribute"
        name - name (for attributes, links... ignored for objects)
        timestamp - time (otherwise current time will be used)

       returns - nothing

    """
    def setModifiedTime(self, uuid, objType="object", name=None, timestamp=None):
        if not self.update_timestamps:
            return
        mtime_grp = self.dbGrp["{mtime}"]
        ts_name = self.getTimeStampName(uuid, objType, name)
        if timestamp is None:
            timestamp = time.time()
        mtime_grp.attrs.create(ts_name, timestamp, dtype='int64')

    """
      getModifiedTime - gets the modified time timestamp for the
            given object.
        uuid - id of object
        objtype - one of "object", "link", "attribute"
        name - name (for attributes, links... ignored for objects)
        useRoot - if true, use the time value for root object as default

       returns - create time for object, or create time for root if not set
    """
    def getModifiedTime(self, uuid, objType="object", name=None, useRoot=True):
        mtime_grp = self.dbGrp["{mtime}"]
        ts_name = self.getTimeStampName(uuid, objType, name)
        timestamp = None
        if ts_name in mtime_grp.attrs:
            timestamp = mtime_grp.attrs[ts_name]
        else:
            # return create time if no modified time has been set
            ctime_grp = self.dbGrp["{ctime}"]
            if ts_name in ctime_grp.attrs:
                timestamp = ctime_grp.attrs[ts_name]
            elif useRoot:
                # return root timestamp
                root_uuid = self.dbGrp.attrs["rootUUID"]
                timestamp = mtime_grp.attrs[root_uuid]
        return timestamp

    """
      getAclGroup - return the db group "{acl}" if present,
        otherwise return None
    """
    def getAclGroup(self, create=False):
        if not self.dbGrp:
            return None  # file not initialized
        if "{acl}" in self.dbGrp:
            return self.dbGrp["{acl}"]
        if not create:
            return None
        self.dbGrp.create_group("{acl}")
        return self.dbGrp["{acl}"]

    """
      getAclDtype - return detype for ACL
    """
    def getAclDtype(self):
        fields = []
        fields.append(('userid', np.int32))
        fields.append(('create', np.int8))
        fields.append(('read', np.int8))
        fields.append(('update', np.int8))
        fields.append(('delete', np.int8))
        fields.append(('readACL', np.int8))
        fields.append(('updateACL', np.int8))
        dt = np.dtype(fields)
        return dt

    """
      getAclDataset - return ACL datset for given uuid
    """
    def getAclDataset(self, obj_uuid, create=False):
        acl_group = self.getAclGroup(create=create)

        if acl_group is None:
            return None

        if obj_uuid in acl_group:
            return acl_group[obj_uuid]

        if not create:
            return None

        # create dataset
        dt = self.getAclDtype()
        acl_group.create_dataset(obj_uuid, (0,), dtype=dt, maxshape=(None,))
        return acl_group[obj_uuid]

    """
      getNumAcls - return number of acls associatted with given uuid
    """
    def getNumAcls(self, obj_uuid):
        acl_group = self.getAclGroup()
        if acl_group is None:
            return 0
        if obj_uuid not in acl_group:
            return 0
        acls = acl_group[obj_uuid]
        return acls.shape[0]

    """
      convertAclNdArrayToDict - helper function - return acl item to dict
    """
    def convertAclNdArrayToDict(self, acl_ndarray):
        fields = acl_ndarray.dtype.fields.keys()
        acl = {}
        for field in fields:
            value = int(acl_ndarray[field])
            acl[field] = value
        return acl

    """
      Get default acl - returns dict obj
    """
    def getDefaultAcl(self):
        dt = self.getAclDtype()
        acl = {}
        for field in dt.fields.keys():
            if field == 'userid':
                acl[field] = 0
            else:
                acl[field] = 1  # default is allowed
        return acl

    """
      getAcl - return ACL for given uuid and userid
        returns ACL associated with the given uuid, or if none exists,
        the ACL associatted with the root group.

        If an ACL is not present for a userid/obj and ACL will be returned
        via the following precedence:

        1) obj_uuid, user_id
        2) root_uuid, user_id
        3) obj_uuid, 0
        4) root_uuid, 0
        5) 'all perm' ACL
    """
    def getAcl(self, obj_uuid, userid):
        acl_grp = self.getAclGroup()

        if acl_grp is not None:
            acl = self.getAclByObjAndUser(obj_uuid, userid)
            if acl is not None:
                return acl

            if obj_uuid != self.root_uuid and userid != 0:
                # get the root acl for this user
                acl = self.getAclByObjAndUser(self.root_uuid, userid)
                if acl is not None:
                    return acl

            if userid != 0:
                # get acl for default user
                acl = self.getAclByObjAndUser(obj_uuid, 0)
                if acl is not None:
                    return acl

            if obj_uuid != self.root_uuid:
                # get root acl for default user
                acl = self.getAclByObjAndUser(self.root_uuid, 0)
                if acl is not None:
                    return acl

        # create an ACL with default permissions
        acl = self.getDefaultAcl()

        return acl

    """
      get ACL for specific uuid and user
         return None if not found
    """
    def getAclByObjAndUser(self, obj_uuid, userid):

        acl = None
        acl_dset = self.getAclDataset(obj_uuid)

        if acl_dset:
            # iterate through elements, looking for user_id
            acls = acl_dset[...]
            num_acls = acl_dset.shape[0]
            acl = None
            for i in range(num_acls):
                item = acls[i]
                if item['userid'] == userid:
                    acl = item
                    break

        if acl is not None:
            acl = self.convertAclNdArrayToDict(acl)
        return acl

    """
      getAcls - get all acls for given uuid
    """

    def getAcls(self, obj_uuid):

        acls = []
        acl_dset = self.getAclDataset(obj_uuid)

        if acl_dset:
            # iterate through elements, looking for user_id
            num_acls = acl_dset.shape[0]

            for i in range(num_acls):
                item = acl_dset[i]
                acl = self.convertAclNdArrayToDict(item)
                acls.append(acl)

        return acls

    """
      setAcl -  set the acl for given uuid.
    """
    def setAcl(self, obj_uuid, acl):
        acl_dset = self.getAclDataset(obj_uuid, create=True)

        if acl_dset is None:
            msg = "Unexpected error acl not created for uuid:[" + obj_uuid + "]"
            self.log.error(msg)
            raise IOError(errno.EIO, msg)

        userid = acl['userid']

        # iterate through elements, looking for user_id
        acls = acl_dset[...]
        num_acls = acl_dset.shape[0]

        user_index = None

        for i in range(num_acls):
            item = acls[i]
            if item['userid'] == userid:
                # update this element
                user_index = i
                break

        if user_index is None:
            # userid not found - add row
            acl_dset.resize(((num_acls+1),))
            user_index = num_acls

        # update the acl dataset
        item = acl_dset[user_index]
        for field in acl.keys():
            item[field] = acl[field]
        acl_dset[user_index] = item  # save back to the file

    def initFile(self):
        # self.log.info("initFile")
        if self.readonly:
            self.dbGrp = self.dbf
            if "{groups}" in self.dbf:
                # file already initialized
                self.root_uuid = self.dbGrp.attrs["rootUUID"]
                return

        else:
            if "__db__" in self.f:
                # file already initialized
                self.dbGrp = self.f["__db__"]
                self.root_uuid = self.dbGrp.attrs["rootUUID"]
                return  # already initialized
            self.dbGrp = self.f.create_group("__db__")

        self.log.info("initializing file")
        if not self.root_uuid:
            self.root_uuid = str(uuid.uuid1())
        self.dbGrp.attrs["rootUUID"] = self.root_uuid
        self.dbGrp.create_group("{groups}")
        self.dbGrp.create_group("{datasets}")
        self.dbGrp.create_group("{datatypes}")
        self.dbGrp.create_group("{addr}") # store object address
        self.dbGrp.create_group("{ctime}") # stores create timestamps
        self.dbGrp.create_group("{mtime}") # store modified timestamps

        mtime = op.getmtime(self.f.filename)
        ctime = mtime
        self.setCreateTime(self.root_uuid, timestamp=ctime)
        self.setModifiedTime(self.root_uuid, timestamp=mtime)

        self.f.visititems(visitObj)

    def visit(self, path, obj):
        name = obj.__class__.__name__
        if len(path) >= 6 and path[:6] == '__db__':
            return  # don't include the db objects
        self.log.info('visit: ' + path + ' name: ' + name)
        col = None
        if name == 'Group':
            col = self.dbGrp["{groups}"].attrs
        elif name == 'Dataset':
            col = self.dbGrp["{datasets}"].attrs
        elif name == 'Datatype':
            col = self.dbGrp["{datatypes}"].attrs
        else:
            msg = "Unknown object type: " + __name__ + " found during scan of HDF5 file"
            self.log.error(msg)
            raise IOError(errno.EIO, msg)
        uuid1 = uuid.uuid1()  # create uuid
        id = str(uuid1)
        addrGrp = self.dbGrp["{addr}"]
        if not self.readonly:
            # storing db in the file itself, so we can link to the object directly
            col[id] = obj.ref  # save attribute ref to object
        else:
            #store path to object
            col[id] = obj.name
        addr = h5py.h5o.get_info(obj.id).addr
        # store reverse map as an attribute
        addrGrp.attrs[str(addr)] = id

    #
    # Get Datset creation properties
    #
    def getDatasetCreationProps(self, dset_uuid):
        prop_list = {}
        if "{dataset_props}" not in self.dbGrp:
            # no, group, so no properties
            return prop_list  # return empty dict
        dbPropsGrp = self.dbGrp["{dataset_props}"]

        if dset_uuid not in dbPropsGrp.attrs:
            return prop_list  # return empty dict
        prop_str = dbPropsGrp.attrs[dset_uuid]
        # expand json string
        try:
            prop_list = json.loads(prop_str)
        except ValueError as ve:
            msg = "Unable to load creation properties for dataset:[" + dset_uuid + "]: " + ve.message
            self.log.error(msg)
            raise IOError(errno.EIO, msg)

        # fill in Filter class values
        if 'filters' in prop_list:
            prop_filters = prop_list['filters']
            for prop_filter in prop_filters:
                if 'class' not in prop_filter:
                    filter_id = prop_filter['id']
                    if filter_id in _HDF_FILTERS:
                        hdf_filter = _HDF_FILTERS[filter_id]
                        prop_filter['class'] = hdf_filter['class']
                    else:
                        prop_filter['class'] = 'H5Z_FILTER_USER'

        return prop_list

    #
    # Set dataset creation property
    #
    def setDatasetCreationProps(self, dset_uuid, prop_dict):
        self.log.info('setDataProp([' + dset_uuid + ']')
        if not prop_dict:
            # just ignore if empty dictionary
            return
        if "{dataset_props}" not in self.dbGrp:
            self.dbGrp.create_group("{dataset_props}")
        dbPropsGrp = self.dbGrp["{dataset_props}"]
        if dset_uuid in dbPropsGrp.attrs:
            # this should be write once
            msg = "Unexpected error setting dataset creation properties for dataset:[" + dset_uuid + "]"
            self.log.error(msg)
            raise IOError(errno.EIO, msg)
        prop_str = json.dumps(prop_dict)
        dbPropsGrp.attrs[dset_uuid] = prop_str

    def getUUIDByAddress(self, addr):
        if "{addr}" not in self.dbGrp:
            self.log.error("expected to find {addr} group")
            return None
        addrGrp = self.dbGrp["{addr}"]
        obj_uuid = None
        if str(addr) in addrGrp.attrs:
            obj_uuid = addrGrp.attrs[str(addr)]
        if obj_uuid and type(obj_uuid) is not str:
            # convert bytes to unicode
            obj_uuid = obj_uuid.decode('utf-8')
        return obj_uuid

    """
     Get the number of links in a group to an object
    """
    def getNumLinksToObjectInGroup(self, grp, obj):
        objAddr = h5py.h5o.get_info(obj.id).addr
        numLinks = 0
        for name in grp:
            try:
                child = grp[name]
            except KeyError:
                # UDLink? Ignore for now
                self.log.info("ignoring link (UDLink?): " + name)
                continue

            addr = h5py.h5o.get_info(child.id).addr
            if addr == objAddr:
                numLinks = numLinks + 1

        return numLinks

    """
     Get the number of links to the given object
    """
    def getNumLinksToObject(self, obj):
        self.initFile()
        groups = self.dbGrp["{groups}"]
        numLinks = 0
        # iterate through each group in the file and unlink tgt if it is linked
        # by the group
        for uuidName in groups:
            # iterate through anonymous groups
            grp = groups[uuidName]
            nLinks = self.getNumLinksToObjectInGroup(grp, obj)
            if nLinks > 0:
                numLinks += nLinks
        for uuidName in groups.attrs:
            # now non anonymous groups
            grpRef = groups.attrs[uuidName]
            grp = self.f[grpRef]  # dereference
            nLinks = self.getNumLinksToObjectInGroup(grp, obj)
            if nLinks > 0:
                numLinks += nLinks
        # finally, check the root group
        root = self.getObjByPath("/")
        nLinks = self.getNumLinksToObjectInGroup(root, obj)
        numLinks += nLinks

        return numLinks

    def getUUIDByPath(self, path):
        self.initFile()
        self.log.info("getUUIDByPath: [" + path + "]")
        if len(path) >= 6 and path[:6] == '__db__':
            msg = "getUUIDByPath called with invalid path: [" + path + "]"
            self.log.error(msg)
            raise IOError(errno.EIO, msg)
        if path == '/':
            # just return the root UUID
            root_uuid = self.dbGrp.attrs["rootUUID"]
            if root_uuid and type(root_uuid) is not str:
                # convert bytes to unicode
                root_uuid = root_uuid.decode('utf-8')
            return root_uuid

        obj = self.f[path]  # will throw KeyError if object doesn't exist
        addr = h5py.h5o.get_info(obj.id).addr
        obj_uuid = self.getUUIDByAddress(addr)
        return obj_uuid

    def getObjByPath(self, path):
        if len(path) >= 6 and path[:6] == '__db__':
            return None # don't include the db objects
        obj = self.f[path]  # will throw KeyError if object doesn't exist
        return obj

    def getObjectByUuid(self, col_type, obj_uuid):
        #col_type should be either "datasets", "groups", or "datatypes"
        if col_type not in ("datasets", "groups", "datatypes"):
            msg = "Unexpectd error, invalid col_type: [" + col_type + "]"
            self.log.error(msg)
            raise IOError(errno.EIO, msg)
        if col_type == "groups" and obj_uuid == self.dbGrp.attrs["rootUUID"]:
            return self.f['/']  # returns root group

        obj = None  # Group, Dataset, or Datatype
        col_name = '{' + col_type + '}'
        # get the collection group for this collection type
        col = self.dbGrp[col_name]
        if obj_uuid in col.attrs:
            ref = col.attrs[obj_uuid]
            obj = self.f[ref]  # this works for read-only as well
        elif obj_uuid in col:
            # anonymous object
            obj = col[obj_uuid]

        return obj

    def getDatasetObjByUuid(self, obj_uuid):
        self.initFile()
        self.log.info("getDatasetObjByUuid(" + obj_uuid + ")")

        obj = self.getObjectByUuid("datasets", obj_uuid)

        return obj

    def getGroupObjByUuid(self, obj_uuid):
        self.initFile()
        self.log.info("getGroupObjByUuid(" + obj_uuid + ")")

        obj = self.getObjectByUuid("groups", obj_uuid)

        return obj

    def getDatasetTypeItemByUuid(self, obj_uuid):
        dset = self.getDatasetObjByUuid(obj_uuid)  # throws exception if not found
        item = { 'id': obj_uuid }
        item['type'] = getTypeItem(dset.dtype)
        if self.update_timestamps:
            item['ctime'] = self.getCreateTime(obj_uuid)
            item['mtime'] = self.getModifiedTime(obj_uuid)

        return item

    """
    getNullReference - return a null object reference
    """
    def getNullReference(self):
        tmpGrp = None
        if "{tmp}" not in self.dbGrp:
            tmpGrp = self.dbGrp.create_group("{tmp}")
        else:
            tmpGrp = self.dbGrp["{tmp}"]
        if 'nullref' not in tmpGrp:
            dt = h5py.special_dtype(ref=h5py.Reference)
            tmpGrp.create_dataset('nullref', (1,), dtype=dt)
        nullref_dset = tmpGrp['nullref']
        return nullref_dset[0]

    """
    getNullRegionReference - return a null region reference
    """
    def getNullRegionReference(self):
        tmpGrp = None
        if "{tmp}" not in self.dbGrp:
            tmpGrp = self.dbGrp.create_group("{tmp}")
        else:
            tmpGrp = self.dbGrp["{tmp}"]
            if 'nullregref' not in tmpGrp:
                dt = h5py.special_dtype(ref=h5py.RegionReference)
                tmpGrp.create_dataset('nullregref', (1,), dtype=dt)
                nullregref_dset = tmpGrp['nullregref']
                return nullregref_dset[0]

    def getShapeItemByDsetObj(self, obj):
        item = {}
        if obj.shape is None:
            # new with h5py 2.6, null space datasets will return None for shape
            item['class'] = 'H5S_NULL'
        elif len(obj.shape) == 0:
            # check to see if this is a null space vs a scalar dataset we'll do
            # this by seeing if an exception is raised when reading the dataset
            # h5py issue https://github.com/h5py/h5py/issues/279 will provide a
            # better way to determine null spaces
            # Update 3/10/17: Above issue is closed, but waiting on 2.7 final release
            try:
                val = obj[...]
                if val is None:
                    self.log.warning("no value returned for scalar dataset")
                item['class'] = 'H5S_SCALAR'
            except IOError:
                item['class'] = 'H5S_NULL'
        else:
            item['class'] = 'H5S_SIMPLE'
            item['dims'] = obj.shape
            maxshape = []
            include_maxdims = False
            for i in range(len(obj.shape)):
                extent = 0
                if len(obj.maxshape) > i:
                    extent = obj.maxshape[i]
                    if extent is None:
                        extent = 0
                    if extent > obj.shape[i] or extent == 0:
                        include_maxdims = True
                maxshape.append(extent)
            if include_maxdims:
                item['maxdims'] = maxshape
        return item

    def getShapeItemByAttrObj(self, obj):
        item = {}
        if obj.shape is None or obj.get_storage_size() == 0:
            # If storage size is 0, assume this is a null space obj
            # See: h5py issue https://github.com/h5py/h5py/issues/279
            item['class'] = 'H5S_NULL'
        else:
            if obj.shape:
                item['class'] = 'H5S_SIMPLE'
                item['dims'] = obj.shape
            else:
                item['class'] = 'H5S_SCALAR'
        return item

    #
    # Get dataset creation properties maintained by HDF5 library
    #
    def getHDF5DatasetCreationProperties(self, obj_uuid, type_class):
        dset = self.getDatasetObjByUuid(obj_uuid)
        #
        # Fill in creation properties
        #
        creationProps = {}
        plist = h5py.h5d.DatasetID.get_create_plist(dset.id)

        # alloc time
        nAllocTime = plist.get_alloc_time()
        if nAllocTime == h5py.h5d.ALLOC_TIME_DEFAULT:
            creationProps['allocTime'] = 'H5D_ALLOC_TIME_DEFAULT'
        elif nAllocTime == h5py.h5d.ALLOC_TIME_LATE:
            creationProps['allocTime'] = 'H5D_ALLOC_TIME_LATE'
        elif nAllocTime == h5py.h5d.ALLOC_TIME_EARLY:
            creationProps['allocTime'] = 'H5D_ALLOC_TIME_EARLY'
        elif nAllocTime == h5py.h5d.ALLOC_TIME_INCR:
            creationProps['allocTime'] = 'H5D_ALLOC_TIME_INCR'
        else:
            self.log.warning("Unknown alloc time value: " + str(nAllocTime))

        # fill time
        nFillTime = plist.get_fill_time()
        if nFillTime == h5py.h5d.FILL_TIME_ALLOC:
            creationProps['fillTime'] = 'H5D_FILL_TIME_ALLOC'
        elif nFillTime == h5py.h5d.FILL_TIME_NEVER:
            creationProps['fillTime'] = 'H5D_FILL_TIME_NEVER'
        elif nFillTime == h5py.h5d.FILL_TIME_IFSET:
            creationProps['fillTime'] = 'H5D_FILL_TIME_IFSET'
        else:
            self.log.warning("unknown fill time value: " + str(nFillTime))

        if type_class not in ('H5T_VLEN', 'H5T_OPAQUE'):
            if plist.fill_value_defined() == h5py.h5d.FILL_VALUE_USER_DEFINED:
                creationProps['fillValue'] =  self.bytesArrayToList(dset.fillvalue)

        # layout
        nLayout = plist.get_layout()
        if nLayout == h5py.h5d.COMPACT:
            creationProps['layout'] = {'class': 'H5D_COMPACT'}
        elif nLayout == h5py.h5d.CONTIGUOUS:
            creationProps['layout'] = {'class': 'H5D_CONTIGUOUS'}
        elif nLayout == h5py.h5d.CHUNKED:
            creationProps['layout'] = {'class': 'H5D_CHUNKED', 'dims': dset.chunks }
        else:
            self.log.warning("Unknown layout value:" + str(nLayout))

        num_filters = plist.get_nfilters()
        filter_props = []
        if num_filters:
            for n in range(num_filters):
                filter_info = plist.get_filter(n)
                opt_values = filter_info[2]
                filter_prop = {}
                filter_id = filter_info[0]
                filter_prop['id'] = filter_id
                if filter_info[3]:
                    filter_prop['name'] = self.bytesArrayToList(filter_info[3])
                if filter_id in _HDF_FILTERS:
                    hdf_filter = _HDF_FILTERS[filter_id]
                    filter_prop['class'] = hdf_filter['class']
                    if 'options' in hdf_filter:
                        filter_opts = hdf_filter['options']
                        for i in range(len(filter_opts)):
                            if len(opt_values) <= i:
                                break  # end of option values
                            opt_value = opt_values[i]
                            opt_value_enum = None
                            option_name = filter_opts[i]
                            if option_name in _HDF_FILTER_OPTION_ENUMS:
                                option_enums = _HDF_FILTER_OPTION_ENUMS[option_name]
                                if opt_value in option_enums:
                                    opt_value_enum = option_enums[opt_value]
                            if opt_value_enum:
                                filter_prop[option_name] = opt_value_enum
                            else:
                                filter_prop[option_name] = opt_value
                else:
                    # custom filter
                    filter_prop['class'] = 'H5Z_FILTER_USER'
                    if opt_values:
                        filter_prop['parameters'] = opt_values
                filter_props.append(filter_prop)
            creationProps['filters'] = filter_props

        return creationProps

    #
    # Get dataset information - type, shape, num attributes, creation properties
    #
    def getDatasetItemByUuid(self, obj_uuid):
        dset = self.getDatasetObjByUuid(obj_uuid)
        if dset is None:
            if self.getModifiedTime(obj_uuid, useRoot=False):
                msg = "Dataset with uuid: " + obj_uuid + " has been previously deleted"
                self.log.info(msg)
                raise IOError(errno.ENOENT, msg)
            else:
                msg = "Dataset with uuid: " + obj_uuid + " was not found"
                self.log.info(msg)
                raise IOError(errno.ENXIO, msg)

        # fill in the item info for the dataset
        item = { 'id': obj_uuid }

        alias = []
        if dset.name and not dset.name.startswith("/__db__"):
            alias.append(dset.name)   # just use the default h5py path for now
        item['alias'] = alias

        item['attributeCount'] = len(dset.attrs)

        # check if the dataset is using a committed type
        typeid = h5py.h5d.DatasetID.get_type(dset.id)
        typeItem = None
        if h5py.h5t.TypeID.committed(typeid):
            type_uuid = None
            addr = h5py.h5o.get_info(typeid).addr
            type_uuid = self.getUUIDByAddress(addr)
            committedType = self.getCommittedTypeItemByUuid(type_uuid)
            typeItem = committedType['type']
            typeItem['uuid'] = type_uuid
        else:
            typeItem = getTypeItem(dset.dtype)

        item['type'] = typeItem

        # get shape
        item['shape'] = self.getShapeItemByDsetObj(dset)

        if self.update_timestamps:
            item['ctime'] = self.getCreateTime(obj_uuid)
            item['mtime'] = self.getModifiedTime(obj_uuid)

        creationProps = self.getDatasetCreationProps(obj_uuid)
        if creationProps:
            # if chunks is not in the db props, add it from the dataset prop
            # (so auto-chunk values can be returned)
            if dset.chunks and 'layout' not in creationProps:
                creationProps['layout'] = {'class': 'H5D_CHUNKED',
                                           'dims': dset.chunks}
        else:
            # no db-tracked creation properties, pull properties from library
            creationProps = self.getHDF5DatasetCreationProperties(obj_uuid, typeItem['class'])

        if creationProps:
            item['creationProperties'] = creationProps

        return item

    """
    createTypeFromItem - create type given dictionary definition
    """
    def createTypeFromItem(self, attr_type):
        dt = None

        if type(attr_type) in (six.text_type, six.binary_type) and len(attr_type) == UUID_LEN:
            # assume attr_type is a uuid of a named datatype
            tgt = self.getCommittedTypeObjByUuid(attr_type)
            if tgt is None:
                msg = "Unable to create attribute, committed type with uuid of: " + attr_type + " not found"
                self.log.info(msg)
                raise IOError(errno.ENXIO, msg)
            dt = tgt  # can use the object as the dt parameter
        else:
            try:
                dt = createDataType(attr_type)
            except KeyError as ke:
                msg = "Unable to create type: " + ke.message
                self.log.info(msg)
                raise IOError(errno.EINVAL, msg)
            except TypeError as te:
                msg = "Unable to create type: " + str(te)
                self.log.info(msg)
                raise IOError(errno.EINVAL, msg)
            if dt is None:
                msg = "Unexpected error creating type"
                self.log.error(msg)
                raise IOError(errno, errno.EIO, msg)
        return dt

    """
    createCommittedType - creates new named datatype
    Returns item
    """
    def createCommittedType(self, datatype, obj_uuid=None):
        self.log.info("createCommittedType")
        self.initFile()
        if self.readonly:
            msg = "Can't create committed type (updates are not allowed)"
            self.log.info(msg)
            raise IOError(errno.EPERM, msg)
        datatypes = self.dbGrp["{datatypes}"]
        if not obj_uuid:
            obj_uuid = str(uuid.uuid1())
        dt = self.createTypeFromItem(datatype)

        datatypes[obj_uuid] = dt

        if obj_uuid not in datatypes:
            msg = "Unexpected failure to create committed datatype"
            self.log.error(msg)
            raise IOError(errno.EIO, msg)
        newType = datatypes[obj_uuid] # this will be a h5py Datatype class
        # store reverse map as an attribute
        addr = h5py.h5o.get_info(newType.id).addr
        addrGrp = self.dbGrp["{addr}"]
        addrGrp.attrs[str(addr)] = obj_uuid
        # set timestamp
        now = time.time()
        self.setCreateTime(obj_uuid, timestamp=now)
        self.setModifiedTime(obj_uuid, timestamp=now)
        item = { 'id': obj_uuid }
        item['attributeCount'] = len(newType.attrs)
        #item['type'] = hdf5dtype.getTypeItem(datatype.dtype)
        if self.update_timestamps:
            item['ctime'] = self.getCreateTime(obj_uuid)
            item['mtime'] = self.getModifiedTime(obj_uuid)
        return item

    """
    getCommittedTypeObjByUuid - get obj from {datatypes} collection
    Returns type obj
    """
    def getCommittedTypeObjByUuid(self, obj_uuid):
        self.log.info("getCommittedTypeObjByUuid(" + obj_uuid + ")")
        self.initFile()
        datatype = None
        datatypesGrp = self.dbGrp["{datatypes}"]
        if obj_uuid in datatypesGrp.attrs:
            typeRef = datatypesGrp.attrs[obj_uuid]
            # typeRef could be a reference or (for read-only) a path
            datatype = self.f[typeRef]
        elif obj_uuid in datatypesGrp:
            datatype = datatypesGrp[obj_uuid]  # non-linked type
        else:
            msg = "Committed datatype: " + obj_uuid + " not found"
            self.log.info(msg)

        return datatype

    """
    getCommittedTypeItemByUuid - get json from {datatypes} collection
    Returns type obj
    """
    def getCommittedTypeItemByUuid(self, obj_uuid):
        self.log.info("getCommittedTypeItemByUuid(" + obj_uuid + ")")
        self.initFile()
        datatype = self.getCommittedTypeObjByUuid(obj_uuid)

        if datatype is None:
            if self.getModifiedTime(obj_uuid, useRoot=False):
                msg = "Datatype with uuid: " + obj_uuid + " has been previously deleted"
                self.log.info(msg)
                raise IOError(errno.ENOENT, msg)
            else:
                msg = "Datatype with uuid: " + obj_uuid + " was not found"
                self.log.info(msg)
                raise IOError(errno.ENXIO, msg)

        item = { 'id': obj_uuid }
        alias = []
        if datatype.name and not datatype.name.startswith("/__db__"):
            alias.append(datatype.name)   # just use the default h5py path for now
        item['alias'] = alias
        item['attributeCount'] = len(datatype.attrs)
        item['type'] = getTypeItem(datatype.dtype)
        if self.update_timestamps:
            item['ctime'] = self.getCreateTime(obj_uuid)
            item['mtime'] = self.getModifiedTime(obj_uuid)

        return item

    """
      Get attribute given an object and name
      returns: JSON object
    """
    def getAttributeItemByObj(self, obj, name, includeData=True):
        if name not in obj.attrs:
            msg = "Attribute: [" + name + "] not found in object: " + obj.name
            self.log.info(msg)
            return None

        # get the attribute!
        attrObj = h5py.h5a.open(obj.id, np.string_(name))
        attr = None

        item = { 'name': name }

        # check if the dataset is using a committed type
        typeid = attrObj.get_type()
        typeItem = None
        if h5py.h5t.TypeID.committed(typeid):
            type_uuid = None
            addr = h5py.h5o.get_info(typeid).addr
            type_uuid = self.getUUIDByAddress(addr)
            committedType = self.getCommittedTypeItemByUuid(type_uuid)
            typeItem = committedType['type']
            typeItem['uuid'] = type_uuid
        else:
            typeItem = getTypeItem(attrObj.dtype)
        item['type'] = typeItem
        # todo - don't include data for OPAQUE until JSON serialization
        # issues are addressed

        if type(typeItem) == dict and typeItem['class'] in ('H5T_OPAQUE'):
            includeData = False

        shape_json = self.getShapeItemByAttrObj(attrObj)
        item['shape'] = shape_json
        if shape_json['class'] == 'H5S_NULL':
            includeData = False
        if includeData:
            try:
                attr = obj.attrs[name]  # returns a numpy array
            except TypeError:
                self.log.warning("type error reading attribute")

        if includeData and attr is not None:
            if shape_json['class'] == 'H5S_SCALAR':
                data = self.getDataValue(typeItem, attr)
            else:
                dims = shape_json["dims"]
                rank = len(dims)
                # convert numpy object to python list
                # values = self.toList(typeItem, attr)
                data = self.toList(rank, typeItem, attr)
            #data = self.bytesToString(data)
            item['value'] = data
        # timestamps will be added by getAttributeItem()
        return item

    def getAttributeItems(self, col_type, obj_uuid, marker=None, limit=0):
        self.log.info("db.getAttributeItems(" + obj_uuid + ")")
        if marker:
            self.log.info("...marker: " + marker)
        if limit:
            self.log.info("...limit: " + str(limit))

        self.initFile()
        obj = self.getObjectByUuid(col_type, obj_uuid)
        if obj is None:
            msg = "Object: " + obj_uuid + " could not be loaded"
            self.log.info(msg)
            raise IOError(errno.ENXIO, msg)

        items = []
        gotMarker = True
        if marker is not None:
            gotMarker = False
        count = 0
        for name in obj.attrs:
            if not gotMarker:
                if name == marker:
                    gotMarker = True
                    continue  # start filling in result on next pass
                else:
                    continue  # keep going!
            item = self.getAttributeItemByObj(obj, name, False)
            # mix-in timestamps
            if self.update_timestamps:
                item['ctime'] = self.getCreateTime(obj_uuid, objType="attribute", name=name)
                item['mtime'] = self.getModifiedTime(obj_uuid, objType="attribute", name=name)

            items.append(item)
            count += 1
            if limit > 0 and count == limit:
                break  # return what we got
        return items

    def getAttributeItem(self, col_type, obj_uuid, name):
        self.log.info("getAttributeItemByUuid(" + col_type + ", " + obj_uuid
                      + ", " + name + ")")
        self.initFile()
        obj = self.getObjectByUuid(col_type, obj_uuid)
        if obj is None:
            msg = "Parent object: " + obj_uuid + " of attribute not found"
            self.log.info(msg)
            raise IOError(errno.ENXIO, msg)
            return None
        item = self.getAttributeItemByObj(obj, name)
        if item is None:
            if self.getModifiedTime(obj_uuid, objType="attribute", name=name, useRoot=False):
                # attribute has been removed
                msg = "Attribute: [" + name + "] of object: " + obj_uuid + " has been previously deleted"
                self.log.info(msg)
                raise IOError(errno.ENOENT, msg)
            msg = "Attribute: [" + name + "] of object: " + obj_uuid + " not found"
            self.log.info(msg)
            raise IOError(errno.ENXIO, msg)
        # mix-in timestamps
        if self.update_timestamps:
            item['ctime'] = self.getCreateTime(obj_uuid, objType="attribute", name=name)
            item['mtime'] = self.getModifiedTime(obj_uuid, objType="attribute", name=name)

        return item

    """
    isDimensionList - return True if this attribute json looks like a dimension list
    """
    def isDimensionList(self, attr_name, attr_type):
        if attr_name != "DIMENSION_LIST":
            return False
        if type(attr_type) is not dict:
            return False
        if attr_type['class'] != "H5T_VLEN":
            return False
        base_type = attr_type['base']
        if base_type['class'] != 'H5T_REFERENCE':
            return False
        return True

    """
    isReferenceList - return True if this attribute json looks like a reference list
    """
    def isReferenceList(self, attr_name, attr_type):
        if attr_name != "REFERENCE_LIST":
            return False
        if type(attr_type) is not dict:
            return False
        if attr_type['class'] != "H5T_COMPOUND":
            return False

        return True

    """
     makeDimensionList - work-around for h5py problems saving dimension list -
        types which are vlen's of references are not working directly, so use dim_scale api
        Note: this is a work-around for h5py issue:
         https://github.com/h5py/h5py/issues/553
    """
    def makeDimensionList(self, obj, shape, value):
        dset_refs = self.listToRef(value)
        for i in range(len(dset_refs)):
            refs = dset_refs[i]
            if type(refs) not in (list, tuple):
                msg = "Invalid dimension list value"
                self.log.info(msg)
                raise IOError(errno.EINVAL, msg)
            for j in range(len(refs)):
                scale_obj = self.f[refs[j]]
                if scale_obj is None:
                    self.log.warning("dimension list, missing obj reference: " + value[i])
                    continue
                if "CLASS" not in scale_obj.attrs:
                    self.log.warning("dimension list, no scale obj")
                    continue
                if scale_obj.attrs["CLASS"] != b"DIMENSION_SCALE":
                    self.log.warning("dimension list, invalid class for scale obj")
                    continue

                try:
                    h5py.h5ds.attach_scale(obj.id, scale_obj.id, i)
                except RuntimeError:
                    self.log.error("got runtime error attaching scale")

    """
    writeNdArrayToAttribute - create an attribute given numpy array
    """
    def writeNdArrayToAttribute(self, attrs, attr_name, npdata, shape, dt):
        attrs.create(attr_name, npdata, shape=shape, dtype=dt)

    """
    create a scalar string attribute using nullterm padding
    """
    def makeNullTermStringAttribute(self, obj, attr_name, strLength, value):
        self.log.info(
            "make nullterm, length: " + str(strLength) + " value:" + str(value))
        if type(value) == unicode:
            value = str(value)
        if strLength < len(value):
            self.log.warning("makeNullTermStringAttribute: value string longer than length")
            #value = value[:strLength]  # truncate to length
        
        
        if six.PY3 and type(attr_name) is str:    
            try:
                attr_name = attr_name.encode('ascii')
            except UnicodeDecodeError:
                raise TypeError("non-ascii attribute name not allowed")

        # create the attribute
        tid = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
        tid.set_size(strLength)
        tid.set_strpad(h5py.h5t.STR_NULLTERM)
        sid = h5py.h5s.create(h5py.h5s.SCALAR)
        aid = h5py.h5a.create(obj.id, attr_name, tid, sid)
        # write the value
        dtype_code = 'S' + str(strLength)
        ndarr = np.array(value, dtype=np.dtype(dtype_code))
        aid.write(ndarr)

    def makeAttribute(self, obj, attr_name, shape, attr_type, value):
        """
        makeAttribute - create an attribute (except for dimension list
        attribute)
        """
        is_committed_type = False
        if type(attr_type) in (str, unicode) and len(attr_type) == UUID_LEN:
            # assume attr_type is a uuid of a named datatype
            is_committed_type = True

        dt = self.createTypeFromItem(attr_type)

        if shape is None:
            self.log.info("shape is null - will create null space attribute")
            # create null space attribute
            # null space datasets/attributes not supported in h5py yet:
            # See: https://github.com/h5py/h5py/issues/279
            # work around this by using low-level interface.
            # first create a temp scalar dataset so we can pull out the typeid
            tmpGrp = None
            if "{tmp}" not in self.dbGrp:
                tmpGrp = self.dbGrp.create_group("{tmp}")
            else:
                tmpGrp = self.dbGrp["{tmp}"]
            tmpGrp.attrs.create(attr_name, 0, shape=(), dtype=dt)
            if six.PY3:
                b_attr_name = attr_name.encode('utf-8')
                tmpAttr = h5py.h5a.open(tmpGrp.id, name=b_attr_name)
            else:
                tmpAttr = h5py.h5a.open(tmpGrp.id, name=attr_name)
            if not tmpAttr:
                msg = "Unexpected error creating datatype for nullspace attribute"
                self.log.error(msg)
                raise IOError(errno.EIO, msg)
            tid = tmpAttr.get_type()
            sid = sid = h5py.h5s.create(h5py.h5s.NULL)
            # now create the permanent attribute
            if attr_name in obj.attrs:
                self.log.info("deleting attribute: " + attr_name)
                del obj.attrs[attr_name]
            if six.PY3:
                attr_id = h5py.h5a.create(obj.id, b_attr_name, tid, sid)
            else:
                attr_id = h5py.h5a.create(obj.id, attr_name, tid, sid)
            # delete the temp attribute
            del tmpGrp.attrs[attr_name]
            if not attr_id:
                msg = "Unexpected error creating nullspace attribute"
                self.log.error(msg)
                raise IOError(errno.EIO, msg)
        else:
            if type(value) is tuple:
                value = list(value)
            if type(shape) is list:
                shape = tuple(shape)
            if not is_committed_type:
                # apparently committed types can not be used as reference types
                # todo - verify why that is

                rank = len(shape)
                # convert python list to numpy object
                strPad = None
                strLength = 0
                if type(attr_type) == dict and attr_type['class'] == 'H5T_STRING' and "strPad" in attr_type:
                    strPad = attr_type["strPad"]
                    strLength = attr_type['length']

                if rank == 0 and type(strLength) == int and strPad == "H5T_STR_NULLTERM":
                    self.makeNullTermStringAttribute(obj, attr_name, strLength, value)
                else:
                    typeItem = getTypeItem(dt)
                    value = self.toRef(rank, typeItem, value)

                    # create numpy array
                    npdata = np.zeros(shape, dtype=dt)
                    
                    if rank == 0:
                        npdata[()] = self.toNumPyValue(attr_type, value, npdata[()])
                    else:
                        self.toNumPyArray(rank, attr_type, value, npdata)

                    self.writeNdArrayToAttribute(obj.attrs, attr_name, npdata, shape, dt)

    """
    createAttribute - create an attribute
    """
    def createAttribute(self, col_name, obj_uuid, attr_name, shape, attr_type, value):
        self.log.info("createAttribute: [" + attr_name + "]")

        self.initFile()
        if self.readonly:
            msg = "Unable to create attribute (updates are not allowed)"
            self.log.info(msg)
            raise IOError(errno.EPERM, msg)
        obj = self.getObjectByUuid(col_name, obj_uuid)
        if not obj:
            msg = "Object with uuid: " + obj_uuid + " not found"
            self.log.info(msg)
            raise IOError(errno.ENXIO, msg)

        if self.isDimensionList(attr_name, attr_type):
            self.makeDimensionList(obj, shape, value)
        elif self.isReferenceList(attr_name, attr_type):
            pass  # Skip since reference list will be created by attach scale
        else:
            self.makeAttribute(obj, attr_name, shape, attr_type, value)

        now = time.time()
        self.setCreateTime(obj_uuid, objType="attribute", name=attr_name, timestamp=now)
        self.setModifiedTime(obj_uuid, objType="attribute", name=attr_name, timestamp=now)
        self.setModifiedTime(obj_uuid, timestamp=now)  # owner entity is modified

    def deleteAttribute(self, col_name, obj_uuid, attr_name):
        self.initFile()
        if self.readonly:
            msg = "Unable to delete attribute (updates are not allowed)"
            self.log.info(msg)
            raise IOError(errno.EPERM, msg)
        obj = self.getObjectByUuid(col_name, obj_uuid)

        if attr_name not in obj.attrs:
            msg = "Attribute with name: [" + attr_name + "] of object: " + obj_uuid + " not found"
            self.log.info(msg)
            raise IOError(errno.ENXIO, msg)

        del obj.attrs[attr_name]
        now = time.time()
        self.setModifiedTime(obj_uuid, objType="attribute", name=attr_name, timestamp=now)

        return True

    """
      Return a json-serializable representation of the numpy value
    """
    def getDataValue(self, typeItem, value, dimension=0, dims=None):
        if dimension > 0:
            if type(dims) not in (list, tuple):
                msg = "unexpected type for type array dimensions"
                self.log.error(msg)
                raise IOError(errno.EIO, msg)
            out = []
            rank = len(dims)
            if dimension > rank:
                msg = "unexpected dimension for type array"
                self.log.error(msg)
                raise IOError(errno.EIO, msg)
            nElements = dims[rank - dimension]
            for i in range(nElements):
                item_value = self.getDataValue(typeItem, value[i],
                                               dimension=(dimension-1),
                                               dims=dims)
                out.append(item_value)
            return out  # done for array case

        out = None
        typeClass = typeItem['class']
        if isinstance(value, (np.ndarray, np.generic)):
            value = value.tolist()  # convert numpy object to list
        if typeClass == 'H5T_COMPOUND':

            if type(value) not in (list, tuple):
                msg = "Unexpected type for compound value"
                self.log.error(msg)
                raise IOError(errno.EIO, msg)

            fields = typeItem['fields']
            if len(fields) != len(value):
                msg = "Number of elements in compound type does not match type"
                self.log.error(msg)
                raise IOError(errno.EIO, msg)
            nFields = len(fields)
            out = []
            for i in range(nFields):
                field = fields[i]
                item_value = self.getDataValue(field['type'], value[i])
                out.append(item_value)
        elif typeClass == 'H5T_VLEN':
            if type(value) not in (list, tuple):
                msg = "Unexpected type for vlen value"
                self.log.error(msg)
                raise IOError(errno.EIO, msg)

            baseType = typeItem['base']
            out = []
            nElements = len(value)
            for i in range(nElements):
                item_value = self.getDataValue(baseType, value[i])
                out.append(item_value)
        elif typeClass == 'H5T_REFERENCE':
            out = self.refToList(value)
        elif typeClass == 'H5T_OPAQUE':
            out = "???"  # todo
        elif typeClass == 'H5T_ARRAY':
            type_dims = typeItem["dims"]
            if type(type_dims) not in (list, tuple):
                msg = "unexpected type for type array dimensions"
                self.log.error(msg)
                raise IOError(errno.EIO, msg)
            rank = len(type_dims)
            baseType = typeItem['base']
            out = self.getDataValue(baseType, value, dimension=rank,
                                    dims=type_dims)

        elif typeClass in ('H5T_INTEGER', 'H5T_FLOAT', 'H5T_ENUM'):
            out = value  # just copy value
        elif typeClass == 'H5T_STRING':
            if six.PY3:
                if "charSet" in typeItem:
                    charSet = typeItem["charSet"]
                else:
                    charSet =  "H5T_CSET_ASCII"
                if charSet == "H5T_CSET_ASCII":
                    out = value.decode("utf-8")
                else:
                    out = value
            else:
                # things are simpler in PY2
                out = value
        else:
            msg = "Unexpected type class: " + typeClass
            self.log.info(msg)
            raise IOError(errno.ENINVAL, msg)
        return out

    """
      Return a numpy value based on json representation
    """
    def getRefValue(self, typeItem, value):
        out = None
        typeClass = typeItem['class']
        if typeClass == 'H5T_COMPOUND':

            if type(value) not in (list, tuple):
                msg = "Unexpected type for compound value"
                self.log.error(msg)
                raise IOError(errno.EIO, msg)

            fields = typeItem['fields']
            if len(fields) != len(value):
                msg = "Number of elements in compound type does not match type"
                self.log.error(msg)
                raise IOError(errno.EIO, msg)
            nFields = len(fields)
            out = []
            for i in range(nFields):
                field = fields[i]
                item_value = self.getRefValue(field['type'], value[i])
                out.append(item_value)
        elif typeClass == 'H5T_VLEN':
            if type(value) not in (list, tuple):
                msg = "Unexpected type for vlen value"
                self.log.error(msg)
                raise IOError(errno.EIO, msg)

            baseType = typeItem['base']
            out = []
            nElements = len(value)
            for i in range(nElements):
                item_value = self.getRefValue(baseType, value[i])
                out.append(item_value)
        elif typeClass == 'H5T_REFERENCE':
            out = self.listToRef(value)
        elif typeClass == 'H5T_OPAQUE':
            out = "???"  # todo
        elif typeClass == 'H5T_ARRAY':
            out = value
        elif typeClass in ('H5T_INTEGER', 'H5T_FLOAT', 'H5T_ENUM'):
            out = value  # just copy value
        elif typeClass == 'H5T_STRING':
            if typeItem['charSet'] == 'H5T_CSET_UTF8':
                # out = value.encode('utf-8')
                out = value
            else:
                out = value.encode()
        else:
            msg = "Unexpected type class: " + typeClass
            self.log.info(msg)
            raise IOError(errno.ENINVAL, msg)

        if type(out) == list:
            out = tuple(out)  # convert to tuple
        return out

    """
      Return a numpy value based on json representation
    """
    def toNumPyValue(self, typeItem, src, des):

        typeClass = 'H5T_INTEGER'  # default to int type
        if type(typeItem) is dict:
            typeClass = typeItem['class']
        if typeClass == 'H5T_COMPOUND':
            fields = typeItem['fields']
            if len(fields) != len(src):
                msg = "Number of elements in compound type does not match type"
                self.log.error(msg)
                raise IOError(errno.EIO, msg)
            nFields = len(fields)

            for i in range(nFields):
                field = fields[i]
                field_name = field['name']
                des[field_name] = src[i]

        elif typeClass == 'H5T_VLEN':
            if type(src) not in (list, tuple):
                msg = "Unexpected type for vlen value"
                self.log.error(msg)
                raise IOError(errno.EIO, msg)

            baseType = typeItem['base']

            dt = self.createTypeFromItem(baseType)
            des = np.array(src, dtype=dt)

        elif typeClass == 'H5T_REFERENCE':
            des = src  # self.listToRef(src)

        elif typeClass == 'H5T_OPAQUE':
            des = "???"  # todo
        elif typeClass == 'H5T_ARRAY':
            des = src
        elif typeClass in ('H5T_INTEGER', 'H5T_FLOAT', 'H5T_ENUM'):
            des = src  # just copy value
        elif typeClass == 'H5T_STRING':
            if typeItem['charSet'] == 'H5T_CSET_UTF8':
                des = src  # src.encode('utf-8')
            else:
                if type(src) is str:
                    try:
                        src.encode('ascii')
                    except UnicodeDecodeError:
                        raise TypeError("non-ascii value not allowed with H5T_CSET_ASCII")         
                des = src
             
        else:
            msg = "Unexpected type class: " + typeClass
            self.log.info(msg)
            raise IOError(errno.ENINVAL, msg)
        return des

    """
       copy src data to numpy array
    """
    def toNumPyArray(self, rank, typeItem, src, des):

        if rank == 0:
            msg = "unexpected rank value"
            self.log.error(msg)
            raise IOError(errno.EIO, msg)  # shouldn't be called with rank 0

        for i in range(len(des)):
            des_sec = des[i]  # numpy slab

            src_sec = src[i]

            if rank > 1:
                self.toNumPyArray(rank - 1, typeItem, src_sec, des_sec)
            else:
                rv = self.toNumPyValue(typeItem, src_sec, des_sec)
                # if the numpy object is writeable, des_sec will be
                # already updated.  Otherwise, update the des by assignment
                if not hasattr(des_sec, 'flags') or not des_sec.flags['WRITEABLE']:
                    des[i] = rv

    """
       Convert json list to h5py compatible values
    """
    def toRef(self, rank, typeItem, data):
        out = None

        if type(typeItem) in (str, unicode):
            # commited type - get json representation
            committed_type_item = self.getCommittedTypeItemByUuid(typeItem)
            typeItem = committed_type_item['type']

        typeClass = typeItem['class']
        if typeClass in ('H5T_INTEGER', 'H5T_FLOAT'):
            out = data   # just use as is

        elif rank == 0:
            # scalar value
            out = self.getRefValue(typeItem, data)
        else:
            out = []
            for item in data:
                if rank > 1:
                    out_item = self.toRef(rank - 1, typeItem, item)
                    out.append(out_item)
                else:
                    out_item = self.getRefValue(typeItem, item)
                    out.append(out_item)

        return out

    """
       Convert list to json serializable values.
    """
    def toList(self, rank, typeItem, data):
        out = None
        typeClass = typeItem['class']
        if typeClass in ('H5T_INTEGER', 'H5T_FLOAT'):
            out = data.tolist()  # just use as is

        elif rank == 0:
            # scalar value
            out = self.getDataValue(typeItem, data)
        else:
            out = []
            for item in data:
                if rank > 1:
                    out_item = self.toList(rank - 1, typeItem, item)
                    out.append(out_item)
                else:
                    out_item = self.getDataValue(typeItem, item)
                    out.append(out_item)

        return out

    """
       Create ascii representation of vlen data object
    """
    def vlenToList(self, data):
        # todo - verify that data is a numpy.ndarray
        out = None
        if len(data.shape) == 0:
            out = []
        else:
            try:
                if data.dtype.kind != 'O':
                    out = data.tolist()
                else:
                    out = []
                    for item in data:
                        out.append(self.vlenToList(item))  # recursive call
            except AttributeError:
                # looks like this is not a numpy ndarray, just return the value
                out = data
        return out

    """
       Create ascii representation of ref data object
    """
    def refToList(self, data):
        # todo - verify that data is a numpy.ndarray
        out = None
        if type(data) is h5py.h5r.Reference:
            if bool(data):
                grpref = self.f[data]
                addr = h5py.h5o.get_info(grpref.id).addr
                uuid = self.getUUIDByAddress(addr)
                if self.getGroupObjByUuid(uuid):
                    out = "groups/" + uuid
                elif self.getDatasetObjByUuid(uuid):
                    out = "datasets/" + uuid
                elif self.getCommittedTypeObjByUuid(uuid):
                    out = "datatypes/" + uuid
                else:
                    self.log.warning(
                        "uuid in region ref not found: [" + uuid + "]")
                    return None
            else:
                out = "null"
        elif type(data) is h5py.h5r.RegionReference:
            out = self.getRegionReference(data)
        else:
            out = []
            for item in data:
                out.append(self.refToList(item))  # recursive call
        return out

    """
       Convert ascii representation of data references to data ref
    """
    def listToRef(self, data):
        out = None
        if not data:
            # null reference
            out = self.getNullReference()
        elif type(data) in (bytes, str, unicode):
            obj_ref = None
            # object reference should be in the form: <collection_name>/<uuid>
            for prefix in ("datasets", "groups", "datatypes"):
                if data.startswith(prefix):
                    uuid_ref = data[len(prefix):]
                    if len(uuid_ref) == (UUID_LEN + 1) and uuid_ref.startswith('/'):
                        obj = self.getObjectByUuid(prefix, uuid_ref[1:])
                        if obj:
                            obj_ref = obj.ref
                        else:
                            msg = "Invalid object reference value: [" + uuid_ref + "] not found"
                            self.log.info(msg)
                            raise IOError(errno.ENXIO, msg)
                    break
            if not obj_ref:
                msg = "Invalid object reference value: [" + data + "]"
                self.log.info(msg)
                raise IOError(errno.EINVAL, msg)
            else:
                out = obj_ref

        elif type(data) in (list, tuple):
            out = []
            for item in data:
                out.append(self.listToRef(item))  # recursive call
        elif type(data) == dict:
            # assume region ref
            out = self.createRegionReference(data)
        else:
            msg = "Invalid object reference value type: [" + str(type(data)) + "]"
            self.log.info(msg)
            raise IOError(errno.EINVAL, msg)
        return out
        
    """
       Convert list that may contain bytes type elements to list of string elements  
    """
    def bytesArrayToList(self, data):
        if type(data) in (bytes, str, unicode):
            is_list = False
        elif isinstance(data, (np.ndarray, np.generic)):
            if len(data.shape) == 0:
                is_list = False
                data = data.tolist()  # tolist will return a scalar in this case
                if type(data) in (list, tuple):
                    is_list = True
                else:
                    is_list = False
            else:
                is_list = True        
        elif type(data) in (list, tuple):
            is_list = True
        else:
            is_list = False
                
        if is_list:
            out = []
            for item in data:
                out.append(self.bytesArrayToList(item)) # recursive call  
        elif type(data) is bytes:
            if six.PY3:
                out = data.decode("utf-8")
            else:
                out = data
        else:
            out = data
                   
        return out
    
    """
      Get item description of region reference value
    """
    def getRegionReference(self, regionRef):
        selectionEnums = {h5py.h5s.SEL_NONE:       'H5S_SEL_NONE',
                          h5py.h5s.SEL_ALL:        'H5S_SEL_ALL',
                          h5py.h5s.SEL_POINTS:     'H5S_SEL_POINTS',
                          h5py.h5s.SEL_HYPERSLABS: 'H5S_SEL_HYPERSLABS'}

        item = {}
        objid = h5py.h5r.dereference(regionRef, self.f.file.file.id)
        if objid:
            item['id'] = self.getUUIDByAddress(h5py.h5o.get_info(objid).addr)
        else:
                self.log.info("region reference unable to find item with objid: " + objid)
                return item

        sel = h5py.h5r.get_region(regionRef, objid)
        select_type = sel.get_select_type()
        if select_type not in selectionEnums:
            msg = "Unexpected selection type: " + regionRef.typecode
            self.log.error(msg)
            raise IOError(errno.EIO, msg)
        item['select_type'] = selectionEnums[select_type]
        pointlist = None
        if select_type == h5py.h5s.SEL_POINTS:
            # retrieve a numpy array of selection points
            points = sel.get_select_elem_pointlist()
            pointlist = points.tolist()
        elif select_type == h5py.h5s.SEL_HYPERSLABS:
            points = sel.get_select_hyper_blocklist()
            if points is not None:
                pointlist = points[...].tolist()
                # bump up the second coordinate by one to match api spec
                for point in pointlist:
                    coord2 = point[1]
                    for i in range(len(coord2)):
                        coord2[i] = coord2[i] + 1

        item['selection'] = pointlist

        return item

    """
      Create region reference from item description of region reference value
    """
    def createRegionReference(self, item):
        selectionEnums = {'H5S_SEL_NONE': h5py.h5s.SEL_NONE,
                          'H5S_SEL_ALL': h5py.h5s.SEL_ALL,
                          'H5S_SEL_POINTS': h5py.h5s.SEL_POINTS,
                          'H5S_SEL_HYPERSLABS': h5py.h5s.SEL_HYPERSLABS}
        region_ref = None

        if 'select_type' not in item:
            msg = "select_type not provided for region selection"
            self.log.info(msg)
            raise IOError(errno.EINVAL, msg)
        select_type = item['select_type']
        if select_type not in selectionEnums.keys():
            msg = "selection type: [" + select_type + "] is not valid"
            self.log.info(msg)
            raise IOError(errno.EINVAL, msg)
        dset = None
        if select_type == 'H5S_SEL_NONE':
                if 'id' not in item:
                        #        select none on null dataset, return null ref
                        out = self.getNullReference()
                        return out
        else:  # select_type != 'H5S_SEL_NONE'
            if 'id' not in item:
                msg = "id not provided for region selection"
                self.log.info(msg)
                raise IOError(errno.EINVAL, msg)

        # Otherwise need to provide uuid of dataset
        uuid_ref = item['id']
        if len(uuid_ref) != UUID_LEN:
            msg = "uuid value: [" + uuid_ref + "] for region reference is not valid"
            self.log.info(msg)
            raise IOError(errno.EINVAL, msg)

        obj = self.getObjectByUuid("datasets", uuid_ref)
        if obj:
            dset = obj
        else:
            msg = "Invalid region refence value: [" + uuid_ref + "] not found"
            self.log.info(msg)
            raise IOError(errno.EINVAL, msg)

        if select_type in ('H5S_SEL_POINTS', 'H5S_SEL_HYPERSLABS'):
            if 'selection' not in item:
                msg = "selection key not provided for region selection"
                self.log.info(msg)
                raise IOError(errno.EINVAL, msg)

        rank = len(dset.shape)
        space_id = h5py.h5d.DatasetID.get_space(dset.id)
        h5py.h5s.SpaceID.select_none(space_id)

        if select_type == 'H4S_SEL_NONE':
                pass  # did select_none above
        elif select_type == 'H5S_SEL_ALL':
            h5py.h5s.SpaceID.select_all(space_id)
        elif select_type == 'H5S_SEL_POINTS':
            selection = item['selection']
            for point in selection:
                if len(point) != rank:
                        msg = "point selection number of elements must mach rank of referenced dataset"
                        self.log.info(msg)
                        raise IOError(errno.EINVAL, msg)
            h5py.h5s.SpaceID.select_elements(space_id, selection)
        elif select_type == 'H5S_SEL_HYPERSLABS':
            selection = item['selection']

            for slab in selection:
                    # each item should be a two element array defining the hyperslab boundary
                    if len(slab) != 2:
                        msg = "selection value not valid (not a 2 element array)"
                        self.log.info(msg)
                        raise IOError(errno.EINVAL, msg)
                    start = slab[0]
                    if type(start) == list:
                        start = tuple(start)
                    if type(start) is not tuple or len(start) != rank:
                        msg = "selection value not valid, start element should have number "
                        msg += "elements equal to rank of referenced dataset"
                        self.log.info(msg)
                        raise IOError(errno.EINVAL, msg)
                    stop = slab[1]
                    if type(stop) == list:
                        stop = tuple(stop)
                    if type(stop) is not tuple or len(stop) != rank:
                        msg = "selection value not valid, count element should have number "
                        msg += "elements equal to rank of referenced dataset"
                        self.log.info(msg)
                        raise IOError(errno.EINVAL, msg)
                    count = []
                    for i in range(rank):
                        if start[i] < 0:
                                msg = "start value for hyperslab selection must be non-negative"
                                self.log.info(msg)
                                raise IOError(errno.EINVAL, msg)
                        if stop[i] <= start[i]:
                                msg = "stop value must be greater than start value for hyperslab selection"
                                self.log.info(msg)
                                raise IOError(errno.EINVAL, msg)
                        count.append(stop[i] - start[i])
                    count = tuple(count)

                    h5py.h5s.SpaceID.select_hyperslab(space_id, start, count, op=h5py.h5s.SELECT_OR)

        # now that we've selected the desired region in the space, return a region reference
        
        if six.PY3:
            dset_name = dset.name.encode('utf-8')
        else:
            dset_name = dset.name
        region_ref = h5py.h5r.create(self.f.id, dset_name, h5py.h5r.DATASET_REGION, space_id)

        return region_ref

    """
      Convert a list to a tuple, recursively.
      Example. [[1,2],[3,4]] -> ((1,2),(3,4))
    """
    def toTuple(self, rank, data):
        if type(data) in (list, tuple):
            if rank > 0:
                return list(self.toTuple(rank-1, x) for x in data)
            else:
                return tuple(self.toTuple(rank-1, x) for x in data)
        else:
            return data

    """
    Get values from dataset identified by obj_uuid.
    If a slices list or tuple is provided, it should have the same
    number of elements as the rank of the dataset.
    """
    def getDatasetValuesByUuid(self, obj_uuid, slices=Ellipsis, format="json"):
        dset = self.getDatasetObjByUuid(obj_uuid)
        if format not in ("json", "binary"):
            msg = "only json and binary formats are supported"
            self.log.info(msg)
            raise IOError(errno.EINVAL, msg)
            
        if dset is None:
            msg = "Dataset: " + obj_uuid + " not found"
            self.log.info(msg)
            raise IOError(errno.ENXIO, msg)
            
        values = None
        dt = dset.dtype
        typeItem = getTypeItem(dt)
        itemSize = getItemSize(typeItem)
        if itemSize == "H5T_VARIABLE" and format == "binary":
            msg = "Only JSON is supported for for this data type"
            self.log.info(msg)
            raise IOError(errno.EINVAL, msg)
            
        if dset.shape is None:
            # null space dataset (with h5py 2.6.0)
            return None   
               
        rank = len(dset.shape)
         
        if rank == 0:
            # check for null dataspace
            try:
                val = dset[...]
            except IOError:
                # assume null dataspace, return none
                return None
            if val is None:
                self.log.warning("no value returned from scalar dataset")

        if type(slices) != list and type(slices) != tuple and slices is not Ellipsis:
            msg = "Unexpected error: getDatasetValuesByUuid: bad type for dim parameter"
            self.log.error(msg)
            raise IOError(errno.EIO, msg)

        if (type(slices) == list or type(slices) == tuple) and len(slices) != rank:
            msg = "Unexpected error: getDatasetValuesByUuid: number of dims in selection not same as rank"
            self.log.error(msg)
            raise IOError(errno.EIO, msg)
       
        if dt.kind == 'O':
            if format != "json":
                msg = "Only JSON is supported for for this data type"
                self.log.info(msg)
                raise IOError(errno.EINVAL, msg)
            # numpy object type - could be a vlen string or generic vlen
            h5t_check = h5py.h5t.check_dtype(vlen=dt)
            if h5t_check == str or h5t_check == unicode:
                values = dset[slices].tolist()  # just dump to list
            elif six.PY3 and h5t_check == bytes:
                values = self.bytesArrayToList(dset[slices])
            elif h5t_check is not None:
                # other vlen data
                values = self.vlenToList(dset[slices])
            else:
                # check for reference type
                h5t_check = h5py.h5t.check_dtype(ref=dt)
                if h5t_check is not None:
                    # reference type
                    values = self.refToList(dset[slices])
                else:
                    msg = "Unexpected error, object type unknown"
                    self.log.error(msg)
                    raise IOError(errno.EIO, msg)
        elif dt.kind == 'V' and len(dt) <= 1 and len(dt.shape) == 0:
            # opaque type - skip for now
            self.log.warning("unable to get opaque type values")
            values =  "????"
        elif dt.kind == 'S' and format == "json" and six.PY3:
            values = self.bytesArrayToList(dset[slices])
        elif len(dt) > 1:
            # compound type
            if format == "json":
                values = self.bytesArrayToList(dset[slices])
            else:
                values = dset[slices].tobytes()
        else:
            values = dset[slices]
            
            # just use tolist to dump
            if format == "json":             
                values = values.tolist()
            else:
                #values = base64.b64encode(dset[slices].tobytes())
                values = values.tobytes()
            
        return values
        
    """
      doDatasetQueryByUuid: return rows based on query string
        Return rows from a dataset that matches query string.
        
        Note: Only supported for compound_type/one-dimensional datasets
    """
    def doDatasetQueryByUuid(self, obj_uuid, query, start=0, stop=-1, step=1, limit=None):
        self.log.info("doQueryByUuid - uuid: " + obj_uuid + " query:" + query)
        self.log.info("start: " + str(start) + " stop: " + str(stop) + " step: " + str(step) + " limit: " + str(limit))
        dset = self.getDatasetObjByUuid(obj_uuid)   
        if dset is None:
            msg = "Dataset: " + obj_uuid + " not found"
            self.log.info(msg)
            raise IOError(errno.ENXIO, msg)
            
        values = []
        dt = dset.dtype
        typeItem = getTypeItem(dt)
        itemSize = getItemSize(typeItem)
        if typeItem['class'] != "H5T_COMPOUND":
            msg = "Only compound type datasets can be used as query target"
            self.log.info(msg)
            raise IOError(errno.EINVAL, msg)
            
        if dset.shape is None:
            # null space dataset (with h5py 2.6.0)
            return None   
               
        rank = len(dset.shape)
        if rank != 1:
            msg = "One one-dimensional datasets can be used as query target"
            self.log.info(msg)
            raise IOError(errno.EINVAL, msg)
        

        values = []
        indexes = []
        count = 0
      
        num_elements = dset.shape[0]
        if stop == -1:
            stop = num_elements
        elif stop > num_elements:
            stop = num_elements
        block_size = self._getBlockSize(dset)
        self.log.info("block_size: " + str(block_size))   
        
        field_names = list(dset.dtype.fields.keys())
        eval_str = self._getEvalStr(query, field_names) 
        
        while start < stop:
            if limit and (count == limit):
                break  # no more rows for this batch
            end = start  + block_size
            if end > stop:
                end = stop
            rows = dset[start:end]  # read from dataset
            where_result = np.where(eval(eval_str))
            index = where_result[0].tolist()
            if len(index) > 0:
                for i in index:
                    row = rows[i]
                    item = self.bytesArrayToList(row)
                    values.append(item)
                    indexes.append(start + i)
                    count += 1
                    if limit and (count == limit):
                        break  # no more rows for this batch
                              
            start = end  # go to next block
            
         
        # values = self.getDataValue(item_type, values, dimension=1, dims=(len(values),))
        
        self.log.info("got " + str(count) + " query matches")    
        return (indexes, values)
    
    """
     _getBlockSize: Get number of rows to read from disk
     
        heurestic to get reasonable sized chunk of data to fetch.
        make multiple of chunk_size if possible
    """    
    def _getBlockSize(self, dset):
        target_block_size = 256 * 1000
        if dset.chunks:
            chunk_size = dset.chunks[0]
            if chunk_size < target_block_size:
                block_size = (target_block_size // chunk_size) * chunk_size
            else:
                block_size = target_block_size
        else:
            block_size = target_block_size
        return block_size
    
    """
     _getEvalStr: Get eval string for given query
     
        Gets Eval string to use with numpy where method.
    """    
    def _getEvalStr(self, query, field_names):
        i = 0
        eval_str = ""
        var_name = None
        end_quote_char = None
        var_count = 0
        paren_count = 0
        black_list = ( "import", ) # field names that are not allowed
        self.log.info("getEvalStr(" + query + ")")
        for item in black_list:
            if item in field_names:
                msg = "invalid field name"
                self.log.info("EINVAL: " + msg)
                raise IOError(errno.EINVAL, msg)
        while i < len(query):
            ch = query[i]
            if (i+1) < len(query):
                ch_next = query[i+1]
            else:
                ch_next = None
            if var_name and not ch.isalnum():
                # end of variable
                if var_name not in field_names:
                    # invalid
                    msg = "unknown field name"
                    self.log.info("EINVAL: " + msg)
                    raise IOError(errno.EINVAL, msg)
                eval_str += "rows['" + var_name + "']"
                var_name = None
                var_count += 1
            
            if end_quote_char:
                if ch == end_quote_char:
                    # end of literal
                    end_quote_char = None
                eval_str += ch
            elif ch in ("'", '"'):
                end_quote_char = ch
                eval_str += ch
            elif ch.isalpha():
                if ch == 'b' and ch_next in ("'", '"'):
                    eval_str += 'b' # start of a byte string literal
                elif var_name is None:
                    var_name = ch  # start of a variable
                else:
                    var_name += ch
            elif ch == '(' and end_quote_char is None:
                paren_count += 1
                eval_str += ch
            elif ch == ')' and end_quote_char is None:
                paren_count -= 1
                if paren_count < 0:
                    msg = "Mismatched paren"
                    self.log.info("EINVAL: " + msg)
                    raise IOError(errno.EINVAL, msg)
                eval_str += ch
            else:
                # just add to eval_str
                eval_str += ch
            i = i+1
        if end_quote_char:
            msg = "no matching quote character"
            self.log.info("EINVAL: " + msg)
            raise IOError(errno.EINVAL, msg)
        if var_count == 0:
            msg = "No field value"
            self.log.info("EINVAL: " + msg)
            raise IOError(errno.EINVAL, msg)
        if paren_count != 0:
            msg = "Mismatched paren"
            self.log.info("EINVAL: " + msg)
            raise IOError(errno.EINVAL, msg)
            
        return eval_str

    """
    Get values from dataset identified by obj_uuid using the given
    point selection.
    """
    def getDatasetPointSelectionByUuid(self, obj_uuid, points):
        dset = self.getDatasetObjByUuid(obj_uuid)
        if dset is None:
            msg = "Dataset: " + obj_uuid + " not found"
            self.log.info(msg)
            raise IOError(errno.ENXIO, msg)
            
        rank = len(dset.shape)
        values = np.zeros(len(points), dtype=dset.dtype)
        try:
            i = 0
            for point in points:
                if rank == 1:
                    values[i] = dset[[point]]
                else:
                    values[i] = dset[tuple(point)]
                i += 1
        except ValueError:
            # out of range error
            msg = "getDatasetPointSelection, out of range error"
            self.log.info(msg)
            raise IOError(errno.EINVAL, msg)
        return values.tolist()

    """
    setDatasetValuesByUuid - update the given dataset values with supplied data
      and optionally a hyperslab selection (slices)
    """
    def setDatasetValuesByUuid(self, obj_uuid, data, slices=None, format="json"):
        dset = self.getDatasetObjByUuid(obj_uuid)
        
        if format not in ("json", "binary"):
            msg = "only json and binary formats are supported"
            self.log.info(msg)
            raise IOError(errno.EINVAL, msg)
            
        if format == "binary" and type(data) is not bytes:
            msg ="data must be of type bytes for binary writing"
            self.log.info(msg)
            raise IOError(errno.EINVAL, msg)
            
        if dset is None:
            msg = "Dataset: " + obj_uuid + " not found"
            self.log.info(msg)
            raise IOError(errno.ENXIO, msg) 
            
        dt = dset.dtype
        typeItem = getTypeItem(dt)
        itemSize = getItemSize(typeItem)
        rank = len(dset.shape)
        arraySize = 1
        for extent in dset.shape:
            arraySize *= arraySize
            
        if itemSize == "H5T_VARIABLE" and format == "binary":
            msg = "Only JSON is supported for for this data type"
            self.log.info(msg)
            raise IOError(errno.EINVAL, msg)     

        if slices is None:
            slices = []
            # create selection that covers entire dataset
            for dim in range(rank):
                s = slice(0, dset.shape[dim], 1)
                slices.append(s)
            slices = tuple(slices)
            
             
        if type(slices) != tuple:
            msg = "setDatasetValuesByUuid: bad type for dim parameter"
            self.log.error(msg)
            raise IOError(erno.EIO, msg)
            

        if len(slices) != rank:
            msg = "number of dims in selection not same as rank"
            self.log.info(msg)
            raise IOError(errno.EINVAL, msg)
           
        npoints = 1
        np_shape = []
        for i in range(rank):
            s = slices[i]
            
            if s.start < 0 or s.step <= 0 or s.stop < s.start:
                msg = "invalid slice specification"
                self.log.info(msg)
                raise IOError(errno.EINVAL, msg)
            if s.stop > dset.shape[i]:
                msg = "invalid slice specification"
                self.log.info(msg)
                raise IOError(errno.EINVAL, msg)
            np_shape.append(s.stop - s.start)
                        
            count = (s.stop - s.start) // s.step
            if count <= 0:
                msg = "invalid slice specification"
                self.log.info(msg)
                raise IOError(errno.EINVAL, msg)  
                    
            npoints *= count        
               
        np_shape = tuple(np_shape)  # for comparison with ndarray shape
                
        self.log.info("selection shape:" + str(np_shape))


        # need some special conversion for compound types --
        # each element must be a tuple, but the JSON decoder
        # gives us a list instead.
        if format != "binary" and len(dset.dtype) > 1 and type(data) in (list, tuple):
            data = self.toTuple(rank, data)
            #for i in range(len(data)):
            #    converted_data.append(self.toTuple(data[i]))
            #data = converted_data
        else:
            h5t_check = h5py.check_dtype(ref=dset.dtype)
            if h5t_check in (h5py.Reference, h5py.RegionReference):
                # convert data to data refs
                if format == "binary":
                    msg = "Only JSON is supported for for this data type"
                    self.log.info(msg)
                    raise IOError(errno.EINVAL, msg)  
                data = self.listToRef(data)
                    
        if format == "binary":
            if npoints*itemSize != len(data):
                msg = "Expected: " + str(npoints*itemSize) + " bytes, but got: " + str(len(data))
                self.log.info(msg)
                raise IOError(errno.EINVAL, msg)
            if dset.dtype.shape == ():
                arr = np.fromstring(data, dtype=dset.dtype)
                arr = arr.reshape(np_shape)  # conform to selection shape
            else:
                # tricy array type!
                arr = np.empty(np_shape, dtype=dset.dtype)
                base_arr = np.fromstring(data, dtype=dset.dtype.base)
                base_shape = list(np_shape)
                base_shape.extend(dset.dtype.shape)  # add on the type dimensions
                base_arr = base_arr.reshape(base_shape)
                arr[...] = base_arr
        else:
            # data is json
            if npoints == 1 and len(dset.dtype) > 1:
                # convert to tuple for compound singleton writes
                data = [tuple(data),]

            arr = np.array(data, dtype=dset.dtype)
            # raise an exception of the array shape doesn't match the selection shape
            # allow if the array is a scalar and the selection shape is one element,
            # numpy is ok with this
            np_index = 0
            for dim in range(len(arr.shape)):
                data_extent = arr.shape[dim]
                selection_extent = 1
                if np_index < len(np_shape):
                    selection_extent = np_shape[np_index]
                if selection_extent == data_extent:
                    np_index += 1
                    continue  # good
                if data_extent == 1:
                    continue  # skip singleton selection
                if selection_extent == 1:
                    np_index += 1
                    continue  # skip singleton selection
                 
                # selection/data mismatch!
                msg = "data shape doesn't match selection shape"
                msg += "--data shape: " + str(arr.shape)
                msg += "--selection shape: " + str(np_shape)
                
                self.log.info(msg)
                raise IOError(errno.EINVAL, msg)
                    
        # write temp numpy array to dataset        
        if rank == 1:
            s = slices[0]
            try:
                dset[s] = arr
            except TypeError as te:
                self.log.info("h5py setitem exception: " + str(te))
                raise IOError(errno.EINVAL, str(te))
        else:
            try:
                dset[slices] = arr
            except TypeError as te:
                self.log.info("h5py setitem exception: " + str(te))
                raise IOError(errno.EINVAL, str(te))

        # update modified time
        self.setModifiedTime(obj_uuid)
        return True

    """
    setDatasetValuesByPointSelection - Update the dataset values using the given
      data and point selection
    """
    def setDatasetValuesByPointSelection(self, obj_uuid, data, points, format="json"):
        dset = self.getDatasetObjByUuid(obj_uuid)
        
        if format not in ("json", "binary"):
            msg = "only json and binary formats are supported"
            self.log.info(msg)
            raise IOError(errno.EINVAL, msg)
            
        if format == "binary" and type(data) is not bytes:
            msg ="data must be of type bytes for binary writing"
            self.log.info(msg)
            raise IOError(errno.EINVAL, msg)
            
        if dset is None:
            msg = "Dataset: " + obj_uuid + " not found"
            self.log.info(msg)
            raise IOError(errno.ENXIO, msg) 
            
        dt = dset.dtype
        typeItem = getTypeItem(dt)
        itemSize = getItemSize(typeItem)
        if itemSize == "H5T_VARIABLE" and format == "binary":
            msg = "Only JSON is supported for for this data type"
            self.log.info(msg)
            raise IOError(errno.EINVAL, msg)     

        rank = len(dset.shape)

        # need some special conversion for compound types --
        # each element must be a tuple, but the JSON decoder
        # gives us a list instead.
        if format == "json" and len(dset.dtype) > 1 and type(data) in (list, tuple):
            converted_data = self.toTuple(rank, data)
            #for i in range(len(data)):
            #    converted_data.append(self.toTuple(data[i]))
            #data = converted_data
        
        if format == "json":

            try:
                i = 0
                for point in points:
                    if rank == 1:
                        dset[[point]] = data[i]
                    else:
                        dset[tuple(point)] = data[i]
                    i += 1
            except ValueError:
                # out of range error
                msg = "setDatasetValuesByPointSelection, out of range error"
                self.log.info(msg)
                raise IOError(errno.EINVAL, msg)
            
        else:
            #binary
            arr = np.fromstring(data, dtype=dset.dtype)
            dset[points] = arr     # coordinate write

        # update modified time
        self.setModifiedTime(obj_uuid)
        return True

    """
    createDataset - creates new dataset given shape and datatype
    Returns item
    """
    def createDataset(self, datatype, datashape, max_shape=None,
                      creation_props=None, obj_uuid=None):
        self.initFile()
        if self.readonly:
            msg = "Unable to create dataset (Updates are not allowed)"
            self.log.info(msg)
            raise IOError(errno.EPERM, msg)
        datasets = self.dbGrp["{datasets}"]
        if not obj_uuid:
            obj_uuid = str(uuid.uuid1())
        dt = None
        item = {}

        # h5py.createdataset fields
        kwargs = {}  # key word arguments for h5py dataset creation

        fillvalue = None

        if creation_props is None:
            creation_props = {}  # create empty list for convience

        if creation_props:
            if "fillValue" in creation_props:
                fillvalue = creation_props["fillValue"]
            if "trackTimes" in creation_props:
                kwargs['track_times'] = creation_props["trackTimes"]
            if "layout" in creation_props:
                layout = creation_props["layout"]
                if "dims" in layout:
                    kwargs['chunks'] = tuple(layout["dims"])
            if "filters" in creation_props:
                filter_props = creation_props["filters"]
                for filter_prop in filter_props:
                    if "id" not in filter_prop:
                        msg = "filter id not provided"
                        self.log.info(msg)
                        raise IOError(errno.EINVAL, msg)
                    filter_id = filter_prop["id"]
                    if filter_id not in _HDF_FILTERS:
                        self.log.info("unknown filter id: " + str(filter_id) + " ignoring")
                        continue

                    hdf_filter = _HDF_FILTERS[filter_id]

                    self.log.info("got filter: " + str(filter_id))
                    if "alias" not in hdf_filter:
                        self.log.info("unsupported filter id: " + str(filter_id) + " ignoring")
                        continue

                    filter_alias = hdf_filter["alias"]
                    if not h5py.h5z.filter_avail(filter_id):
                        self.log.info("compression filter not available, filter: " + filter_alias + " will be ignored")
                        continue
                    if filter_alias in _H5PY_COMPRESSION_FILTERS:
                        if kwargs.get('compression'):
                            self.log.info("compression filter already set, filter: " + filter_alias + " will be ignored")
                            continue

                        kwargs['compression'] = filter_alias
                        self.log.info("setting compression filter to: " + kwargs['compression'])
                        if filter_alias == "gzip":
                            # check for an optional compression value
                            if "level" in filter_prop:
                                kwargs['compression_opts'] = filter_prop["level"]
                        elif filter_alias == "szip":
                            bitsPerPixel = None
                            coding = 'nn'

                            if "bitsPerPixel" in filter_prop:
                                bitsPerPixel = filter_prop["bitsPerPixel"]
                            if "coding" in filter_prop:
                                if filter_prop["coding"] == "H5_SZIP_EC_OPTION_MASK":
                                    coding = 'ec'
                                elif filter_prop["coding"] == "H5_SZIP_NN_OPTION_MASK":
                                    coding = 'nn'
                                else:
                                    msg = "invalid szip option: 'coding'"
                                    self.log.info(msg)
                                    raise IOError(errno.EINVAL, msg)
                            # note: pixelsPerBlock, and pixelsPerScanline not supported by h5py,
                            # so these options will be ignored
                            if "pixelsPerBlock" in filter_props:
                                self.log.info("ignoring szip option: 'pixelsPerBlock'")
                            if "pixelsPerScanline" in filter_props:
                                self.log.info("ignoring szip option: 'pixelsPerScanline'")
                            if bitsPerPixel:
                                kwargs['compression_opts'] = (coding, bitsPerPixel)
                    else:
                        if filter_alias == "shuffle":
                            kwargs['shuffle'] = True
                        elif filter_alias == "fletcher32":
                            kwargs['fletcher32'] = True
                        elif filter_alias == "scaleoffset":
                            if "scaleOffset" not in filter_prop:
                                msg = "No scale_offset provided for scale offset filter"
                                self.log(msg)
                                raise IOError(errno.EINVAL, msg)
                            kwargs['scaleoffset'] = filter_prop["scaleOffset"]
                        else:
                            self.log.info("Unexpected filter name: " + filter_alias + " , ignoring")

        dt_ref = self.createTypeFromItem(datatype)
        if dt_ref is None:
            msg = 'Unexpected error, no type returned'
            self.log.error(msg)
            raise IOError(errno.EIO, msg)

        dt = dt_ref
        if hasattr(dt_ref, 'dtype'):
            # dt_ref is actualy a handle to a committed type
            # get the dtype prop, but use dt_ref for the actual dataset creation
            dt = dt_ref.dtype

        if fillvalue and len(dt) > 1 and type(fillvalue) in (list, tuple):
            # for compound types, need to convert from list to dataset compatible element

            if len(dt) != len(fillvalue):
                msg = 'fillvalue has incorrect number of elements'
                self.log.info(msg)
                raise IOError(errno.EINVAL, msg)
            ndscalar = np.zeros((), dtype=dt)
            for i in range(len(fillvalue)):
                field = dt.names[i]
                ndscalar[field] = self.toTuple(0, fillvalue[i])
            fillvalue = ndscalar

        if fillvalue:
            kwargs['fillvalue'] = fillvalue

        dataset_id = None
        if datashape is None:
            # create null space dataset
            # null space datasets not supported in h5py yet:
            # See: https://github.com/h5py/h5py/issues/279
            # work around this by using low-level interface.
            # first create a temp scalar dataset so we can pull out the typeid
            tmpGrp = None
            if "{tmp}" not in self.dbGrp:
                tmpGrp = self.dbGrp.create_group("{tmp}")
            else:
                tmpGrp = self.dbGrp["{tmp}"]
            tmpDataset = tmpGrp.create_dataset(obj_uuid, shape=(1,),
                                               dtype=dt_ref)
            tid = tmpDataset.id.get_type()
            sid = sid = h5py.h5s.create(h5py.h5s.NULL)
            # now create the permanent dataset
            gid = datasets.id
            if six.PY3:
                b_obj_uuid = obj_uuid.encode('utf-8')
                dataset_id = h5py.h5d.create(gid, b_obj_uuid, tid, sid)
            else:
                dataset_id = h5py.h5d.create(gid, obj_uuid, tid, sid)
            # delete the temp dataset
            del tmpGrp[obj_uuid]
        else:

            # create the dataset
           
            try:
                newDataset = datasets.create_dataset(
                    obj_uuid, shape=datashape, maxshape=max_shape,
                    dtype=dt_ref, **kwargs)
            except ValueError as ve:
                msg = "Unable to create dataset"
                try:
                    msg += ": " + ve.message
                except AttributeError:
                    pass  # no message
                self.log.info(msg)
                raise IOError(errno.EINVAL, msg)  # assume this is due to invalid params

            if newDataset:
                dataset_id = newDataset.id

        if dataset_id is None:
            msg = 'Unexpected failure to create dataset'
            self.log.error(msg)
            raise IOError(errno.EIO, msg)
        # store reverse map as an attribute
        addr = h5py.h5o.get_info(dataset_id).addr
        addrGrp = self.dbGrp["{addr}"]
        addrGrp.attrs[str(addr)] = obj_uuid

        # save creation props if any
        if creation_props:
            self.setDatasetCreationProps(obj_uuid, creation_props)

        # set timestamp
        now = time.time()
        self.setCreateTime(obj_uuid, timestamp=now)
        self.setModifiedTime(obj_uuid, timestamp=now)

        item['id'] = obj_uuid
        if self.update_timestamps:
            item['ctime'] = self.getCreateTime(obj_uuid)
            item['mtime'] = self.getModifiedTime(obj_uuid)
        item['attributeCount'] = 0
        return item

    """
    Resize existing Dataset
    """
    def resizeDataset(self, obj_uuid, shape):
        self.log.info("resizeDataset(") #  + obj_uuid + "): ") # + str(shape))
        self.initFile()
        if self.readonly:
            msg = "Unable to resize dataset (Updates are not allowed)"
            self.log.info(msg)
            raise IOError(errno.EACESS, msg)
        dset = self.getDatasetObjByUuid(obj_uuid)  # will throw exception if not found
        if len(shape) != len(dset.shape):
            msg = "Unable to resize dataset, shape has wrong number of dimensions"
            self.log.info(msg)
            raise IOError(errno.EINVAL, msg)
        for i in range(len(shape)):
            if shape[i] < dset.shape[i]:
                msg = "Unable to resize dataset, cannot make extent smaller"
                self.log.info(msg)
                raise IOError(errno.EINVAL, msg)
            if dset.maxshape[i] != None and shape[i] > dset.maxshape[i]:
                msg = "Unable to resize dataset, max extent exceeded"
                self.log.info(msg)
                raise IOError(errno.EINVAL, msg)

        dset.resize(shape)  # resize

        # update modified time
        self.setModifiedTime(obj_uuid)

    """
    Check if link points to given target (as a HardLink)
    """
    def isObjectHardLinked(self, parentGroup, targetGroup, linkName):
        try:
            linkObj = parentGroup.get(linkName, None, False, True)
            linkClass = linkObj.__class__.__name__
        except TypeError:
            # UDLink? Ignore for now
            return False
        if linkClass == 'SoftLink':
            return False
        elif linkClass == 'ExternalLink':
            return False
        elif linkClass == 'HardLink':
            if parentGroup[linkName] == targetGroup:
                return True
        else:
            self.log.warning("unexpected linkclass: " + linkClass)
            return False

    """
    Delete Dataset, Group or Datatype by UUID
    """
    def deleteObjectByUuid(self, objtype, obj_uuid):
        if objtype not in ('group', 'dataset', 'datatype'):
            msg = "unexpected objtype: " + objtype
            self.log.error(msg)
            raise IOError(errno.EIO, msg)
        self.initFile()
        self.log.info("delete uuid: " + obj_uuid)
        if self.readonly:
            msg = "Unable to delete object (Updates are not allowed)"
            self.log.info(msg)
            raise IOError(errno.EPERM, msg)

        if obj_uuid == self.dbGrp.attrs["rootUUID"] and objtype == 'group':
            # can't delete root group
            msg = "Unable to delete group (root group may not be deleted)"
            self.log.info(msg)
            raise IOError(errno.EPERM, msg)

        dbCol = None
        tgt = None
        if objtype == 'dataset':
            tgt = self.getDatasetObjByUuid(obj_uuid)
            dbCol = self.dbGrp["{datasets}"]
        elif objtype == 'group':
            tgt = self.getGroupObjByUuid(obj_uuid)
            dbCol = self.dbGrp["{groups}"]
        else:  # datatype
            tgt = self.getCommittedTypeObjByUuid(obj_uuid)
            dbCol = self.dbGrp["{datatypes}"]

        if tgt is None:
            msg = "Unable to delete " + objtype + ", uuid: " + obj_uuid + " not found"
            self.log.info(msg)
            raise IOError(errno.ENXIO, msg)

        # unlink from root (if present)
        self.unlinkObject(self.f['/'], tgt)

        groups = self.dbGrp["{groups}"]
        # iterate through each group in the file and unlink tgt if it is linked
        # by the group.
        # We'll store a list of links to be removed as we go, and then actually
        # remove the links after the iteration is done (otherwise we can run into issues
        # where the key has become invalid)
        linkList = []  # this is our list
        for uuidName in groups.attrs:
            grpRef = groups.attrs[uuidName]
            # de-reference handle
            grp = self.f[grpRef]
            for linkName in grp:
                if self.isObjectHardLinked(grp, tgt, linkName):
                    linkList.append({'group': grp, 'link': linkName})
        for item in linkList:
            self.unlinkObjectItem(item['group'], tgt, item['link'])

        addr = h5py.h5o.get_info(tgt.id).addr
        addrGrp = self.dbGrp["{addr}"]
        del addrGrp.attrs[str(addr)]  # remove reverse map
        dbRemoved = False

        # finally, remove the dataset from db
        if obj_uuid in dbCol:
            # should be here (now it is anonymous)
            del dbCol[obj_uuid]
            dbRemoved = True

        if not dbRemoved:
            self.log.warning("did not find: " + obj_uuid + " in anonymous collection")

            if obj_uuid in dbCol.attrs:
                self.log.info("removing: " + obj_uuid + " from non-anonymous collection")
                del dbCol.attrs[obj_uuid]
                dbRemoved = True

        if not dbRemoved:
            msg = "Unexpected Error, did not find reference to: " + obj_uuid
            self.log.error(msg)
            raise IOError(errno.EIO, msg)

        # note when the object was deleted
        self.setModifiedTime(obj_uuid)

        return True

    def getGroupItemByUuid(self, obj_uuid):
        self.initFile()
        grp = self.getGroupObjByUuid(obj_uuid)
        if grp is None:
            if self.getModifiedTime(obj_uuid, useRoot=False):
                msg = "Group with uuid: " + obj_uuid + " has been previously deleted"
                self.log.info(msg)
                raise IOError(errno.ENOENT, msg)
            else:
                msg = "Group with uuid: " + obj_uuid + " was not found"
                self.log.info(msg)
                raise IOError(errno.ENXIO, msg)

        linkCount = len(grp)
        if "__db__" in grp:
            linkCount -= 1  # don't include the db group

        item = { 'id': obj_uuid }
        alias = []
        if grp.name and not grp.name.startswith("/__db__"):
            alias.append(grp.name)   # just use the default h5py path for now
        item['alias'] = alias
        item['attributeCount'] = len(grp.attrs)
        item['linkCount'] = linkCount
        if self.update_timestamps:
            item['ctime'] = self.getCreateTime(obj_uuid)
            item['mtime'] = self.getModifiedTime(obj_uuid)

        return item

    """
    getLinkItemByObj - return info about a link
        parent: reference to group
        linkName: name of link
        return: item dictionary with link attributes, or None if not found
    """
    def getLinkItemByObj(self, parent, link_name):
        if link_name not in parent:
            return None

        if link_name == "__db__":
            return None  # don't provide link to db group
        #  "http://somefile/#h5path(somepath)")
        item = { 'title': link_name }
        # get the link object, one of HardLink, SoftLink, or ExternalLink
        try:
            linkObj = parent.get(link_name, None, False, True)
            linkClass = linkObj.__class__.__name__
        except TypeError:
            # UDLink? set class as 'user'
            linkClass = 'UDLink'  # user defined links
            item['class'] = 'H5L_TYPE_USER_DEFINED'
        if linkClass == 'SoftLink':
            item['class'] = 'H5L_TYPE_SOFT'
            item['h5path'] = linkObj.path
            item['href'] = '#h5path(' + linkObj.path + ')'
        elif linkClass == 'ExternalLink':
            item['class'] = 'H5L_TYPE_EXTERNAL'
            item['h5path'] = linkObj.path
            item['file'] = linkObj.filename
            item['href'] = '#h5path(' + linkObj.path + ')'
        elif linkClass == 'HardLink':
            # Hardlink doesn't have any properties itself, just get the linked
            # object
            obj = parent[link_name]
            addr = h5py.h5o.get_info(obj.id).addr
            item['class'] = 'H5L_TYPE_HARD'
            item['id'] = self.getUUIDByAddress(addr)
            class_name = obj.__class__.__name__
            if class_name == 'Dataset':
                item['href'] = 'datasets/' + item['id']
                item['collection'] = 'datasets'
            elif class_name == 'Group':
                item['href'] = 'groups/' + item['id']
                item['collection'] = 'groups'
            elif class_name == 'Datatype':
                item['href'] = 'datatypes/' + item['id']
                item['collection'] = 'datatypes'
            else:
                self.log.warning("unexpected object type: " + item['type'])

        return item

    def getLinkItemByUuid(self, grpUuid, link_name):
        self.log.info(
            "db.getLinkItemByUuid(" + grpUuid + ", [" + link_name + "])")
        if not link_name:
            msg = "link_name not specified"
            self.log.info(msg)
            raise IOError(errno.EINVAL, msg)

        self.initFile()
        parent = self.getGroupObjByUuid(grpUuid)
        if parent is None:
            msg = "Parent group: " + grpUuid + " of link not found"
            self.log.info(msg)
            raise IOError(errno.ENXIO, msg)

        item = self.getLinkItemByObj(parent, link_name)
        # add timestamps
        if item:
            if self.update_timestamps:
                item['ctime'] = self.getCreateTime(grpUuid, objType="link", name=link_name)
                item['mtime'] = self.getModifiedTime(grpUuid, objType="link", name=link_name)
        else:
            self.log.info("link not found")
            mtime = self.getModifiedTime(grpUuid, objType="link", name=link_name, useRoot=False)
            if mtime:
                msg = "Link [" + link_name + "] of: " + grpUuid + " has been previously deleted"
                self.log.info(msg)
                raise IOError(errno.ENOENT, msg)
            else:
                msg = "Link [" + link_name + "] of: " + grpUuid + " not found"
                self.log.info(msg)
                raise IOError(errno.ENXIO, msg)

        return item

    def getLinkItems(self, grpUuid, marker=None, limit=0):
        self.log.info("db.getLinkItems(" + grpUuid + ")")
        if marker:
            self.log.info("...marker: " + marker)
        if limit:
            self.log.info("...limit: " + str(limit))

        self.initFile()
        parent = self.getGroupObjByUuid(grpUuid)
        if parent is None:
            msg = "Parent group: " + grpUuid + " not found, no links returned"
            self.log.info(msg)
            raise IOError(errno.ENXIO, msg)
        items = []
        gotMarker = True
        if marker is not None:
            gotMarker = False
        count = 0
        for link_name in parent:
            if link_name == "__db__":
                continue
            if not gotMarker:
                if link_name == marker:
                    gotMarker = True
                    continue  # start filling in result on next pass
                else:
                    continue  # keep going!
            item = self.getLinkItemByObj(parent, link_name)
            items.append(item)

            count += 1
            if limit > 0 and count == limit:
                break  # return what we got
        return items

    def unlinkItem(self, grpUuid, link_name):
        if self.readonly:
            msg = "Unable to unlink item (Updates are not allowed)"
            self.log.info(msg)
            raise IOError(errno.EPERM, msg)
        grp = self.getGroupObjByUuid(grpUuid)
        if grp is None:
            msg = "Parent group: " + grpUuid + " not found, cannot remove link"
            self.log.info(msg)
            raise IOError(errno.ENXIO, msg)

        if link_name not in grp:
            msg = "Link: [" + link_name + "] of group: " + grpUuid + " not found, cannot remove link"
            self.log.info(msg)
            raise IOError(errno.ENXIO, msg)

        if link_name == "__db__":
            # don't allow db group to be unlinked!
            msg = "Unlinking of __db__ group not allowed"
            raise IOError(errno.EPERM, msg)

        obj = None
        try:
            linkObj = grp.get(link_name, None, False, True)
            linkClass = linkObj.__class__.__name__
            if linkClass == 'HardLink':
                # we can safely reference the object
                obj = grp[link_name]
        except TypeError:
            # UDLink? Return false to indicate that we can not delete this
            msg = "Unable to unlink user defined link"
            self.log.info(msg)
            raise IOError(errno.EPERM, msg)

        linkDeleted = False
        if obj is not None:
            linkDeleted = self.unlinkObjectItem(grp, obj, link_name)
        else:
            # SoftLink or External Link - we can just remove the key
            del grp[link_name]
            linkDeleted = True

        if linkDeleted:
            # update timestamp
            self.setModifiedTime(grpUuid, objType="link", name=link_name)

        return linkDeleted

    def getCollection(self, col_type, marker=None, limit=None):
        self.log.info("db.getCollection(" + col_type + ")")
        #col_type should be either "datasets", "groups", or "datatypes"
        if col_type not in ("datasets", "groups", "datatypes"):
            msg = "Unexpected col_type: [" + col_type + "]"
            self.log.error(msg)
            raise IOError(errno.EIO, msg)
        self.initFile()
        col = None  # Group, Dataset, or Datatype
        if col_type == "datasets":
            col = self.dbGrp["{datasets}"]
        elif col_type == "groups":
            col = self.dbGrp["{groups}"]
        else:  # col_type == "datatypes"
            col = self.dbGrp["{datatypes}"]

        uuids = []
        count = 0
        # gather the non-anonymous ids first
        for obj_uuid in col.attrs:
            if marker:
                if obj_uuid == marker:
                    marker = None  # clear and pick up next item
                continue
            uuids.append(obj_uuid)
            count += 1
            if limit is not None and limit > 0 and count == limit:
                break

        if limit == 0 or (limit is not None and count < limit):
            # grab any anonymous obj ids next
            for obj_uuid in col:
                if marker:
                    if obj_uuid == marker:
                        marker = None  # clear and pick up next item
                    continue
                uuids.append(obj_uuid)
                count += 1
                if limit is not None and limit > 0 and count == limit:
                    break

        return uuids

    """
      Get the DB Collection names
    """
    def getDBCollections(self):
        return ("{groups}", "{datasets}", "{datatypes}")

    """
        Return the db collection the uuid belongs to
    """
    def getDBCollection(self, obj_uuid):
        dbCollections = self.getDBCollections()
        for dbCollectionName in dbCollections:
            col = self.dbGrp[dbCollectionName]
            if obj_uuid in col or obj_uuid in col.attrs:
                return col
        return None

    def unlinkObjectItem(self, parentGrp, tgtObj, link_name):
        if self.readonly:
            msg = "Unexpected attempt to unlink object"
            self.log.error(msg)
            raise IOError(errno.EIO, msg)
        if link_name not in parentGrp:
            msg = "Unexpected: did not find link_name: [" + link_name + "]"
            self.log.error(msg)
            raise IOError(errno.EIO, msg)
        try:
            linkObj = parentGrp.get(link_name, None, False, True)
        except TypeError:
            # user defined link?
            msg = "Unable to remove link (user-defined link?)"
            self.log.error(msg)
            raise IOError(errno.EIO, msg)
        linkClass = linkObj.__class__.__name__
        # only deal with HardLinks
        linkDeleted = False
        if linkClass == 'HardLink':
            obj = parentGrp[link_name]
            if tgtObj is None or obj == tgtObj:

                numlinks = self.getNumLinksToObject(obj)
                if numlinks == 1:
                    # last link to this object - convert to anonymous object by
                    # creating link under {datasets} or {groups} or {datatypes}
                    # also remove the attribute UUID key
                    addr = h5py.h5o.get_info(obj.id).addr
                    obj_uuid = self.getUUIDByAddress(addr)
                    self.log.info("converting: " + obj_uuid
                                  + " to anonymous obj")
                    dbCol = self.getDBCollection(obj_uuid)
                    del dbCol.attrs[obj_uuid]  # remove the object ref
                    dbCol[obj_uuid] = obj      # add a hardlink
                self.log.info("deleting link: [" + link_name + "] from: "
                              + parentGrp.name)
                del parentGrp[link_name]
                linkDeleted = True
        else:
            self.log.info("unlinkObjectItem: link is not a hardlink, ignoring")
        return linkDeleted

    def unlinkObject(self, parentGrp, tgtObj):
        for name in parentGrp:
            self.unlinkObjectItem(parentGrp, tgtObj, name)
        return True

    def linkObject(self, parentUUID, childUUID, link_name):
        self.initFile()
        if self.readonly:
            msg = "Unable to create link (Updates are not allowed)"
            self.log.info(msg)
            raise IOError(errno.EPERM, msg)

        parentObj = self.getGroupObjByUuid(parentUUID)
        if parentObj is None:
            msg = "Unable to create link, parent UUID: " + parentUUID + " not found"
            self.log.info(msg)
            raise IOError(errno.ENXIO, msg)

        childObj = self.getDatasetObjByUuid(childUUID)
        if childObj is None:
            # maybe it's a group...
            childObj = self.getGroupObjByUuid(childUUID)
        if childObj is None:
            # or maybe it's a committed datatype...
            childObj = self.getCommittedTypeObjByUuid(childUUID)
        if childObj is None:
            msg = "Unable to link item, child UUID: " + childUUID + " not found"
            self.log.info(msg)
            raise IOError(errno.ENXIO, msg)
        if link_name in parentObj:
            # link already exists
            self.log.info("linkname already exists, deleting")
            self.unlinkObjectItem(parentObj, None, link_name)
        parentObj[link_name] = childObj

        # convert this from an anonymous object to ref if needed
        dbCol = self.getDBCollection(childUUID)
        if childUUID in dbCol:
            # convert to a ref
            del dbCol[childUUID]  # remove hardlink
            dbCol.attrs[childUUID] = childObj.ref # create a ref

        # set link timestamps
        now = time.time()
        self.setCreateTime(parentUUID, objType="link", name=link_name, timestamp=now)
        self.setModifiedTime(parentUUID, objType="link", name=link_name, timestamp=now)
        return True

    def createSoftLink(self, parentUUID, linkPath, link_name):
        self.initFile()
        if self.readonly:
            msg = "Unable to create link (Updates are not allowed)"
            self.log.info(msg)
            raise IOError(errno.EPERM, msg)
        parentObj = self.getGroupObjByUuid(parentUUID)
        if parentObj is None:
            msg = "Unable to create link, parent UUID: " + parentUUID + " not found"
            self.log.info(msg)
            raise IOError(errno.ENXIO, msg)
        if link_name in parentObj:
            # link already exists
            self.log.info("linkname already exists, deleting")
            del parentObj[link_name]  # delete old link
        parentObj[link_name] = h5py.SoftLink(linkPath)

        now = time.time()
        self.setCreateTime(parentUUID, objType="link", name=link_name, timestamp=now)
        self.setModifiedTime(parentUUID, objType="link", name=link_name, timestamp=now)

        return True

    def createExternalLink(self, parentUUID, extPath, linkPath, link_name):
        self.initFile()
        if self.readonly:
            msg = "Unable to create link (Updates are not allowed)"
            self.log.info(msg)
            raise IOError(errno.EPERM, msg)
        parentObj = self.getGroupObjByUuid(parentUUID)
        if parentObj is None:
            msg = "Unable to create link, parent UUID: " + parentUUID + " not found"
            self.log.info(msg)
            raise IOError(errno.ENXIO, msg)
        if link_name in parentObj:
            # link already exists
            self.log.info("linkname already exists, deleting")
            del parentObj[link_name]  # delete old link
        parentObj[link_name] = h5py.ExternalLink(extPath, linkPath)

        now = time.time()
        self.setCreateTime(parentUUID, objType="link", name=link_name, timestamp=now)
        self.setModifiedTime(parentUUID, objType="link", name=link_name, timestamp=now)

        return True

    def createGroup(self, obj_uuid=None):
        self.initFile()
        if self.readonly:
            msg = "Unable to create group (Updates are not allowed)"
            self.log.info(msg)
            raise IOError(errno.EPERM, msg)
        groups = self.dbGrp["{groups}"]
        if not obj_uuid:
            obj_uuid = str(uuid.uuid1())
        newGroup = groups.create_group(obj_uuid)
        # store reverse map as an attribute
        addr = h5py.h5o.get_info(newGroup.id).addr
        addrGrp = self.dbGrp["{addr}"]
        addrGrp.attrs[str(addr)] = obj_uuid

        #set timestamps
        now = time.time()
        self.setCreateTime(obj_uuid, timestamp=now)
        self.setModifiedTime(obj_uuid, timestamp=now)

        return obj_uuid

    def getNumberOfGroups(self):
        self.initFile()
        count = 0
        groups = self.dbGrp["{groups}"]
        count += len(groups)        # anonymous groups
        count += len(groups.attrs)  # linked groups
        count += 1                  # add of for root group

        return count

    def getNumberOfDatasets(self):
        self.initFile()
        count = 0
        datasets = self.dbGrp["{datasets}"]
        count += len(datasets)        # anonymous datasets
        count += len(datasets.attrs)  # linked datasets
        return count

    def getNumberOfDatatypes(self):
        self.initFile()
        count = 0
        datatypes = self.dbGrp["{datatypes}"]
        count += len(datatypes)        # anonymous datatypes
        count += len(datatypes.attrs)  # linked datatypes
        return count
