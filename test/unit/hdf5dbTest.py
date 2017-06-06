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
import unittest
import sys
import os
import time
import base64
import errno
import os.path as op
import stat
import logging
import shutil

from h5json import Hdf5db

UUID_LEN = 36  # length for uuid strings

def getFile(name, tgt, ro=False):
    src = '../../data/hdf5/' + name
    logging.info("copying file to this directory: " + src)

    filepath = "./out/" + tgt

    if op.isfile(filepath):
        # make sure it's writable, before we copy over it
        os.chmod(filepath, stat.S_IWRITE|stat.S_IREAD)
    shutil.copyfile(src, filepath)
    if ro:
        logging.info('make read-only')
        os.chmod(filepath, stat.S_IREAD)
    return filepath


def removeFile(name):
    try:
        os.stat(name)
    except OSError:
        return;   # file does not exist
    os.remove(name)

class Hdf5dbTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Hdf5dbTest, self).__init__(*args, **kwargs)
        # main

        self.log = logging.getLogger()
        if len(self.log.handlers) > 0:
            lhStdout = self.log.handlers[0]  # stdout is the only handler initially
        else:
            lhStdout = None

        self.log.setLevel(logging.INFO)
        # create logger

        handler = logging.FileHandler('./hdf5dbtest.log')
        # add handler to logger
        self.log.addHandler(handler)

        if lhStdout is not None:
            self.log.removeHandler(lhStdout)
        #self.log.propagate = False  # prevent log out going to stdout
        self.log.info('init!')

        #create directory for test output files
        if not os.path.exists('./out'):
            os.makedirs('./out')

    def testInvalidPath(self):
        filepath = "/tmp/thisisnotafile.h5"
        try:
            with Hdf5db(filepath, app_logger=self.log) as db:
                self.assertTrue(False)  # shouldn't get here
        except IOError as e:
            self.assertEqual(e.errno, errno.ENXIO)
            self.assertEqual(e.strerror, "file not found")

    def testInvalidFile(self):
        filepath = getFile('notahdf5file.h5', 'notahdf5file.h5')
        try:
            with Hdf5db(filepath, app_logger=self.log) as db:
                self.assertTrue(False)  # shouldn't get here
        except IOError as e:
            self.assertEqual(e.errno, errno.EINVAL)
            self.assertEqual(e.strerror, "not an HDF5 file")


    def testGetUUIDByPath(self):
        # get test file
        g1Uuid = None
        filepath = getFile('tall.h5', 'getuuidbypath.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            g1Uuid = db.getUUIDByPath('/g1')
            self.assertEqual(len(g1Uuid), UUID_LEN)
            obj = db.getObjByPath('/g1')
            self.assertEqual(obj.name, '/g1')
            for name in obj:
                g = obj[name]
            g1links = db.getLinkItems(g1Uuid)
            self.assertEqual(len(g1links), 2)
            for item in g1links:
                self.assertEqual(len(item['id']), UUID_LEN)

        # end of with will close file
        # open again and verify we can get obj by name
        with Hdf5db(filepath, app_logger=self.log) as db:
            obj = db.getGroupObjByUuid(g1Uuid)
            g1 = db.getObjByPath('/g1')
            self.assertEqual(obj, g1)

    def testGetCounts(self):
        filepath = getFile('tall.h5', 'testgetcounts_tall.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            cnt = db.getNumberOfGroups()
            self.assertEqual(cnt, 6)
            cnt = db.getNumberOfDatasets()
            self.assertEqual(cnt, 4)
            cnt = db.getNumberOfDatatypes()
            self.assertEqual(cnt, 0)

        filepath = getFile('empty.h5', 'testgetcounts_empty.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            cnt = db.getNumberOfGroups()
            self.assertEqual(cnt, 1)
            cnt = db.getNumberOfDatasets()
            self.assertEqual(cnt, 0)
            cnt = db.getNumberOfDatatypes()
            self.assertEqual(cnt, 0)


    def testGroupOperations(self):
        # get test file
        filepath = getFile('tall.h5', 'tall_del_g11.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            rootuuid = db.getUUIDByPath('/')
            root = db.getGroupObjByUuid(rootuuid)
            self.assertEqual('/', root.name)
            rootLinks = db.getLinkItems(rootuuid)
            self.assertEqual(len(rootLinks), 2)
            g1uuid = db.getUUIDByPath("/g1")
            self.assertEqual(len(g1uuid), UUID_LEN)
            g1Links = db.getLinkItems(g1uuid)
            self.assertEqual(len(g1Links), 2)
            g11uuid = db.getUUIDByPath("/g1/g1.1")
            db.deleteObjectByUuid("group", g11uuid)

    def testCreateGroup(self):
        # get test file
        filepath = getFile('tall.h5', 'tall_newgrp.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            rootUuid = db.getUUIDByPath('/')
            numRootChildren = len(db.getLinkItems(rootUuid))
            self.assertEqual(numRootChildren, 2)
            newGrpUuid = db.createGroup()
            newGrp = db.getGroupObjByUuid(newGrpUuid)
            self.assertNotEqual(newGrp, None)
            db.linkObject(rootUuid, newGrpUuid, 'g3')
            numRootChildren = len(db.getLinkItems(rootUuid))
            self.assertEqual(numRootChildren, 3)
            # verify linkObject can be called idempotent-ly
            db.linkObject(rootUuid, newGrpUuid, 'g3')

    def testGetLinkItemsBatch(self):
        # get test file
        filepath = getFile('group100.h5', 'getlinkitemsbatch.h5')
        marker = None
        count = 0
        with Hdf5db(filepath, app_logger=self.log) as db:
            rootUuid = db.getUUIDByPath('/')
            while True:
                # get items 13 at a time
                batch = db.getLinkItems(rootUuid, marker=marker, limit=13)
                if len(batch) == 0:
                    break   # done!
                count += len(batch)
                lastItem = batch[len(batch) - 1]
                marker = lastItem['title']
        self.assertEqual(count, 100)

    def testGetItemHardLink(self):
        filepath = getFile('tall.h5', 'getitemhardlink.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            grpUuid = db.getUUIDByPath('/g1/g1.1')
            item = db.getLinkItemByUuid(grpUuid, "dset1.1.1")
            self.assertTrue('id' in item)
            self.assertEqual(item['title'], 'dset1.1.1')
            self.assertEqual(item['class'], 'H5L_TYPE_HARD')
            self.assertEqual(item['collection'], 'datasets')
            self.assertTrue('target' not in item)
            self.assertTrue('mtime' in item)
            self.assertTrue('ctime' in item)

    def testGetItemSoftLink(self):
        filepath = getFile('tall.h5', 'getitemsoftlink.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            grpUuid = db.getUUIDByPath('/g1/g1.2/g1.2.1')
            item = db.getLinkItemByUuid(grpUuid, "slink")
            self.assertTrue('id' not in item)
            self.assertEqual(item['title'], 'slink')
            self.assertEqual(item['class'], 'H5L_TYPE_SOFT')
            self.assertEqual(item['h5path'], 'somevalue')
            self.assertTrue('mtime' in item)
            self.assertTrue('ctime' in item)

    def testGetItemExternalLink(self):
        filepath = getFile('tall_with_udlink.h5', 'getitemexternallink.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            grpUuid = db.getUUIDByPath('/g1/g1.2')
            item = db.getLinkItemByUuid(grpUuid, "extlink")
            self.assertTrue('uuid' not in item)
            self.assertEqual(item['title'], 'extlink')
            self.assertEqual(item['class'], 'H5L_TYPE_EXTERNAL')
            self.assertEqual(item['h5path'], 'somepath')
            self.assertEqual(item['file'], 'somefile')
            self.assertTrue('mtime' in item)
            self.assertTrue('ctime' in item)

    def testGetItemUDLink(self):
        filepath = getFile('tall_with_udlink.h5', 'getitemudlink.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            grpUuid = db.getUUIDByPath('/g2')
            item = db.getLinkItemByUuid(grpUuid, "udlink")
            self.assertTrue('uuid' not in item)
            self.assertEqual(item['title'], 'udlink')
            self.assertEqual(item['class'], 'H5L_TYPE_USER_DEFINED')
            self.assertTrue('h5path' not in item)
            self.assertTrue('file' not in item)
            self.assertTrue('mtime' in item)
            self.assertTrue('ctime' in item)

    def testGetNumLinks(self):
        items = None
        filepath = getFile('tall.h5', 'getnumlinks.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            g1= db.getObjByPath('/g1')
            numLinks = db.getNumLinksToObject(g1)
            self.assertEqual(numLinks, 1)

    def testGetLinks(self):
        g12_links = ('extlink', 'g1.2.1')
        hardLink = None
        externalLink = None
        filepath = getFile('tall_with_udlink.h5', 'getlinks.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            grpUuid = db.getUUIDByPath('/g1/g1.2')
            items = db.getLinkItems(grpUuid)
            self.assertEqual(len(items), 2)
            for item in items:
                self.assertTrue(item['title'] in g12_links)
                if item['class'] == 'H5L_TYPE_HARD':
                    hardLink = item
                elif item['class'] == 'H5L_TYPE_EXTERNAL':
                    externalLink = item
        self.assertEqual(hardLink['collection'], 'groups')
        self.assertTrue('id' in hardLink)
        self.assertTrue('id' not in externalLink)
        self.assertEqual(externalLink['h5path'], 'somepath')
        self.assertEqual(externalLink['file'], 'somefile')


    def testDeleteLink(self):
        # get test file
        filepath = getFile('tall.h5', 'deletelink.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            rootUuid = db.getUUIDByPath('/')
            numRootChildren = len(db.getLinkItems(rootUuid))
            self.assertEqual(numRootChildren, 2)
            db.unlinkItem(rootUuid, "g2")
            numRootChildren = len(db.getLinkItems(rootUuid))
            self.assertEqual(numRootChildren, 1)

    def testDeleteUDLink(self):
        # get test file
        filepath = getFile('tall_with_udlink.h5', 'deleteudlink.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            g2Uuid = db.getUUIDByPath('/g2')
            numG2Children = len(db.getLinkItems(g2Uuid))
            self.assertEqual(numG2Children, 3)
            got_exception = False
            try:
                db.unlinkItem(g2Uuid, "udlink")
            except IOError as ioe:
                got_exception = True
                self.assertEqual(ioe.errno, errno.EPERM)
            self.assertTrue(got_exception)
            numG2Children = len(db.getLinkItems(g2Uuid))
            self.assertEqual(numG2Children, 3)


    def testReadOnlyGetUUID(self):
        # get test file
        filepath = getFile('tall.h5', 'readonlygetuuid.h5', ro=True)
        # remove db file!
        removeFile('./out/.' + 'readonlygetuuid.h5')
        g1Uuid = None
        with Hdf5db(filepath, app_logger=self.log) as db:
            g1Uuid = db.getUUIDByPath('/g1')
            self.assertEqual(len(g1Uuid), UUID_LEN)
            obj = db.getObjByPath('/g1')
            self.assertEqual(obj.name, '/g1')

        # end of with will close file
        # open again and verify we can get obj by name
        with Hdf5db(filepath, app_logger=self.log) as db:
            obj = db.getGroupObjByUuid(g1Uuid)
            g1 = db.getObjByPath('/g1')
            self.assertEqual(obj, g1)
            g1links = db.getLinkItems(g1Uuid)
            self.assertEqual(len(g1links), 2)
            for item in g1links:
                self.assertEqual(len(item['id']), UUID_LEN)

    def testReadDataset(self):
         filepath = getFile('tall.h5', 'readdataset.h5')
         d111_values = None
         d112_values = None
         with Hdf5db(filepath, app_logger=self.log) as db:
            d111Uuid = db.getUUIDByPath('/g1/g1.1/dset1.1.1')
            self.assertEqual(len(d111Uuid), UUID_LEN)
            d111_values = db.getDatasetValuesByUuid(d111Uuid)
            self.assertTrue(type(d111_values) is list)
            self.assertEqual(len(d111_values), 10)
            for i in range(10):
                arr = d111_values[i]
                self.assertEqual(len(arr), 10)
                for j in range(10):
                    self.assertEqual(arr[j], i*j)

            d112Uuid = db.getUUIDByPath('/g1/g1.1/dset1.1.2')
            self.assertEqual(len(d112Uuid), UUID_LEN)
            d112_values = db.getDatasetValuesByUuid(d112Uuid)
            self.assertTrue(type(d112_values) is list)
            self.assertEqual(len(d112_values), 20)
            for i in range(20):
                self.assertEqual(d112_values[i], i)
                
    def testReadDatasetBinary(self):
         filepath = getFile('tall.h5', 'readdatasetbinary.h5')
         d111_values = None
         d112_values = None
         with Hdf5db(filepath, app_logger=self.log) as db:
            d111Uuid = db.getUUIDByPath('/g1/g1.1/dset1.1.1')
            self.assertEqual(len(d111Uuid), UUID_LEN)
            d111_data = db.getDatasetValuesByUuid(d111Uuid, format="binary")
            self.assertTrue(type(d111_data) is bytes)           
            self.assertEqual(len(d111_data), 400)  # 10x10x(4 byte type)
                     
            d112Uuid = db.getUUIDByPath('/g1/g1.1/dset1.1.2')
            self.assertEqual(len(d112Uuid), UUID_LEN)
            d112_data = db.getDatasetValuesByUuid(d112Uuid, format="binary")
            self.assertEqual(len(d112_data), 80) # 20x(4 byte type)
             
               
    def testReadCompoundDataset(self):
         filepath = getFile('compound.h5', 'readcompound.h5')
         with Hdf5db(filepath, app_logger=self.log) as db:
            dset_uuid = db.getUUIDByPath('/dset')
            self.assertEqual(len(dset_uuid), UUID_LEN)
            dset_values = db.getDatasetValuesByUuid(dset_uuid)

            self.assertEqual(len(dset_values), 72)
            elem = dset_values[0]
            self.assertEqual(elem[0], 24)
            self.assertEqual(elem[1], "13:53")
            self.assertEqual(elem[2], 63)
            self.assertEqual(elem[3], 29.88)
            self.assertEqual(elem[4], "SE 10")
            
    def testReadDatasetCreationProp(self):
         filepath = getFile('compound.h5', 'readdatasetcreationprop.h5')
         with Hdf5db(filepath, app_logger=self.log) as db:
            dset_uuid = db.getUUIDByPath('/dset')
            self.assertEqual(len(dset_uuid), UUID_LEN)
            dset_item = db.getDatasetItemByUuid(dset_uuid)
            self.assertTrue('creationProperties' in dset_item)
            creationProp = dset_item['creationProperties']
            self.assertTrue('fillValue' in creationProp)
            fillValue = creationProp['fillValue']
            
            self.assertEqual(fillValue[0], 999)
            self.assertEqual(fillValue[1], "99:90")
            self.assertEqual(fillValue[2], 999)
            self.assertEqual(fillValue[3], 999.0)
            self.assertEqual(fillValue[4], "N")

                       

    def testCreateScalarDataset(self):
        creation_props = {
                "allocTime": "H5D_ALLOC_TIME_LATE",
                "fillTime": "H5D_FILL_TIME_IFSET",
                "fillValue": "",
                "layout": {
                    "class": "H5D_CONTIGUOUS"
                }
            }
        datatype = {
                "charSet": "H5T_CSET_ASCII",
                "class": "H5T_STRING",
                "length": 1,
                "strPad": "H5T_STR_NULLPAD"
            }
        filepath = getFile('empty.h5', 'createscalardataset.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            dims = ()  # if no space in body, default to scalar
            max_shape=None

            db.createDataset(datatype, dims, max_shape=max_shape, creation_props=creation_props)

    def testCreate1dDataset(self):
        datatype = "H5T_STD_I64LE"
        dims = (10,)
        filepath = getFile('empty.h5', 'create1ddataset.h5')
        dset_uuid = None
        with Hdf5db(filepath, app_logger=self.log) as db:
            rsp = db.createDataset(datatype, dims)

            dset_uuid = rsp['id']
            item = db.getDatasetItemByUuid(dset_uuid)
            self.assertEqual(item['attributeCount'], 0)
            type_item = item['type']
            self.assertEqual(type_item['class'], 'H5T_INTEGER')
            self.assertEqual(type_item['base'], 'H5T_STD_I64LE')
            shape_item = item['shape']
            self.assertEqual(shape_item['class'], 'H5S_SIMPLE')
            self.assertEqual(shape_item['dims'], (10,))

    def testCreate2dExtendableDataset(self):
        datatype = "H5T_STD_I64LE"
        dims = (10, 10)
        max_shape = (None, 10)
        filepath = getFile('empty.h5', 'create2dextendabledataset.h5')
        dset_uuid = None
        with Hdf5db(filepath, app_logger=self.log) as db:
            rsp = db.createDataset(datatype, dims, max_shape=max_shape)
            dset_uuid = rsp['id']
            item = db.getDatasetItemByUuid(dset_uuid)
            self.assertEqual(item['attributeCount'], 0)
            type_item = item['type']
            self.assertEqual(type_item['class'], 'H5T_INTEGER')
            self.assertEqual(type_item['base'], 'H5T_STD_I64LE')
            shape_item = item['shape']
            self.assertEqual(shape_item['class'], 'H5S_SIMPLE')
            self.assertEqual(shape_item['dims'], (10,10))
            self.assertTrue('maxdims' in shape_item)
            self.assertEqual(shape_item['maxdims'], [0, 10])

    def testCreateCommittedTypeDataset(self):
        filepath = getFile('empty.h5', 'createcommittedtypedataset.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            root_uuid = db.getUUIDByPath('/')

            datatype = { 'charSet':   'H5T_CSET_ASCII',
                     'class':  'H5T_STRING',
                     'strPad': 'H5T_STR_NULLTERM',
                     'length': 15}
            item = db.createCommittedType(datatype)
            type_uuid = item['id']

            dims = ()  # if no space in body, default to scalar
            rsp = db.createDataset(type_uuid, dims, max_shape=None, creation_props=None)
            dset_uuid = rsp['id']
            item = db.getDatasetItemByUuid(dset_uuid)
            type_item = item['type']
            self.assertTrue('uuid' in type_item)
            self.assertEqual(type_item['uuid'], type_uuid)

    def testCreateCommittedCompoundTypeDataset(self):
        filepath = getFile('empty.h5', 'createcommittedcompoundtypedataset.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            root_uuid = db.getUUIDByPath('/')

            datatype = {'class': 'H5T_COMPOUND',
                        'fields': [] }

            type_fields = []
            type_fields.append({'name': 'field_1', 'type': 'H5T_STD_I64BE' })
            type_fields.append({'name': 'field_2', 'type': 'H5T_IEEE_F64BE' })

            datatype['fields'] = type_fields

            creation_props = {
                "fillValue": [
                    0,
                    0.0 ]
            }

            item = db.createCommittedType(datatype)   
            type_uuid = item['id']

            dims = ()  # if no space in body, default to scalar
            rsp = db.createDataset(type_uuid, dims, max_shape=None, creation_props=creation_props)
            dset_uuid = rsp['id']
            item = db.getDatasetItemByUuid(dset_uuid)
            type_item = item['type']
            self.assertTrue('uuid' in type_item)
            self.assertEqual(type_item['uuid'], type_uuid)



    def testReadZeroDimDataset(self):
        filepath = getFile('zerodim.h5', 'readzerodeimdataset.h5')
              
        with Hdf5db(filepath, app_logger=self.log) as db:
            dsetUuid = db.getUUIDByPath('/dset')
            self.assertEqual(len(dsetUuid), UUID_LEN)
            dset_value = db.getDatasetValuesByUuid(dsetUuid)
            self.assertEqual(dset_value, 42)
            
            
    def testReadNullSpaceDataset(self):
        filepath = getFile('null_space_dset.h5', 'readnullspacedataset.h5')
         
        with Hdf5db(filepath, app_logger=self.log) as db:
            dsetUuid = db.getUUIDByPath('/DS1')
            self.assertEqual(len(dsetUuid), UUID_LEN)
            obj = db.getDatasetObjByUuid(dsetUuid)
            shape_item = db.getShapeItemByDsetObj(obj)
            self.assertTrue('class' in shape_item)
            self.assertEqual(shape_item['class'], 'H5S_NULL')

    def testReadScalarSpaceArrayDataset(self):
        filepath = getFile('scalar_array_dset.h5', 'readscalarspacearraydataset.h5')
         
        with Hdf5db(filepath, app_logger=self.log) as db:
            dsetUuid = db.getUUIDByPath('/DS1')
            self.assertEqual(len(dsetUuid), UUID_LEN)
            obj = db.getDatasetObjByUuid(dsetUuid)
            shape_item = db.getShapeItemByDsetObj(obj)
            self.assertTrue('class' in shape_item)
            self.assertEqual(shape_item['class'], 'H5S_SCALAR')
            
    def testReadNullSpaceAttribute(self):
        filepath = getFile('null_space_attr.h5', 'readnullspaceattr.h5')
         
        with Hdf5db(filepath, app_logger=self.log) as db:
            rootUuid = db.getUUIDByPath('/')
            self.assertEqual(len(rootUuid), UUID_LEN)
            item = db.getAttributeItem("groups", rootUuid, "attr1") 
            self.assertTrue('shape' in item)
            shape_item = item['shape']
            self.assertTrue('class' in shape_item)
            self.assertEqual(shape_item['class'], 'H5S_NULL')

    def testReadAttribute(self):
        # getAttributeItemByUuid
        item = None
        filepath = getFile('tall.h5', 'readattribute.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            rootUuid = db.getUUIDByPath('/')
            self.assertEqual(len(rootUuid), UUID_LEN)
            item = db.getAttributeItem("groups", rootUuid, "attr1")   

    def testWriteScalarAttribute(self):
        # getAttributeItemByUuid
        item = None
        filepath = getFile('empty.h5', 'writescalarattribute.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            root_uuid = db.getUUIDByPath('/')
            dims = ()
            datatype = "H5T_STD_I32LE"
            value = 42
            db.createAttribute("groups", root_uuid, "A1", dims, datatype, value)
            item = db.getAttributeItem("groups", root_uuid, "A1")  
            self.assertEqual(item['name'], "A1")
            self.assertEqual(item['value'], 42)
            now = int(time.time())
            self.assertTrue(item['ctime'] > now - 5)
            self.assertTrue(item['mtime'] > now - 5)
            shape = item['shape']
            self.assertEqual(shape['class'], 'H5S_SCALAR')
            item_type = item['type']

            self.assertEqual(item_type['class'], 'H5T_INTEGER')
            self.assertEqual(item_type['base'], 'H5T_STD_I32LE')
            self.assertEqual(len(item_type.keys()), 2)  # just two keys should be returned


    def testWriteFixedStringAttribute(self):
        # getAttributeItemByUuid
        item = None
        filepath = getFile('empty.h5', 'writefixedstringattribute.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            root_uuid = db.getUUIDByPath('/')
            dims = ()
            datatype = { 'charSet':   'H5T_CSET_ASCII',
                     'class':  'H5T_STRING',
                     'strPad': 'H5T_STR_NULLPAD',
                     'length': 13}
            value = "Hello, world!"
            db.createAttribute("groups", root_uuid, "A1", dims, datatype, value)   
            item = db.getAttributeItem("groups", root_uuid, "A1")
            self.assertEqual(item['name'], "A1")
            self.assertEqual(item['value'], "Hello, world!")
            now = int(time.time())
            self.assertTrue(item['ctime'] > now - 5)
            self.assertTrue(item['mtime'] > now - 5)
            shape = item['shape']
            self.assertEqual(shape['class'], 'H5S_SCALAR')
            item_type = item['type']
            self.assertEqual(item_type['length'], 13)
            self.assertEqual(item_type['class'], 'H5T_STRING')
            self.assertEqual(item_type['strPad'], 'H5T_STR_NULLPAD')
            self.assertEqual(item_type['charSet'], 'H5T_CSET_ASCII')


    def testWriteFixedNullTermStringAttribute(self):
        # getAttributeItemByUuid
        item = None
        filepath = getFile('empty.h5', 'writefixednulltermstringattribute.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            root_uuid = db.getUUIDByPath('/')
            dims = ()
            datatype = { 'charSet':   'H5T_CSET_ASCII',
                     'class':  'H5T_STRING',
                     'strPad': 'H5T_STR_NULLTERM',
                     'length': 13}
            value = b"Hello, world!"

            # write the attribute
            db.createAttribute("groups", root_uuid, "A1", dims, datatype, value)   
            # read it back
            item = db.getAttributeItem("groups", root_uuid, "A1")

            self.assertEqual(item['name'], "A1")
            # the following compare fails - see issue #34
            #self.assertEqual(item['value'], "Hello, world!")
            now = int(time.time())
            self.assertTrue(item['ctime'] > now - 5)
            self.assertTrue(item['mtime'] > now - 5)
            shape = item['shape']
            self.assertEqual(shape['class'], 'H5S_SCALAR')
            item_type = item['type']
            self.assertEqual(item_type['length'], 13)
            self.assertEqual(item_type['class'], 'H5T_STRING')
            # NULLTERM get's converted to NULLPAD since the numpy dtype does not
            # support other padding conventions.
            self.assertEqual(item_type['strPad'], 'H5T_STR_NULLPAD')
            self.assertEqual(item_type['charSet'], 'H5T_CSET_ASCII')

    def testWriteVlenStringAttribute(self):
        # getAttributeItemByUuid
        item = None
        filepath = getFile('empty.h5', 'writevlenstringattribute.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            root_uuid = db.getUUIDByPath('/')
            dims = ()
            datatype = { 'charSet':   'H5T_CSET_ASCII',
                     'class':  'H5T_STRING',
                     'strPad': 'H5T_STR_NULLTERM',
                     'length': 'H5T_VARIABLE' }
        
            #value = np.string_("Hello, world!")
            value = "Hello, world!"
            db.createAttribute("groups", root_uuid, "A1", dims, datatype, value)   
            item = db.getAttributeItem("groups", root_uuid, "A1")
            self.assertEqual(item['name'], "A1")
            self.assertEqual(item['value'], "Hello, world!")
            now = int(time.time())
            self.assertTrue(item['ctime'] > now - 5)
            self.assertTrue(item['mtime'] > now - 5)
            shape = item['shape']
            self.assertEqual(shape['class'], 'H5S_SCALAR')
            item_type = item['type']
            self.assertEqual(item_type['class'], 'H5T_STRING')
            self.assertEqual(item_type['strPad'], 'H5T_STR_NULLTERM')
            self.assertEqual(item_type['charSet'], 'H5T_CSET_ASCII')
            self.assertEqual(item_type['length'], 'H5T_VARIABLE')

    def testReadVlenStringDataset(self):
        item = None
        filepath = getFile('vlen_string_dset.h5', 'vlen_string_dset.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            dset_uuid = db.getUUIDByPath('/DS1')
            item = db.getDatasetItemByUuid(dset_uuid)   
            shape = item['shape']
            self.assertEqual(shape['class'], 'H5S_SIMPLE')
            dims = shape['dims']
            self.assertEqual(len(dims), 1)
            self.assertEqual(dims[0], 4)
            item_type = item['type']
            self.assertEqual(item_type['class'], 'H5T_STRING')
            # actual padding is SPACEPAD - See issue #32
            self.assertEqual(item_type['strPad'], 'H5T_STR_NULLTERM')
            self.assertEqual(item_type['charSet'], 'H5T_CSET_ASCII')
            self.assertEqual(item_type['length'], 'H5T_VARIABLE')
            row = db.getDatasetValuesByUuid(dset_uuid, (slice(0, 1),))
            self.assertEqual(row, ['Parting'])

    def testReadVlenStringDataset_utc(self):
        item = None
        filepath = getFile('vlen_string_dset_utc.h5', 'vlen_string_dset_utc.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            dset_uuid = db.getUUIDByPath('/ds1')
            item = db.getDatasetItemByUuid(dset_uuid)   
            shape = item['shape']
            self.assertEqual(shape['class'], 'H5S_SIMPLE')
            dims = shape['dims']
            self.assertEqual(len(dims), 1)
            self.assertEqual(dims[0], 2293)
            item_type = item['type']
            self.assertEqual(item_type['class'], 'H5T_STRING')
            self.assertEqual(item_type['strPad'], 'H5T_STR_NULLTERM')
            self.assertEqual(item_type['charSet'], 'H5T_CSET_ASCII')
            self.assertEqual(item_type['length'], 'H5T_VARIABLE')
            # next line throws conversion error - see issue #19
            #row = db.getDatasetValuesByUuid(dset_uuid, (slice(0, 1),))
            
    def testReadFixedStringDataset(self):
        item = None
        filepath = getFile('fixed_string_dset.h5', 'fixed_string_dset.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            dset_uuid = db.getUUIDByPath('/DS1')
            item = db.getDatasetItemByUuid(dset_uuid)   
            shape = item['shape']
            self.assertEqual(shape['class'], 'H5S_SIMPLE')
            dims = shape['dims']
            self.assertEqual(len(dims), 1)
            self.assertEqual(dims[0], 4)
            item_type = item['type']
            self.assertEqual(item_type['class'], 'H5T_STRING')
            self.assertEqual(item_type['strPad'], 'H5T_STR_NULLPAD')
            self.assertEqual(item_type['charSet'], 'H5T_CSET_ASCII')
            self.assertEqual(item_type['length'], 7)
            row = db.getDatasetValuesByUuid(dset_uuid)
            self.assertEqual(row, ["Parting", "is such", "sweet", "sorrow."])
            row = db.getDatasetValuesByUuid(dset_uuid, (slice(0,1),))
            self.assertEqual(row,  ["Parting",])
            row = db.getDatasetValuesByUuid(dset_uuid, (slice(2,3),))
            self.assertEqual(row,  ["sweet",])

    def testReadFixedStringDatasetBinary(self):
        item = None
        filepath = getFile('fixed_string_dset.h5', 'fixed_string_dset.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            dset_uuid = db.getUUIDByPath('/DS1')
            item = db.getDatasetItemByUuid(dset_uuid)   
            shape = item['shape']
            self.assertEqual(shape['class'], 'H5S_SIMPLE')
            dims = shape['dims']
            self.assertEqual(len(dims), 1)
            self.assertEqual(dims[0], 4)
            item_type = item['type']
            self.assertEqual(item_type['class'], 'H5T_STRING')
            self.assertEqual(item_type['strPad'], 'H5T_STR_NULLPAD')
            self.assertEqual(item_type['charSet'], 'H5T_CSET_ASCII')
            self.assertEqual(item_type['length'], 7)
            row = db.getDatasetValuesByUuid(dset_uuid, format="binary")
            self.assertEqual(row, b"Partingis suchsweet\x00\x00sorrow.")
            row = db.getDatasetValuesByUuid(dset_uuid, (slice(0,1),), format="binary")
            self.assertEqual(row,  b"Parting")
            row = db.getDatasetValuesByUuid(dset_uuid, (slice(2,3),), format="binary")
            self.assertEqual(row,  b"sweet\x00\x00")


    def testWriteVlenUnicodeAttribute(self):
        # getAttributeItemByUuid
        item = None
        filepath = getFile('empty.h5', 'writevlenunicodeattribute.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            root_uuid = db.getUUIDByPath('/')
            dims = ()
            datatype = { 'charSet':   'H5T_CSET_UTF8',
                     'class':  'H5T_STRING',
                     'strPad': 'H5T_STR_NULLTERM',
                     'length': 'H5T_VARIABLE' }
            value =  u'\u6b22\u8fce\u63d0\u4ea4\u5fae\u535a\u641c\u7d22\u4f7f\u7528\u53cd\u9988\uff0c\u8bf7\u76f4\u63a5'
            db.createAttribute("groups", root_uuid, "A1", dims, datatype, value)
            item = db.getAttributeItem("groups", root_uuid, "A1")   

            self.assertEqual(item['name'], "A1")
            self.assertEqual(item['value'], value)
            now = int(time.time())
            self.assertTrue(item['ctime'] > now - 5)
            self.assertTrue(item['mtime'] > now - 5)
            shape = item['shape']
            self.assertEqual(shape['class'], 'H5S_SCALAR')
            item_type = item['type']
            self.assertEqual(item_type['class'], 'H5T_STRING')
            self.assertEqual(item_type['strPad'], 'H5T_STR_NULLTERM')
            self.assertEqual(item_type['charSet'], 'H5T_CSET_UTF8')
            self.assertEqual(item_type['length'], 'H5T_VARIABLE')


    def testWriteIntAttribute(self):
        # getAttributeItemByUuid
        item = None
        filepath = getFile('empty.h5', 'writeintattribute.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            root_uuid = db.getUUIDByPath('/')
            dims = (5,)
            datatype = "H5T_STD_I16LE"
            value = [2, 3, 5, 7, 11]
            db.createAttribute("groups", root_uuid, "A1", dims, datatype, value)
            item = db.getAttributeItem("groups", root_uuid, "A1")  
            self.assertEqual(item['name'], "A1")
            self.assertEqual(item['value'], [2, 3, 5, 7, 11])
            now = int(time.time())
            self.assertTrue(item['ctime'] > now - 5)
            self.assertTrue(item['mtime'] > now - 5)
            shape = item['shape']
            self.assertEqual(shape['class'], 'H5S_SIMPLE')
            item_type = item['type']
            self.assertEqual(item_type['class'], 'H5T_INTEGER')
            self.assertEqual(item_type['base'], 'H5T_STD_I16LE')

    def testCreateReferenceAttribute(self):
        filepath = getFile('empty.h5', 'createreferencedataset.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            root_uuid = db.getUUIDByPath('/')

            dims = ()  # if no space in body, default to scalar
            rsp = db.createDataset("H5T_STD_I64LE", dims, max_shape=None, creation_props=None)
            dset_uuid = rsp['id']
            db.linkObject(root_uuid, dset_uuid, 'DS1')

            dims = (1,)
            datatype = { "class": "H5T_REFERENCE", "base": "H5T_STD_REF_OBJ"}
            ds1_ref = "datasets/" + dset_uuid
            value = [ds1_ref,]
            db.createAttribute("groups", root_uuid, "A1", dims, datatype, value)
            item = db.getAttributeItem("groups", root_uuid, "A1")   

            attr_type = item['type']
            self.assertEqual(attr_type["class"], "H5T_REFERENCE")
            self.assertEqual(attr_type["base"], "H5T_STD_REF_OBJ")
            attr_value = item['value']
            self.assertEqual(len(attr_value), 1)
            self.assertEqual(attr_value[0], ds1_ref)

    def testCreateVlenReferenceAttribute(self):
        filepath = getFile('empty.h5', 'createreferenceattribute.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            root_uuid = db.getUUIDByPath('/')

            dims = ()  # if no space in body, default to scalar
            rsp = db.createDataset("H5T_STD_I64LE", dims, max_shape=None, creation_props=None)
            dset_uuid = rsp['id']
            db.linkObject(root_uuid, dset_uuid, 'DS1')

            dims = (1,)
            datatype = {"class": "H5T_VLEN",
                "base": { "class": "H5T_REFERENCE", "base": "H5T_STD_REF_OBJ"}
            }
            ds1_ref = "datasets/" + dset_uuid
            value = [[ds1_ref,],]
            db.createAttribute("groups", root_uuid, "A1", dims, datatype, value)
            item = db.getAttributeItem("groups", root_uuid, "A1")   

            attr_type = item['type']
            self.assertEqual(attr_type["class"], "H5T_VLEN")
            base_type = attr_type["base"]
            # todo - this should be H5T_REFERENCE, not H5T_OPAQUE
            # See h5py issue: https://github.com/h5py/h5py/issues/553
            import h5py
            # test based on h5py version until we change install requirements
            if h5py.version.version_tuple[1] >= 6:  
                self.assertEqual(base_type["class"], "H5T_REFERENCE")
            else:
                self.assertEqual(base_type["class"], "H5T_OPAQUE")

    def testCreateReferenceListAttribute(self):
        filepath = getFile('empty.h5', 'createreferencelistattribute.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            root_uuid = db.getUUIDByPath('/')

            dims = (10,)

            rsp = db.createDataset("H5T_STD_I64LE", dims, max_shape=None, creation_props=None)
            dset_uuid = rsp['id']
            db.linkObject(root_uuid, dset_uuid, 'dset')

            rsp = db.createDataset("H5T_STD_I64LE", dims, max_shape=None, creation_props=None)
            xscale_uuid = rsp['id']
            nullterm_string_type =  {
                        "charSet": "H5T_CSET_ASCII",
                        "class": "H5T_STRING",
                        "length": 16,
                        "strPad": "H5T_STR_NULLTERM"
            }
            scalar_dims = ()
            db.createAttribute(
                "datasets", xscale_uuid, "CLASS", scalar_dims,
                nullterm_string_type, "DIMENSION_SCALE")  
            db.linkObject(root_uuid, xscale_uuid, 'xscale')


            ref_dims = (1,)
            datatype = {"class": "H5T_VLEN",
                "base": { "class": "H5T_REFERENCE", "base": "H5T_STD_REF_OBJ"}
            }
            xscale_ref = "datasets/" + xscale_uuid
            value = [(xscale_ref,),]
            db.createAttribute("datasets", dset_uuid, "DIMENSION_LIST", ref_dims, datatype, value)
            item = db.getAttributeItem("datasets", dset_uuid, "DIMENSION_LIST")

            attr_type = item['type']
            self.assertEqual(attr_type["class"], "H5T_VLEN")
            base_type = attr_type["base"]
            # todo - this should be H5T_REFERENCE, not H5T_OPAQUE
            self.assertEqual(base_type["class"], "H5T_REFERENCE")



    def testReadCommittedType(self):
        filepath = getFile('committed_type.h5', 'readcommitted_type.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            root_uuid = db.getUUIDByPath('/')
            type_uuid = db.getUUIDByPath('/Sensor_Type')
            item = db.getCommittedTypeItemByUuid(type_uuid)
            self.assertTrue('type' in item)
            item_type = item['type']
            self.assertTrue(item_type['class'], 'H5T_COMPOUND')
            ds1_uuid = db.getUUIDByPath('/DS1')
            item = db.getDatasetItemByUuid(ds1_uuid)
            shape = item['shape']
            self.assertEqual(shape['class'], 'H5S_SIMPLE')
            dims = shape['dims']
            self.assertEqual(len(dims), 1)
            self.assertEqual(dims[0], 4)
            item_type = item['type']
            self.assertTrue('class' in item_type)
            self.assertEqual(item_type['class'], 'H5T_COMPOUND')
            self.assertTrue('uuid' in item_type)
            self.assertEqual(item_type['uuid'], type_uuid)

            item = db.getAttributeItem("groups", root_uuid, "attr1")   
            shape = item['shape']
            self.assertEqual(shape['class'], 'H5S_SCALAR')
            item_type = item['type']
            self.assertTrue('class' in item_type)
            self.assertEqual(item_type['class'], 'H5T_COMPOUND')
            self.assertTrue('uuid' in item_type)
            self.assertEqual(item_type['uuid'], type_uuid)


    def testWriteCommittedType(self):
        filepath = getFile('empty.h5', 'writecommittedtype.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            root_uuid = db.getUUIDByPath('/')
            datatype = { 'charSet':   'H5T_CSET_ASCII',
                     'class':  'H5T_STRING',
                     'strPad': 'H5T_STR_NULLTERM',
                     'length': 15}
            item = db.createCommittedType(datatype)
            type_uuid = item['id']
            item = db.getCommittedTypeItemByUuid(type_uuid)
            self.assertEqual(item['id'], type_uuid)
            self.assertEqual(item['attributeCount'], 0)
            now = int(time.time())
            self.assertTrue(item['ctime'] > now - 5)
            self.assertTrue(item['mtime'] > now - 5)
            self.assertEqual(len(item['alias']), 0)  # anonymous, so no alias

            item_type = item['type']

            self.assertEqual(item_type['class'], 'H5T_STRING')
            self.assertEqual(item_type['strPad'], 'H5T_STR_NULLPAD')
            self.assertEqual(item_type['charSet'], 'H5T_CSET_ASCII')
            self.assertEqual(item_type['length'], 15)

    def testWriteCommittedCompoundType(self):
        filepath = getFile('empty.h5', 'writecommittedcompoundtype.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            root_uuid = db.getUUIDByPath('/')
            datatype = {'class': 'H5T_COMPOUND',
                        'fields': [] }

            fixed_str_type = { 'charSet':   'H5T_CSET_ASCII',
                     'class':  'H5T_STRING',
                     'strPad': 'H5T_STR_NULLTERM',
                     'length': 15}

            var_str_type = {
                     "charSet": "H5T_CSET_ASCII",
                     "class": "H5T_STRING",
                     "length": "H5T_VARIABLE",
                     "strPad": "H5T_STR_NULLTERM" }
            type_fields = []
            type_fields.append({'name': 'field_1', 'type': 'H5T_STD_I64BE' })
            type_fields.append({'name': 'field_2', 'type': 'H5T_IEEE_F64BE' })
            type_fields.append({'name': 'field_3', 'type': fixed_str_type })
            type_fields.append({'name': 'field_4', 'type': var_str_type })
            datatype['fields'] = type_fields


            item = db.createCommittedType(datatype)
            type_uuid = item['id']
            item = db.getCommittedTypeItemByUuid(type_uuid)
            self.assertEqual(item['id'], type_uuid)
            self.assertEqual(item['attributeCount'], 0)
            now = int(time.time())
            self.assertTrue(item['ctime'] > now - 5)
            self.assertTrue(item['mtime'] > now - 5)
            self.assertEqual(len(item['alias']), 0)  # anonymous, so no alias

            item_type = item['type']

            self.assertEqual(item_type['class'], 'H5T_COMPOUND')
            fields = item_type['fields']
            self.assertEqual(len(fields), 4)
            # todo - the last field class should be H5T_STRING, but it is getting
            # saved to HDF5 as Opaque - see: https://github.com/h5py/h5py/issues/613
            # this is fixed in h5py v. 2.6.0 - check the version until 2.6.0 becomes
            # available via pip and anaconda.
            import h5py
            if h5py.version.version_tuple[1] >= 6:  
                field_classes = ('H5T_INTEGER', 'H5T_FLOAT', 'H5T_STRING', 'H5T_STRING')
            else:
                field_classes = ('H5T_INTEGER', 'H5T_FLOAT', 'H5T_STRING', 'H5T_OPAQUE')
            for i in range(4):
                field = fields[i]
                self.assertEqual(field['name'], 'field_' + str(i+1))
                field_type = field['type']
                self.assertEqual(field_type['class'], field_classes[i])



    def testToRef(self):

        filepath = getFile('empty.h5', 'toref.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            type_item = {'order': 'H5T_ORDER_LE', 'base_size': 1, 'class': 'H5T_INTEGER', 'base': 'H5T_STD_I8LE', 'size': 1}
            data_list = [2, 3, 5, 7, 11]
            ref_value = db.toRef(1, type_item, data_list)
            self.assertEqual(ref_value, data_list)

            type_item =  { "charSet": "H5T_CSET_ASCII",
                           "class": "H5T_STRING",
                           "length": 8,
                           "strPad": "H5T_STR_NULLPAD" }
            data_list = [ "Hypertext", "as", "engine", "of", "state" ]
            ref_value = db.toRef(1, type_item, data_list)


    def testToTuple(self):
        filepath = getFile('empty.h5', 'totuple.h5')
        data1d = [1,2,3]
        data2d = [[1,2],[3,4]] 
        data3d = [[[1,2],[3,4]], [[5,6],[7,8]]]
        with Hdf5db(filepath, app_logger=self.log) as db:
            self.assertEqual(db.toTuple(1, data1d ), [1,2,3] )
            self.assertEqual(db.toTuple(2, data2d ), [[1,2],[3,4]]  )
            self.assertEqual(db.toTuple(1, data2d ), [(1,2),(3,4)]  )
            self.assertEqual(db.toTuple(3, data3d), [[[1,2],[3,4]], [[5,6],[7,8]]] )
            self.assertEqual(db.toTuple(2, data3d), [[(1,2),(3,4)], [(5,6),(7,8)]]  )
            self.assertEqual(db.toTuple(1, data3d), [((1,2),(3,4)), ((5,6),(7,8))]  )
    
               
    def testBytesArrayToList(self):
        filepath = getFile('empty.h5', 'bytestostring.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            
            val = db.bytesArrayToList(b'Hello')
            self.assertTrue(type(val) is str)
            val = db.bytesArrayToList([b'Hello',])
            self.assertEqual(len(val), 1)
            self.assertTrue(type(val[0]) is str)
            self.assertEqual(val[0], 'Hello')
            
            import numpy as np
            
            data = np.array([b'Hello'])
            val = db.bytesArrayToList(data)
            self.assertEqual(len(val), 1)
            self.assertTrue(type(val[0]) is str)
            self.assertEqual(val[0], 'Hello')
    
            
    def testGetDataValue(self):
        # typeItem, value, dimension=0, dims=None):
        filepath = getFile('empty.h5', 'bytestostring.h5')
        string_type = { 'charSet':   'H5T_CSET_ASCII',
                     'class':  'H5T_STRING',
                     'strPad': 'H5T_STR_NULLTERM',
                     'length': 'H5T_VARIABLE' }
        
        with Hdf5db(filepath, app_logger=self.log) as db:
            
            import numpy as np
            
            data = np.array([b'Hello'])
            val = db.getDataValue(string_type, data, dimension=1,dims=(1,))
            self.assertTrue(type(val[0]) is str)
                 

    def testGetAclDataset(self):
        filepath = getFile('tall.h5', 'getacldataset.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            d111_uuid = db.getUUIDByPath('/g1/g1.1/dset1.1.1')
            num_acls = db.getNumAcls(d111_uuid)
            self.assertEqual(num_acls, 0)
            acl_dset = db.getAclDataset(d111_uuid, create=True)
            self.assertTrue(acl_dset.name.endswith(d111_uuid))
            self.assertEqual(len(acl_dset.dtype), 7)
            self.assertEqual(len(acl_dset.shape), 1)
            self.assertEqual(acl_dset.shape[0], 0)
            num_acls = db.getNumAcls(d111_uuid)
            self.assertEqual(num_acls, 0)

    def testSetAcl(self):
        filepath = getFile('tall.h5', 'setacl.h5')
        user1 = 123
        user2 = 456
        with Hdf5db(filepath, app_logger=self.log) as db:
            d111_uuid = db.getUUIDByPath('/g1/g1.1/dset1.1.1')
            num_acls = db.getNumAcls(d111_uuid)
            self.assertEqual(num_acls, 0)

            # add read/write acl for user1
            acl_user1 = db.getAcl(d111_uuid, user1)

            self.assertEqual(acl_user1['userid'], 0)
            acl_user1['userid'] = user1
            acl_user1['readACL'] = 0
            acl_user1['updateACL'] = 0
            num_acls = db.getNumAcls(d111_uuid)
            self.assertEqual(num_acls, 0)

            db.setAcl(d111_uuid, acl_user1)
            acl = db.getAcl(d111_uuid, user1)
            num_acls = db.getNumAcls(d111_uuid)
            self.assertEqual(num_acls, 1)


            # add read-only acl for user2
            acl_user2 = db.getAcl(d111_uuid, user2)
            self.assertEqual(acl_user2['userid'], 0)
            acl_user2['userid'] = user2
            acl_user2['create'] = 0
            acl_user2['read'] = 1
            acl_user2['update'] = 0
            acl_user2['delete'] = 0
            acl_user2['readACL'] = 0
            acl_user2['updateACL'] = 0
            db.setAcl(d111_uuid, acl_user2)
            num_acls = db.getNumAcls(d111_uuid)
            self.assertEqual(num_acls, 2)

            # fetch and verify acls
            acl = db.getAcl(d111_uuid, user1)
            self.assertEqual(acl['userid'], user1)
            self.assertEqual(acl['create'], 1)
            self.assertEqual(acl['read'], 1)
            self.assertEqual(acl['update'], 1)
            self.assertEqual(acl['delete'], 1)
            self.assertEqual(acl['readACL'], 0)
            self.assertEqual(acl['updateACL'], 0)

            acl = db.getAcl(d111_uuid, user2)
            self.assertEqual(acl['userid'], user2)
            self.assertEqual(acl['create'], 0)
            self.assertEqual(acl['read'], 1)
            self.assertEqual(acl['update'], 0)
            self.assertEqual(acl['delete'], 0)
            self.assertEqual(acl['readACL'], 0)
            self.assertEqual(acl['updateACL'], 0)

            num_acls = db.getNumAcls(d111_uuid)
            self.assertEqual(num_acls, 2)

            # get acl data_list
            acls = db.getAcls(d111_uuid)
            self.assertEqual(len(acls), 2)


    def testRootAcl(self):
        filepath = getFile('tall.h5', 'rootacl.h5')
        user1 = 123
        with Hdf5db(filepath, app_logger=self.log) as db:
            root_uuid = db.getUUIDByPath('/')
            d111_uuid = db.getUUIDByPath('/g1/g1.1/dset1.1.1')
            num_acls = db.getNumAcls(d111_uuid)
            self.assertEqual(num_acls, 0)

            # add read/write acl for user1 at root
            acl_root = db.getAcl(root_uuid, 0)
            self.assertEqual(acl_root['userid'], 0)
            acl_root['create'] = 0
            acl_root['read'] = 1
            acl_root['update'] = 0
            acl_root['delete'] = 0
            acl_root['readACL'] = 0
            acl_root['updateACL'] = 0
            num_acls = db.getNumAcls(root_uuid)
            self.assertEqual(num_acls, 0)

            db.setAcl(root_uuid, acl_root)
            num_acls = db.getNumAcls(root_uuid)
            self.assertEqual(num_acls, 1)

            acl = db.getAcl(d111_uuid, user1)
            num_acls = db.getNumAcls(d111_uuid)  # this will fetch the root acl
            self.assertEqual(num_acls, 0)
            self.assertEqual(acl['userid'], 0)
            self.assertEqual(acl['create'], 0)
            self.assertEqual(acl['read'], 1)
            self.assertEqual(acl['update'], 0)
            self.assertEqual(acl['delete'], 0)
            self.assertEqual(acl['readACL'], 0)
            self.assertEqual(acl['updateACL'], 0)
            
    def testGetEvalStr(self):
        queries = { "date == 23": "rows['date'] == 23",
                    "wind == b'W 5'": "rows['wind'] == b'W 5'",
                    "temp > 61": "rows['temp'] > 61",
                    "(date >=22) & (date <= 24)": "(rows['date'] >=22) & (rows['date'] <= 24)",
                    "(date == 21) & (temp > 70)": "(rows['date'] == 21) & (rows['temp'] > 70)",
                    "(wind == b'E 7') | (wind == b'S 7')": "(rows['wind'] == b'E 7') | (rows['wind'] == b'S 7')" }
                    
        fields = ["date", "wind", "temp" ]  
        filepath = getFile('empty.h5', 'getevalstring.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            
            for query in queries.keys():
                eval_str = db._getEvalStr(query, fields)
                self.assertEqual(eval_str, queries[query])
                #print(query, "->", eval_str)
                
    def testBadQuery(self):
        queries = ( "foobar",    # no variable used
                "wind = b'abc",  # non-closed literal
                "(wind = b'N') & (temp = 32",  # missing paren
                "foobar > 42",                 # invalid field name
                "import subprocess; subprocess.call(['ls', '/'])")  # injection attack
                         
        fields = ("date", "wind", "temp" )  
        filepath = getFile('empty.h5', 'badquery.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            
            for query in queries:
                try:
                    eval_str = db._getEvalStr(query, fields)
                    self.assertTrue(False)  # shouldn't get here
                except IOError as e:
                    pass  # ok
                    
    def testInjectionBlock(self):
        queries = ( 
            "import subprocess; subprocess.call(['ls', '/'])", ) # injection attack
                         
        fields = ("import", "subprocess", "call" ) 
        filepath = getFile('empty.h5', 'injectionblock.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            
            for query in queries:
                try:
                    eval_str = db._getEvalStr(query, fields)
                    self.assertTrue(False)  # shouldn't get here
                except IOError as e:
                    pass  # ok
             
 


if __name__ == '__main__':
    #setup test files

    unittest.main()
