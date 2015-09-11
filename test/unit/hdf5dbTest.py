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
import errno
import os.path as op
import stat
import logging
import shutil

sys.path.append('../../lib')
from hdf5db import Hdf5db

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
            self.failUnlessEqual(e.errno, errno.ENXIO)
            self.failUnlessEqual(e.strerror, "file not found")
            
    def testInvalidFile(self):       
        filepath = getFile('notahdf5file.h5', 'notahdf5file.h5')
        try:
            with Hdf5db(filepath, app_logger=self.log) as db:
                self.assertTrue(False)  # shouldn't get here
        except IOError as e:
            self.failUnlessEqual(e.errno, errno.EINVAL)
            self.failUnlessEqual(e.strerror, "not an HDF5 file")              
             
                
    def testGetUUIDByPath(self):
        # get test file
        g1Uuid = None
        filepath = getFile('tall.h5', 'getuuidbypath.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            g1Uuid = db.getUUIDByPath('/g1')
            self.failUnlessEqual(len(g1Uuid), UUID_LEN)
            obj = db.getObjByPath('/g1')
            self.failUnlessEqual(obj.name, '/g1')
            for name in obj:
                g = obj[name]
            g1links = db.getLinkItems(g1Uuid)
            self.failUnlessEqual(len(g1links), 2)
            for item in g1links:
                self.failUnlessEqual(len(item['id']), UUID_LEN)
          
        # end of with will close file
        # open again and verify we can get obj by name
        with Hdf5db(filepath, app_logger=self.log) as db:
            obj = db.getGroupObjByUuid(g1Uuid) 
            g1 = db.getObjByPath('/g1')
            self.failUnlessEqual(obj, g1)
            
    def testGetCounts(self):
        filepath = getFile('tall.h5', 'testgetcounts_tall.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            cnt = db.getNumberOfGroups()
            self.failUnlessEqual(cnt, 6)
            cnt = db.getNumberOfDatasets()
            self.failUnlessEqual(cnt, 4)
            cnt = db.getNumberOfDatatypes()
            self.failUnlessEqual(cnt, 0)
        
        filepath = getFile('empty.h5', 'testgetcounts_empty.h5')    
        with Hdf5db(filepath, app_logger=self.log) as db:
            cnt = db.getNumberOfGroups()
            self.failUnlessEqual(cnt, 1)
            cnt = db.getNumberOfDatasets()
            self.failUnlessEqual(cnt, 0)
            cnt = db.getNumberOfDatatypes()
            self.failUnlessEqual(cnt, 0)
            
               
    def testGroupOperations(self):
        # get test file
        filepath = getFile('tall.h5', 'tall_del_g11.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            rootuuid = db.getUUIDByPath('/')
            root = db.getGroupObjByUuid(rootuuid)
            self.failUnlessEqual('/', root.name)
            rootLinks = db.getLinkItems(rootuuid)
            self.failUnlessEqual(len(rootLinks), 2)
            g1uuid = db.getUUIDByPath("/g1")
            self.failUnlessEqual(len(g1uuid), UUID_LEN)
            g1Links = db.getLinkItems(g1uuid)
            self.failUnlessEqual(len(g1Links), 2)
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
            self.failUnlessEqual(len(g1Uuid), UUID_LEN)
            obj = db.getObjByPath('/g1')
            self.failUnlessEqual(obj.name, '/g1')
    
        # end of with will close file
        # open again and verify we can get obj by name
        with Hdf5db(filepath, app_logger=self.log) as db:
            obj = db.getGroupObjByUuid(g1Uuid) 
            g1 = db.getObjByPath('/g1')
            self.failUnlessEqual(obj, g1)
            g1links = db.getLinkItems(g1Uuid)
            self.failUnlessEqual(len(g1links), 2)
            for item in g1links:
                self.failUnlessEqual(len(item['id']), UUID_LEN)
                
    def testReadDataset(self):
         filepath = getFile('tall.h5', 'readdataset.h5')
         d111_values = None
         d112_values = None
         with Hdf5db(filepath, app_logger=self.log) as db:
            d111Uuid = db.getUUIDByPath('/g1/g1.1/dset1.1.1')
            self.failUnlessEqual(len(d111Uuid), UUID_LEN)
            d111_values = db.getDatasetValuesByUuid(d111Uuid)
            
            self.assertEqual(len(d111_values), 10)  
            for i in range(10):
                arr = d111_values[i]
                self.assertEqual(len(arr), 10)
                for j in range(10):
                    self.assertEqual(arr[j], i*j)
            
            d112Uuid = db.getUUIDByPath('/g1/g1.1/dset1.1.2')
            self.failUnlessEqual(len(d112Uuid), UUID_LEN)
            d112_values = db.getDatasetValuesByUuid(d112Uuid) 
            self.assertEqual(len(d112_values), 20)
            for i in range(20):
                self.assertEqual(d112_values[i], i)
                
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
            self.failUnlessEqual(item['attributeCount'], 0)
            type_item = item['type']
            self.failUnlessEqual(type_item['class'], 'H5T_INTEGER')
            self.failUnlessEqual(type_item['base'], 'H5T_STD_I64LE')
            shape_item = item['shape']
            self.failUnlessEqual(shape_item['class'], 'H5S_SIMPLE')
            self.failUnlessEqual(shape_item['dims'], (10,))
            
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
            self.failUnlessEqual(item['attributeCount'], 0)
            type_item = item['type']
            self.failUnlessEqual(type_item['class'], 'H5T_INTEGER')
            self.failUnlessEqual(type_item['base'], 'H5T_STD_I64LE')
            shape_item = item['shape']
            self.failUnlessEqual(shape_item['class'], 'H5S_SIMPLE')
            self.failUnlessEqual(shape_item['dims'], (10,10))
            self.assertTrue('maxdims' in shape_item)
            self.failUnlessEqual(shape_item['maxdims'], [0, 10])
                       
                
    def testReadZeroDimDataset(self):
         filepath = getFile('zerodim.h5', 'readzerodeimdataset.h5')
         d111_values = None
         d112_values = None
         with Hdf5db(filepath, app_logger=self.log) as db:
            dsetUuid = db.getUUIDByPath('/dset')
            self.failUnlessEqual(len(dsetUuid), UUID_LEN)
            dset_value = db.getDatasetValuesByUuid(dsetUuid)
            self.assertEqual(dset_value, 42)
            
    def testReadAttribute(self):
        # getAttributeItemByUuid
        item = None
        filepath = getFile('tall.h5', 'readattribute.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            rootUuid = db.getUUIDByPath('/')
            self.failUnlessEqual(len(rootUuid), UUID_LEN)
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
            self.failUnlessEqual(item['name'], "A1")
            self.failUnlessEqual(item['value'], 42)
            now = int(time.time())
            self.assertTrue(item['ctime'] > now - 5)
            self.assertTrue(item['mtime'] > now - 5)
            shape = item['shape']
            self.failUnlessEqual(shape['class'], 'H5S_SCALAR')
            item_type = item['type']
            
            self.failUnlessEqual(item_type['class'], 'H5T_INTEGER') 
            self.failUnlessEqual(item_type['base'], 'H5T_STD_I32LE') 
            self.failUnlessEqual(len(item_type.keys()), 2)  # just two keys should be returned
            
            
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
            self.failUnlessEqual(item['name'], "A1")
            self.failUnlessEqual(item['value'], "Hello, world!")
            now = int(time.time())
            self.assertTrue(item['ctime'] > now - 5)
            self.assertTrue(item['mtime'] > now - 5)
            shape = item['shape']
            self.failUnlessEqual(shape['class'], 'H5S_SCALAR')
            item_type = item['type']
            self.failUnlessEqual(item_type['length'], 13)
            self.failUnlessEqual(item_type['class'], 'H5T_STRING') 
            self.failUnlessEqual(item_type['strPad'], 'H5T_STR_NULLPAD')
            self.failUnlessEqual(item_type['charSet'], 'H5T_CSET_ASCII') 
            
            
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
            value = "Hello, world!"
            
            # write the attribute
            db.createAttribute("groups", root_uuid, "A1", dims, datatype, value)
            # read it back
            item = db.getAttributeItem("groups", root_uuid, "A1")
            
            self.failUnlessEqual(item['name'], "A1")
            # the following compare fails - see issue #34
            #self.failUnlessEqual(item['value'], "Hello, world!")
            now = int(time.time())
            self.assertTrue(item['ctime'] > now - 5)
            self.assertTrue(item['mtime'] > now - 5)
            shape = item['shape']
            self.failUnlessEqual(shape['class'], 'H5S_SCALAR')
            item_type = item['type']
            self.failUnlessEqual(item_type['length'], 13)
            self.failUnlessEqual(item_type['class'], 'H5T_STRING') 
            # NULLTERM get's converted to NULLPAD since the numpy dtype does not
            # support other padding conventions.
            self.failUnlessEqual(item_type['strPad'], 'H5T_STR_NULLPAD')  
            self.failUnlessEqual(item_type['charSet'], 'H5T_CSET_ASCII') 
            
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
            value = "Hello, world!"
            db.createAttribute("groups", root_uuid, "A1", dims, datatype, value)
            item = db.getAttributeItem("groups", root_uuid, "A1")
            self.failUnlessEqual(item['name'], "A1")
            self.failUnlessEqual(item['value'], "Hello, world!")
            now = int(time.time())
            self.assertTrue(item['ctime'] > now - 5)
            self.assertTrue(item['mtime'] > now - 5)
            shape = item['shape']
            self.failUnlessEqual(shape['class'], 'H5S_SCALAR')
            item_type = item['type']
            self.failUnlessEqual(item_type['class'], 'H5T_STRING') 
            self.failUnlessEqual(item_type['strPad'], 'H5T_STR_NULLTERM')
            self.failUnlessEqual(item_type['charSet'], 'H5T_CSET_ASCII') 
            self.failUnlessEqual(item_type['length'], 'H5T_VARIABLE')
            
    def testReadVlenStringDataset(self):
        item = None
        filepath = getFile('vlen_string_dset.h5', 'vlen_string_dset.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            dset_uuid = db.getUUIDByPath('/DS1')
            item = db.getDatasetItemByUuid(dset_uuid)
            shape = item['shape']
            self.failUnlessEqual(shape['class'], 'H5S_SIMPLE')
            dims = shape['dims']
            self.failUnlessEqual(len(dims), 1)
            self.failUnlessEqual(dims[0], 4)
            item_type = item['type']
            self.failUnlessEqual(item_type['class'], 'H5T_STRING')
            # actual padding is SPACEPAD - See issue #32
            self.failUnlessEqual(item_type['strPad'], 'H5T_STR_NULLTERM')
            self.failUnlessEqual(item_type['charSet'], 'H5T_CSET_ASCII') 
            self.failUnlessEqual(item_type['length'], 'H5T_VARIABLE')
            row = db.getDatasetValuesByUuid(dset_uuid, (slice(0, 1),))
            self.failUnlessEqual(row, ['Parting'])
            
    def testReadVlenStringDataset_utc(self):
        item = None
        filepath = getFile('vlen_string_dset_utc.h5', 'vlen_string_dset_utc.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            dset_uuid = db.getUUIDByPath('/ds1')
            item = db.getDatasetItemByUuid(dset_uuid)
            shape = item['shape']
            self.failUnlessEqual(shape['class'], 'H5S_SIMPLE')
            dims = shape['dims']
            self.failUnlessEqual(len(dims), 1)
            self.failUnlessEqual(dims[0], 2293)
            item_type = item['type']
            self.failUnlessEqual(item_type['class'], 'H5T_STRING') 
            self.failUnlessEqual(item_type['strPad'], 'H5T_STR_NULLTERM')
            self.failUnlessEqual(item_type['charSet'], 'H5T_CSET_ASCII') 
            self.failUnlessEqual(item_type['length'], 'H5T_VARIABLE')
            # next line throws conversion error - see issue #19
            #row = db.getDatasetValuesByUuid(dset_uuid, (slice(0, 1),))
            
            
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
           
            self.failUnlessEqual(item['name'], "A1")
            self.failUnlessEqual(item['value'], value)
            now = int(time.time())
            self.assertTrue(item['ctime'] > now - 5)
            self.assertTrue(item['mtime'] > now - 5)
            shape = item['shape']
            self.failUnlessEqual(shape['class'], 'H5S_SCALAR')
            item_type = item['type']
            self.failUnlessEqual(item_type['class'], 'H5T_STRING') 
            self.failUnlessEqual(item_type['strPad'], 'H5T_STR_NULLTERM')
            self.failUnlessEqual(item_type['charSet'], 'H5T_CSET_UTF8') 
            self.failUnlessEqual(item_type['length'], 'H5T_VARIABLE')
            
            
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
            self.failUnlessEqual(item['name'], "A1")
            self.failUnlessEqual(item['value'], [2, 3, 5, 7, 11])
            now = int(time.time())
            self.assertTrue(item['ctime'] > now - 5)
            self.assertTrue(item['mtime'] > now - 5)
            shape = item['shape']
            self.failUnlessEqual(shape['class'], 'H5S_SIMPLE')
            item_type = item['type']  
            self.failUnlessEqual(item_type['class'], 'H5T_INTEGER') 
            self.failUnlessEqual(item_type['base'], 'H5T_STD_I16LE') 
             
            
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
            self.failUnlessEqual(item['id'], type_uuid)
            self.failUnlessEqual(item['attributeCount'], 0)
            now = int(time.time())
            self.assertTrue(item['ctime'] > now - 5)
            self.assertTrue(item['mtime'] > now - 5)
            self.failUnlessEqual(len(item['alias']), 0)  # anonymous, so no alias
             
            item_type = item['type']
             
            self.failUnlessEqual(item_type['class'], 'H5T_STRING') 
            self.failUnlessEqual(item_type['strPad'], 'H5T_STR_NULLPAD')
            self.failUnlessEqual(item_type['charSet'], 'H5T_CSET_ASCII') 
            self.failUnlessEqual(item_type['length'], 15)  
            
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
            self.failUnlessEqual(item['id'], type_uuid)
            self.failUnlessEqual(item['attributeCount'], 0)
            now = int(time.time())
            self.assertTrue(item['ctime'] > now - 5)
            self.assertTrue(item['mtime'] > now - 5)
            self.failUnlessEqual(len(item['alias']), 0)  # anonymous, so no alias
             
            item_type = item['type']
             
            self.failUnlessEqual(item_type['class'], 'H5T_COMPOUND') 
            fields = item_type['fields']
            self.failUnlessEqual(len(fields), 4)
            # todo - the last field class should be H5T_STRING, but it is getting
            # saved to HDF5 as Opaque - see: https://github.com/h5py/h5py/issues/613 
            field_classes = ('H5T_INTEGER', 'H5T_FLOAT', 'H5T_STRING', 'H5T_OPAQUE')
            for i in range(4):
                field = fields[i]
                self.failUnlessEqual(field['name'], 'field_' + str(i+1))
                field_type = field['type']
                self.failUnlessEqual(field_type['class'], field_classes[i])
                
                 
        
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
        with Hdf5db(filepath, app_logger=self.log) as db:
            self.assertEqual(db.toTuple( [1,2,3] ), (1,2,3) ) 
            self.assertEqual(db.toTuple( [[1,2],[3,4]] ), ((1,2),(3,4))  )
            self.assertEqual(db.toTuple( ([1,2],[3,4]) ), ((1,2),(3,4))  )
            self.assertEqual(db.toTuple( [(1,2),(3,4)] ), ((1,2),(3,4))  )
            self.assertEqual(db.toTuple( [[[1,2],[3,4]], [[5,6],[7,8]]] ), 
                (((1,2),(3,4)), ((5,6),(7,8)))  )
                
    def testGetAclDataset(self):
        filepath = getFile('tall.h5', 'getacldataset.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            d111_uuid = db.getUUIDByPath('/g1/g1.1/dset1.1.1')
            num_acls = db.getNumAcls(d111_uuid)
            self.failUnlessEqual(num_acls, 0)
            acl_dset = db.getAclDataset(d111_uuid, create=True)
            self.assertTrue(acl_dset.name.endswith(d111_uuid))
            self.failUnlessEqual(len(acl_dset.dtype), 7)
            self.failUnlessEqual(len(acl_dset.shape), 1)
            self.failUnlessEqual(acl_dset.shape[0], 0)
            num_acls = db.getNumAcls(d111_uuid)
            self.failUnlessEqual(num_acls, 0)
            
    def testSetAcl(self):
        filepath = getFile('tall.h5', 'setacl.h5')
        user1 = 123
        user2 = 456
        with Hdf5db(filepath, app_logger=self.log) as db:
            d111_uuid = db.getUUIDByPath('/g1/g1.1/dset1.1.1')
            num_acls = db.getNumAcls(d111_uuid)
            self.failUnlessEqual(num_acls, 0)
            
            # add read/write acl for user1
            acl_user1 = db.getAcl(d111_uuid, user1)
            
            self.failUnlessEqual(acl_user1['userid'], 0)
            acl_user1['userid'] = user1
            acl_user1['readACL'] = 0
            acl_user1['updateACL'] = 0
            num_acls = db.getNumAcls(d111_uuid)       
            self.failUnlessEqual(num_acls, 0)
             
            db.setAcl(d111_uuid, acl_user1)
            acl = db.getAcl(d111_uuid, user1)
            num_acls = db.getNumAcls(d111_uuid)       
            self.failUnlessEqual(num_acls, 1)
            
            
            # add read-only acl for user2
            acl_user2 = db.getAcl(d111_uuid, user2)
            self.failUnlessEqual(acl_user2['userid'], 0)
            acl_user2['userid'] = user2
            acl_user2['create'] = 0
            acl_user2['read'] = 1
            acl_user2['update'] = 0
            acl_user2['delete'] = 0
            acl_user2['readACL'] = 0
            acl_user2['updateACL'] = 0
            db.setAcl(d111_uuid, acl_user2)
            num_acls = db.getNumAcls(d111_uuid)
            self.failUnlessEqual(num_acls, 2)
             
            # fetch and verify acls
            acl = db.getAcl(d111_uuid, user1)
            self.failUnlessEqual(acl['userid'], user1)
            self.failUnlessEqual(acl['create'], 1)
            self.failUnlessEqual(acl['read'], 1)
            self.failUnlessEqual(acl['update'], 1)
            self.failUnlessEqual(acl['delete'], 1)
            self.failUnlessEqual(acl['readACL'], 0)
            self.failUnlessEqual(acl['updateACL'], 0)
            
            acl = db.getAcl(d111_uuid, user2)
            self.failUnlessEqual(acl['userid'], user2)
            self.failUnlessEqual(acl['create'], 0)
            self.failUnlessEqual(acl['read'], 1)
            self.failUnlessEqual(acl['update'], 0)
            self.failUnlessEqual(acl['delete'], 0)
            self.failUnlessEqual(acl['readACL'], 0)
            self.failUnlessEqual(acl['updateACL'], 0)
            
            num_acls = db.getNumAcls(d111_uuid)
            self.failUnlessEqual(num_acls, 2)  
            
            # get acl data_list
            acls = db.getAcls(d111_uuid)
            self.failUnlessEqual(len(acls), 2)
             
            
    def testRootAcl(self):
        filepath = getFile('tall.h5', 'rootacl.h5')
        user1 = 123
        with Hdf5db(filepath, app_logger=self.log) as db:
            root_uuid = db.getUUIDByPath('/')
            d111_uuid = db.getUUIDByPath('/g1/g1.1/dset1.1.1')
            num_acls = db.getNumAcls(d111_uuid)
            self.failUnlessEqual(num_acls, 0)
            
            # add read/write acl for user1 at root
            acl_root = db.getAcl(root_uuid, 0)
            self.failUnlessEqual(acl_root['userid'], 0)
            acl_root['create'] = 0
            acl_root['read'] = 1
            acl_root['update'] = 0
            acl_root['delete'] = 0
            acl_root['readACL'] = 0
            acl_root['updateACL'] = 0
            num_acls = db.getNumAcls(root_uuid)
            self.failUnlessEqual(num_acls, 0)
             
            db.setAcl(root_uuid, acl_root)
            num_acls = db.getNumAcls(root_uuid)
            self.failUnlessEqual(num_acls, 1)
            
            acl = db.getAcl(d111_uuid, user1)
            num_acls = db.getNumAcls(d111_uuid)  # this will fetch the root acl
            self.failUnlessEqual(num_acls, 0)
            self.failUnlessEqual(acl['userid'], 0)
            self.failUnlessEqual(acl['create'], 0)
            self.failUnlessEqual(acl['read'], 1)
            self.failUnlessEqual(acl['update'], 0)
            self.failUnlessEqual(acl['delete'], 0)
            self.failUnlessEqual(acl['readACL'], 0)
            self.failUnlessEqual(acl['updateACL'], 0)
                  
             
             
if __name__ == '__main__':
    #setup test files
    
    unittest.main()
    

 



