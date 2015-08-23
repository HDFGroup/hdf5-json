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
    """        
    def testWriteFixedUTF8StringAttribute(self):
        # getAttributeItemByUuid
        item = None
        filepath = getFile('empty.h5', 'writefixedutf8stringattribute.h5')
        with Hdf5db(filepath, app_logger=self.log) as db:
            root_uuid = db.getUUIDByPath('/')
            dims = ()
            datatype = { 'charSet':   'H5T_CSET_UTF8', 
                     'class':  'H5T_STRING', 
                     'strPad': 'H5T_STR_NULLPAD', 
                     'length': 40}
            value = "e4bda0e5a5bde4b896e7958c".decode('hex')
            print value
            db.createAttribute("groups", root_uuid, "A1", dims, datatype, value)
            item = db.getAttributeItem("groups", root_uuid, "A1")
            self.failUnlessEqual(item['name'], "A1")
            #self.failUnlessEqual(item['value'], value)
            now = int(time.time())
            self.assertTrue(item['ctime'] > now - 5)
            self.assertTrue(item['mtime'] > now - 5)
            shape = item['shape']
            self.failUnlessEqual(shape['class'], 'H5S_SCALAR')
            item_type = item['type']
            self.failUnlessEqual(item_type['length'], 40)
            self.failUnlessEqual(item_type['class'], 'H5T_STRING') 
            self.failUnlessEqual(item_type['strPad'], 'H5T_STR_NULLPAD')
            self.failUnlessEqual(item_type['charSet'], 'H5T_CSET_UTF8') 
    """        
            
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
                     'length': 15}
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
            self.failUnlessEqual(item_type['length'], 15)
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
        # getAttributeItemByUuid
        item = None
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
            
         
            
        
            
         
             
             
if __name__ == '__main__':
    #setup test files
    
    unittest.main()
    

 



