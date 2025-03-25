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
import time
import logging
import numpy as np
from h5json import Hdf5db
from h5json import selections
from h5json.objid import isRootObjId, isValidUuid, isSchema2Id
from h5json.hdf5dtype import special_dtype, Reference


class Hdf5dbTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Hdf5dbTest, self).__init__(*args, **kwargs)
        # main

        self.log = logging.getLogger()
        if len(self.log.handlers) > 0:
            lhStdout = self.log.handlers[0]  # stdout is the only handler initially
        else:
            lhStdout = None

        self.log.setLevel(logging.DEBUG)
        # create logger

        handler = logging.FileHandler("./hdf5dbtest.log")
        # add handler to logger
        self.log.addHandler(handler)

        if lhStdout is not None:
            self.log.removeHandler(lhStdout)
        # self.log.propagate = False  # prevent log out going to stdout
        self.log.info("init!")


    def testGroup(self):
        with Hdf5db(app_logger=self.log) as db:
            root_id = db.getObjectIdByPath("/")
            self.assertTrue(isSchema2Id(root_id))
            self.assertTrue(isRootObjId(root_id))

            g1_id = db.createGroup()
            self.assertTrue(isSchema2Id(g1_id))
            self.assertFalse(isRootObjId(g1_id))
            self.assertTrue(isValidUuid(g1_id, obj_class="groups"))
            db.createHardLink(root_id, "g1", g1_id)

            g2_id = db.createGroup()
            self.assertTrue(isSchema2Id(g2_id))
            self.assertFalse(isRootObjId(g2_id))
            self.assertTrue(isValidUuid(g2_id, obj_class="groups"))
            db.createHardLink(root_id, "g2", g2_id)

            g1_1_id = db.createGroup()
            self.assertTrue(isSchema2Id(g1_1_id))
            self.assertFalse(isRootObjId(g1_1_id))
            self.assertTrue(isValidUuid(g1_1_id, obj_class="groups"))
            db.createHardLink(g1_id, "g1.1", g1_1_id)

            self.assertEqual(db.getObjectIdByPath("g1"), g1_id)
            self.assertEqual(db.getObjectIdByPath("/g1"), g1_id)
            self.assertEqual(db.getObjectIdByPath("g1/"), g1_id)

            self.assertEqual(db.getObjectIdByPath("g1/g1.1"), g1_1_id)
            self.assertEqual(db.getObjectIdByPath("/g1/g1.1"), g1_1_id)
            self.assertEqual(db.getObjectIdByPath("g1/g1.1/"), g1_1_id)

            grp1_json = db.getObjectById(g1_id)
            self.assertTrue("links" in grp1_json)
            g1_links = grp1_json["links"]
            self.assertTrue("g1.1" in g1_links)
            g1_1_link = db.getLink(g1_id, "g1.1")
            self.assertEqual(g1_1_link["class"], "H5L_TYPE_HARD")
            self.assertEqual(g1_1_link["id"], g1_1_id)
            self.assertTrue(g1_1_link["created"] > time.time() - 1.0)

            db.createSoftLink(g2_id, "slink", "somewhere")
            soft_link = db.getLink(g2_id, "slink")
            self.assertEqual(soft_link["class"], "H5L_TYPE_SOFT")
            self.assertEqual(soft_link["h5path"], "somewhere")
            self.assertTrue(soft_link["created"] > time.time() - 1.0)

            db.createExternalLink(g2_id, "extlink", "somewhere", "someplace")
            ext_link = db.getLink(g2_id, "extlink")
            self.assertEqual(ext_link["class"], "H5L_TYPE_EXTERNAL")
            self.assertEqual(ext_link["h5path"], "somewhere")
            self.assertEqual(ext_link["file"], "someplace")
            self.assertTrue(ext_link["created"] > time.time() - 1.0)

            db.createCustomLink(g2_id, "cust", {"foo": "bar"})
            cust_link = db.getLink(g2_id, "cust")
            self.assertEqual(cust_link["class"], "H5L_TYPE_USER_DEFINED")
            self.assertEqual(cust_link["foo"], "bar")
            self.assertTrue(cust_link["created"] > time.time() - 1.0)

            links = db.getLinks(g2_id)
            self.assertEqual(len(links), 3)
            for title in  "slink", "extlink", "cust":
                self.assertTrue(title in links)

            db.deleteLink(g2_id, "cust")
            links = db.getLinks(g2_id)
            self.assertEqual(len(links), 2)
            for title in  "slink", "extlink":
                self.assertTrue(title in links)

            try:
                db.getObjectIdByPath("/g1/foo")
                self.assertTrue(False)
            except KeyError:
                pass  # expected

            ret = db.getLink(g2_id, "not_a_link")
            self.assertTrue(ret is None)


    def testNullSpaceAttribute(self):
        with Hdf5db(app_logger=self.log) as db:
            root_id = db.getObjectIdByPath("/")
            db.createAttribute(root_id, "A1", None, shape="H5S_NULL", dtype=np.int32)
            item = db.getAttribute(root_id, "A1")
            self.assertTrue("shape" in item)
            shape_item = item["shape"]
            self.assertTrue("class" in shape_item)
            self.assertEqual(shape_item["class"], "H5S_NULL")
            self.assertTrue(item["created"] > time.time() - 1.0)
            value = db.getAttributeValue(root_id, "A1")
            self.assertEqual(value, None)

    def testScalarAttribute(self):
        with Hdf5db(app_logger=self.log) as db:
            root_id = db.getObjectIdByPath("/")
            dims = ()
            value = 42
            db.createAttribute(root_id, "A1", value, shape=dims, dtype=np.int32)
            item = db.getAttribute(root_id, "A1")
            shape_json = item["shape"]
            self.assertEqual(shape_json["class"], "H5S_SCALAR")
            self.assertEqual(len(shape_json.keys()), 1)  # just one key should be returned
            item_type = item["type"]
            self.assertEqual(item_type["class"], "H5T_INTEGER")
            self.assertEqual(item_type["base"], "H5T_STD_I32LE")
            self.assertEqual(len(item_type.keys()), 2)  # just two keys should be returned
            self.assertEqual(item["value"], 42)
            now = int(time.time())
            self.assertTrue(item["created"] > now - 1)
            shape = item["shape"]
            self.assertEqual(shape["class"], "H5S_SCALAR")

            self.assertEqual(item_type["class"], "H5T_INTEGER")
            self.assertEqual(item_type["base"], "H5T_STD_I32LE")
            

    def testFixedStringAttribute(self):
        with Hdf5db(app_logger=self.log) as db:
            root_id = db.getObjectIdByPath("/")
            value = "Hello, world!"
            db.createAttribute(root_id, "A1", value, dtype=np.dtype("S13"))  # dims, datatype, value)
            item = db.getAttribute(root_id, "A1")
            shape_json = item["shape"]
            self.assertEqual(shape_json["class"], "H5S_SCALAR")
            item_type = item["type"]
            self.assertEqual(item_type["class"], "H5T_STRING")
            self.assertEqual(item_type["strPad"], "H5T_STR_NULLPAD")
            self.assertEqual(item_type["length"], 13)
            self.assertEqual(item_type["charSet"], "H5T_CSET_ASCII")
            self.assertEqual(item["value"], "Hello, world!")
            now = int(time.time())
            self.assertTrue(item["created"] > now - 1)
            ret_value = db.getAttributeValue(root_id, "A1")
            self.assertEqual(ret_value, value.encode("ascii"))
       

    def testVlenAsciiAttribute(self):
        with Hdf5db(app_logger=self.log) as db:
            root_id = db.getObjectIdByPath("/")
 
            value = b"Hello, world!"
            dt = special_dtype(vlen=bytes)

            # write the attribute
            db.createAttribute(root_id, "A1", value, dtype=dt)
            # read it back
            item = db.getAttribute(root_id, "A1")
            shape_json = item["shape"]
            self.assertEqual(shape_json["class"], "H5S_SCALAR")
            item_type = item["type"]
            self.assertEqual(item_type["class"], "H5T_STRING")
            self.assertEqual(item_type["strPad"], "H5T_STR_NULLTERM")
            self.assertEqual(item_type["length"], "H5T_VARIABLE")
            self.assertEqual(item_type["charSet"], "H5T_CSET_ASCII")
            self.assertEqual(item["value"], "Hello, world!")
            now = int(time.time())
            self.assertTrue(item["created"] > now - 1)

    def testVlenUtf8Attribute(self):
        with Hdf5db(app_logger=self.log) as db:
            root_id = db.getObjectIdByPath("/")
 
            value = b"Hello, world!"
            dt = special_dtype(vlen=str)

            # write the attribute
            db.createAttribute(root_id, "A1", value, dtype=dt)
            # read it back
            item = db.getAttribute(root_id, "A1")
            shape_json = item["shape"]
            self.assertEqual(shape_json["class"], "H5S_SCALAR")
            item_type = item["type"]
            self.assertEqual(item_type["class"], "H5T_STRING")
            self.assertEqual(item_type["strPad"], "H5T_STR_NULLTERM")
            self.assertEqual(item_type["length"], "H5T_VARIABLE")
            self.assertEqual(item_type["charSet"], "H5T_CSET_UTF8")
            self.assertEqual(item["value"], "Hello, world!")
            now = int(time.time())
            self.assertTrue(item["created"] > now - 1)
 

    def testIntAttribute(self):
        with Hdf5db(app_logger=self.log) as db:
            root_id = db.getObjectIdByPath("/")
            value = [2, 3, 5, 7, 11]
            db.createAttribute(root_id, "A1", value, dtype=np.int16)
            item = db.getAttribute(root_id, "A1")
            self.assertEqual(item["value"], [2, 3, 5, 7, 11])
            now = int(time.time())
            self.assertTrue(item["created"] > now - 1)
            item_shape = item["shape"]
            self.assertEqual(item_shape["class"], "H5S_SIMPLE")
            self.assertEqual(item_shape["dims"], [5,])
            item_type = item["type"]
            self.assertEqual(item_type["class"], "H5T_INTEGER")
            self.assertEqual(item_type["base"], "H5T_STD_I16LE")

    def testCreateReferenceAttribute(self):
        with Hdf5db(app_logger=self.log) as db:
            root_id = db.getObjectIdByPath("/")

            dset_id = db.createDataset(shape=(), dtype=np.int32)                
            db.createHardLink(root_id, "DS1", dset_id)

            dt = special_dtype(ref=Reference)

            ds1_ref = "datasets/" + dset_id
            value = [ds1_ref,]
            db.createAttribute(root_id, "A1", value, dtype=dt)
            item = db.getAttribute(root_id, "A1")
            attr = db.getAttribute(root_id, "A1")
            self.assertTrue("shape" in attr)
            
            attr_type = attr["type"]
            self.assertEqual(attr_type["class"], "H5T_REFERENCE")
            self.assertEqual(attr_type["base"], "H5T_STD_REF_OBJ")
            attr_value = item["value"]
            self.assertEqual(len(attr_value), 1)
            self.assertEqual(attr_value[0], ds1_ref)

    def testCreateVlenReferenceAttribute(self):
        with Hdf5db(app_logger=self.log) as db:
            root_id = db.getObjectIdByPath("/")
            dset_id = db.createDataset(shape=(), dtype=np.int32)                
            db.createHardLink(root_id, "DS1", dset_id)
            grp_id = db.createGroup()
            db.createHardLink(root_id, "G1", grp_id)

            dt_base = special_dtype(ref=Reference)
            dt = special_dtype(vlen=dt_base)
             
            ds1_ref = "datasets/" + dset_id
            grp_ref = "groups/" + grp_id
            ref_arr = np.zeros((2,), dtype=dt_base)
            ref_arr[0] = ds1_ref
            ref_arr[1] = grp_ref
            vlen_arr = np.zeros((), dtype=dt)
            vlen_arr[()] = ref_arr
             
            db.createAttribute(root_id, "A1", vlen_arr)
            item = db.getAttribute(root_id, "A1")

            item_type = item["type"]
            self.assertEqual(item_type["class"], "H5T_VLEN")
            self.assertEqual(item_type["size"], "H5T_VARIABLE")
            base_type = item_type["base"]
            self.assertEqual(base_type["class"], "H5T_REFERENCE")
            self.assertEqual(base_type["base"], "H5T_STD_REF_OBJ")

            item_shape = item["shape"]
            self.assertEqual(item_shape["class"], "H5S_SCALAR")
            

    def testCommittedType(self):
        with Hdf5db(app_logger=self.log) as db:
            root_id = db.getObjectIdByPath("/")
            dt = np.dtype("S15")
             
            ctype_id = db.createCommittedType(dt)
            db.createHardLink(root_id, "ctype", ctype_id)
            item = db.getObjectById(ctype_id)
            now = int(time.time())
            self.assertTrue(item["created"] > now - 1)

            item_type = item["type"]

            self.assertEqual(item_type["class"], "H5T_STRING")
            self.assertEqual(item_type["strPad"], "H5T_STR_NULLPAD")
            self.assertEqual(item_type["charSet"], "H5T_CSET_ASCII")
            self.assertEqual(item_type["length"], 15)

            # create an attribute using the committed type
            db.createAttribute(root_id, "A1", "hello world!", dtype=f"datatypes/{ctype_id}")
            attr = db.getAttribute(root_id, "A1")
            self.assertEqual(attr["value"], "hello world!")

            attr_type = attr["type"]
            self.assertEqual(attr_type["class"], "H5T_STRING")
            self.assertEqual(attr_type["length"], 15)
            self.assertEqual(attr_type["charSet"], "H5T_CSET_ASCII")

    def testCommittedCompoundType(self):
        with Hdf5db(app_logger=self.log) as db:
            root_id = db.getObjectIdByPath("/")

            dt_str = special_dtype(vlen=str)
            fields = []
            fields.append(("field_1", np.dtype(">i8")))
            fields.append(("field_2", ">f8"))
            fields.append(("field_3", np.dtype("S15")))
            fields.append(("field_4", dt_str))
            dt = np.dtype(fields)

            ctype_id = db.createCommittedType(dt)
            db.createHardLink(root_id, "ctype", ctype_id)
            item = db.getObjectById(ctype_id)
            now = int(time.time())
            self.assertTrue(item["created"] > now - 1)

            item_type = item["type"]

            self.assertEqual(item_type["class"], "H5T_COMPOUND")
            fields = item_type["fields"]
            self.assertEqual(len(fields), 4)

            # create an attribute using the committed type
            attr_value = (42, 3.14, "circle", "area = R^2 * PI")
            db.createAttribute(root_id, "A1", attr_value, dtype=f"datatypes/{ctype_id}")
            attr = db.getAttribute(root_id, "A1")
            self.assertEqual(attr["value"], list(attr_value))
            attr_shape = attr["shape"]
            self.assertEqual(attr_shape["class"], "H5S_SCALAR")

            attr_type = attr["type"]
            self.assertEqual(attr_type["class"], "H5T_COMPOUND")
            
            value = db.getAttributeValue(root_id, "A1")
            self.assertTrue(isinstance(value, np.ndarray))

    def testSimpleDataset(self):
        with Hdf5db(app_logger=self.log) as db:
            nrows = 8
            ncols = 10
            shape = (nrows, ncols)
            dtype = np.int32
            root_id = db.getObjectIdByPath("/")
            dset_id = db.createDataset(shape, dtype=dtype)
            db.createHardLink(root_id, "dset", dset_id)
            db.createAttribute(dset_id, "a1", "Hello, world")
            sel_all = selections.select(shape, ...)
            arr = db.getDatasetValues(dset_id, sel_all)
            self.assertEqual(arr.dtype, dtype)
            self.assertEqual(arr.shape, shape)
            self.assertEqual(arr.min(), 0)
            self.assertEqual(arr.max(), 0)
            row = np.zeros((ncols,), dtype=dtype)
            for i in range(nrows):
                row[:] = list(range(i*10, (i + 1)*10))
                row_sel = selections.select(shape, (slice(i, i + 1), slice(0, ncols)))
                db.setDatasetValues(dset_id, row_sel, row)
            arr = db.getDatasetValues(dset_id, sel_all)
            for i in range(nrows):
                row = np.array(list(range(i*10, (i + 1)*10)), dtype=dtype)
                np.testing.assert_array_equal(arr[i, :],  row)
            

    def testScalarDataset(self):
        dtype = np.int32
        with Hdf5db(app_logger=self.log) as db:
            root_id = db.getObjectIdByPath("/")
            dset_id = db.createDataset((), dtype=dtype)
            db.createHardLink(root_id, "dset", dset_id)
            db.createAttribute(dset_id, "a1", "Hello, world")
            sel_all = selections.select((), ...)
            arr = db.getDatasetValues(dset_id, sel_all)
            self.assertEqual(arr.dtype, dtype)
            self.assertEqual(arr.shape, ())
            self.assertEqual(arr[()], 0)
            db.setDatasetValues(dset_id, sel_all, np.array(42, dtype=dtype))
            arr = db.getDatasetValues(dset_id, sel_all)
            self.assertEqual(arr.dtype, dtype)
            self.assertEqual(arr.shape, ())
            self.assertEqual(arr.min(), 42)
            self.assertEqual(arr.max(), 42)

    def testResizableDataset(self):
        with Hdf5db(app_logger=self.log) as db:
            nrows = 8
            ncols = 10
            shape = (nrows, ncols)
            dtype = np.int32
            maxdims = (None, ncols*2)
            root_id = db.getObjectIdByPath("/")
            dset_id = db.createDataset(shape, maxdims=maxdims, dtype=dtype)
            db.createHardLink(root_id, "dset", dset_id)
            db.createAttribute(dset_id, "a1", "Hello, world")
            
            # resize limited dimension
            db.resizeDataset(dset_id, (nrows, ncols*2))

            # try to go beyond max extent
            try:
                db.resizeDataset(dset_id, (nrows, ncols*3))
                self.assertTrue(False)
            except ValueError:
                pass  # expected

            # resize unlimited dimension
            db.resizeDataset(dset_id, (nrows*10, ncols))


            

            





   

if __name__ == "__main__":
    # setup test files

    unittest.main()
