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

    def testOpen(self):
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()
        self.assertTrue(isSchema2Id(root_id))
        self.assertTrue(isRootObjId(root_id))
        self.assertFalse(db.closed)
        self.assertEqual(db.getObjectIdByPath("/"), root_id)
        db.close()
        self.assertTrue(db.closed)
        obj_id = db.open()
        self.assertEqual(obj_id, root_id)
        root_json = db.getObjectById(root_id)
        self.assertFalse("id" in root_json)
        db.close()

    def testWith(self):
        with Hdf5db(app_logger=self.log) as db:
            root_id = db.open()
            self.assertTrue(isRootObjId(root_id))

    def testGroup(self):
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()

        paths = db.getPathsForObjectId(root_id)
        self.assertEqual(paths, ["/"])

        g1_id = db.createGroup()
        self.assertTrue(isSchema2Id(g1_id))
        self.assertFalse(isRootObjId(g1_id))
        self.assertTrue(isValidUuid(g1_id, obj_class="groups"))
        paths = db.getPathsForObjectId(g1_id)
        self.assertEqual(paths, [])
        db.createHardLink(root_id, "g1", g1_id)
        paths = db.getPathsForObjectId(g1_id)
        self.assertEqual(paths, ["/g1"])

        g2_id = db.createGroup()
        self.assertTrue(isSchema2Id(g2_id))
        self.assertFalse(isRootObjId(g2_id))
        self.assertTrue(isValidUuid(g2_id, obj_class="groups"))
        db.createHardLink(root_id, "g2", g2_id)

        root_obj = db.getObjectById(root_id)
        self.assertTrue("links" in root_obj)
        root_links = root_obj["links"]
        self.assertTrue("g1" in root_links)
        self.assertTrue("g2" in root_links)
        self.assertEqual(len(root_links), 2)

        g1_1_id = db.createGroup()
        self.assertTrue(isSchema2Id(g1_1_id))
        self.assertFalse(isRootObjId(g1_1_id))
        self.assertTrue(isValidUuid(g1_1_id, obj_class="groups"))
        db.createHardLink(g1_id, "g1.1", g1_1_id)
        paths = db.getPathsForObjectId(g1_1_id)
        self.assertEqual(paths, ["/g1/g1.1"])

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
        for title in "slink", "extlink", "cust":
            self.assertTrue(title in links)

        db.deleteLink(g2_id, "cust")
        links = db.getLinks(g2_id)
        self.assertEqual(len(links), 2)
        for title in "slink", "extlink":
            self.assertTrue(title in links)

        try:
            db.getObjectIdByPath("/g1/foo")
            self.assertTrue(False)
        except KeyError:
            pass  # expected

        ret = db.getLink(g2_id, "not_a_link")
        self.assertTrue(ret is None)

        db.createAttribute(g1_id, "a1", "hello")
        db.createAttribute(g1_id, "a2", "bye-bye")
        self.assertEqual(len(db.getAttributes(g1_id)), 2)
        a1_attr = db.getAttribute(g1_id, "a1")
        self.assertEqual(a1_attr["value"], "hello")
        self.assertTrue("shape" in a1_attr)
        attr_shape = a1_attr["shape"]
        self.assertEqual(attr_shape["class"], "H5S_SCALAR")

        db.deleteAttribute(g1_id, "a1")
        self.assertEqual(len(db.getAttributes(g1_id)), 1)
        self.assertEqual(db.getAttribute(g1_id, "a1"), None)
        db.close()

    def testCircularLinks(self):
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()
        g1_id = db.createGroup()
        db.createHardLink(root_id, "g1", g1_id)
        g2_id = db.createGroup()
        db.createHardLink(g1_id, "g2", g2_id)
        # create circular link
        db.createHardLink(g2_id, "g1", g1_id)

        g1_json = db.getObjectById(g1_id)
        self.assertTrue("links" in g1_json)
        g1_links = g1_json["links"]
        self.assertTrue("g2" in g1_links)
        self.assertEqual(len(g1_links), 1)

        g2_json = db.getObjectById(g2_id)
        self.assertTrue("links" in g2_json)
        g2_links = g2_json["links"]
        self.assertTrue("g1" in g2_links)
        self.assertEqual(len(g2_links), 1)

        paths = db.getPathsForObjectId(g2_id)
        # only the canonical path is returned
        self.assertEqual(paths, ["/g1/g2"])
        grp_id = db.getObjectIdByPath("/g1/g2")
        self.assertEqual(grp_id, g2_id)
        # you can still get objects via circular paths...
        grp_id = db.getObjectIdByPath("/g1/g2/g1")
        self.assertEqual(grp_id, g1_id)
        grp_id = db.getObjectIdByPath("/g1/g2/g1/g2")
        self.assertEqual(grp_id, g2_id)

        db.close()

    def testNullSpaceAttribute(self):
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()
        db.createAttribute(root_id, "A1", None, shape="H5S_NULL", dtype=np.int32)
        item = db.getAttribute(root_id, "A1")
        self.assertTrue("shape" in item)
        shape_item = item["shape"]
        self.assertTrue("class" in shape_item)
        self.assertEqual(shape_item["class"], "H5S_NULL")
        self.assertFalse("value" in item)
        self.assertTrue(item["created"] > time.time() - 1.0)
        value = db.getAttributeValue(root_id, "A1")
        self.assertEqual(value, None)
        db.close()

    def testScalarAttribute(self):
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()
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

        value = db.getAttributeValue(root_id, "A1")
        self.assertTrue(isinstance(value, np.ndarray))
        self.assertEqual(value.shape, ())
        self.assertEqual(value.dtype, np.int32)
        self.assertEqual(value[()], 42)
        db.close()

    def testFixedStringAttribute(self):
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()
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
        self.assertTrue(isinstance(ret_value, np.ndarray))
        self.assertEqual(ret_value.shape, ())
        self.assertEqual(ret_value.dtype, np.dtype("S13"))
        self.assertEqual(ret_value[()], value.encode("ascii"))
        db.close()

    def testVlenAsciiAttribute(self):
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()

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

        ret_value = db.getAttributeValue(root_id, "A1")
        self.assertTrue(isinstance(ret_value, np.ndarray))
        self.assertEqual(ret_value.shape, ())
        self.assertEqual(ret_value.dtype, dt)
        self.assertEqual(ret_value[()], value)

        now = int(time.time())
        self.assertTrue(item["created"] > now - 1)
        db.close()

    def testVlenUtf8Attribute(self):
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()

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

        ret_value = db.getAttributeValue(root_id, "A1")
        self.assertTrue(isinstance(ret_value, np.ndarray))
        self.assertEqual(ret_value.shape, ())
        self.assertEqual(ret_value.dtype, dt)
        self.assertEqual(ret_value[()].encode(), value)

        now = int(time.time())
        self.assertTrue(item["created"] > now - 1)
        db.close()

    def testIntAttribute(self):
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()
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

        ret_value = db.getAttributeValue(root_id, "A1")
        self.assertTrue(isinstance(ret_value, np.ndarray))
        self.assertEqual(ret_value.shape, (len(value),))
        self.assertEqual(ret_value.dtype, np.int16)
        for i in range(len(value)):
            self.assertEqual(ret_value[i], value[i])

        now = int(time.time())
        self.assertTrue(item["created"] > now - 1)

        db.close()

    def testCompoundAttribute(self):
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()
        dt_compound = np.dtype([("field1", "S8"), ("field2", np.int32)])
        value = [("hello", 42), ('', 0), ("world", 99),]
        db.createAttribute(root_id, "A1", value, dtype=dt_compound)
        item = db.getAttribute(root_id, "A1")
        item_value = item['value']
        self.assertEqual(len(item_value), 3)
        for i in range(3):
            e = item_value[i]
            # self.assertTrue(isinstance(e, tuple))  # TBD
            self.assertEqual(tuple(e), value[i])

        item_shape = item["shape"]
        self.assertEqual(item_shape["class"], "H5S_SIMPLE")
        self.assertEqual(item_shape["dims"], [3,])
        item_type = item["type"]
        self.assertEqual(item_type["class"], "H5T_COMPOUND")

        ret_value = db.getAttributeValue(root_id, "A1")
        self.assertTrue(isinstance(ret_value, np.ndarray))
        self.assertEqual(ret_value.shape, (3,))
        self.assertEqual(ret_value.dtype, dt_compound)
        for i in range(3):
            e = ret_value[i]
            self.assertEqual((e[0].decode(), e[1]), value[i])

        now = int(time.time())
        self.assertTrue(item["created"] > now - 1)

        db.close()

    def testCreateReferenceAttribute(self):
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()

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

        db.close()

    def testCreateVlenReferenceAttribute(self):
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()
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

        db.close()

    def testAttributeCreateOrder(self):
        titles = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
        cpl = {"CreateOrder": True}
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()
        g1_id = db.createGroup()
        db.createHardLink(root_id, "g1", g1_id)
        for title in titles:
            db.createAttribute(g1_id, title, title)
        g2_id = db.createGroup(cpl=cpl)
        db.createHardLink(root_id, "g2", g2_id)
        for title in titles:
            db.createAttribute(g2_id, title, title)
        self.assertEqual(sorted(db.getAttributes(g1_id)), sorted(titles))
        self.assertEqual(tuple(db.getAttributes(g2_id)), titles)
        db.close()

    def testCommittedType(self):
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()
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

        db.close()

    def testCommittedCompoundType(self):
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()

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

        db.close()

    def test1DDataset(self):
        nelements = 10
        shape = (nelements,)
        dtype = np.int32

        db = Hdf5db(app_logger=self.log)
        root_id = db.open()
        dset_id = db.createDataset(shape, dtype=dtype)
        db.createHardLink(root_id, "dset", dset_id)
        db.createAttribute(dset_id, "a1", "Hello, world")
        sel_all = selections.select(shape, ...)
        arr = db.getDatasetValues(dset_id, sel_all)

        self.assertEqual(arr.dtype, dtype)
        self.assertEqual(arr.shape, shape)
        self.assertEqual(arr.min(), 0)
        self.assertEqual(arr.max(), 0)

        # set values element by element
        for i in range(nelements):
            sel = selections.select(shape, slice(i, i + 1))
            db.setDatasetValues(dset_id, sel, np.array([i], dtype=dtype))

        # read entire dataset
        arr = db.getDatasetValues(dset_id, sel_all)
        for i in range(nelements):
            val = np.array([i], dtype=dtype)
            np.testing.assert_array_equal(arr[i], val)

        # read element by element
        for i in range(nelements):
            sel = selections.select(shape, slice(i, i + 1))
            val = db.getDatasetValues(dset_id, sel)
            self.assertTrue(isinstance(val, np.ndarray))
            self.assertEqual(val.shape, (1,))
            self.assertEqual(val[0], i)

        # do a point selection
        sel = selections.select(shape, [2, 3, 5, 7])
        
        val = db.getDatasetValues(dset_id, sel)
        self.assertTrue(isinstance(val, np.ndarray))
        self.assertEqual(val.shape, (4,))

        self.assertEqual(val[0], 2)
        self.assertEqual(val[1], 3)
        self.assertEqual(val[2], 5)
        self.assertEqual(val[3], 7)

        # point selection write
        arr = np.zeros((4,), dtype=dtype)
        db.setDatasetValues(dset_id, sel, arr)
        arr = db.getDatasetValues(dset_id, sel_all)
        for i in range(nelements):
            if i in (2, 3, 5, 7):
                self.assertEqual(arr[i], 0)  # these were set to 0 by point selection write
            else:
                self.assertEqual(arr[i], i)

        # try with broadcasting
        arr_one_value = np.zeros((1), dtype=dtype)
        arr_one_value[0] = 42
        db.setDatasetValues(dset_id, sel_all, arr_one_value)
        # check that entire dataset is updated to the single value
        arr = db.getDatasetValues(dset_id, sel_all)
        self.assertTrue((arr == 42).all())

        db.close()

    def test2DDataset(self):
        nrows = 8
        ncols = 10
        shape = (nrows, ncols)
        dtype = np.int32

        db = Hdf5db(app_logger=self.log)
        root_id = db.open()
        dset_id = db.createDataset(shape, dtype=dtype)
        db.createHardLink(root_id, "dset", dset_id)
        db.createAttribute(dset_id, "a1", "Hello, world")
        sel_all = selections.select(shape, ...)
        arr = db.getDatasetValues(dset_id, sel_all)

        self.assertEqual(arr.dtype, dtype)
        self.assertEqual(arr.shape, shape)
        self.assertEqual(arr.min(), 0)
        self.assertEqual(arr.max(), 0)
        row = np.zeros((1, ncols,), dtype=dtype)

        # set values row by row
        for i in range(nrows):
            row[0, :] = list(range(i * 10, (i + 1) * 10))
            row_sel = selections.select(shape, (slice(i, i + 1), slice(0, ncols)))
            db.setDatasetValues(dset_id, row_sel, row)

        # read entire dataset
        arr = db.getDatasetValues(dset_id, sel_all)
        for i in range(nrows):
            row = np.array(list(range(i * 10, (i + 1) * 10)), dtype=dtype)
            np.testing.assert_array_equal(arr[i, :], row)

        # read row by row
        for i in range(nrows):
            sel = selections.select(shape, (slice(i, i + 1), slice(0, ncols)))
            row = db.getDatasetValues(dset_id, sel)
            self.assertTrue(isinstance(row, np.ndarray))
            self.assertEqual(row.shape, (1, ncols))
            for j in range(ncols):
                self.assertEqual(row[0, j], i * 10 + j)

        # read col by col
        for j in range(ncols):
            sel = selections.select(shape, (slice(0, ncols), slice(j, j + 1)))
            col = db.getDatasetValues(dset_id, sel)
            self.assertTrue(isinstance(col, np.ndarray))
            self.assertEqual(col.shape, (nrows, 1))
            for i in range(nrows):
                self.assertEqual(col[i, 0], i * 10 + j)

        # read element by element
        for i in range(nrows):
            for j in range(ncols):
                sel = selections.select(shape, (slice(i, i + 1), slice(j, j + 1)))
                val = db.getDatasetValues(dset_id, sel)
                self.assertTrue(isinstance(val, np.ndarray))
                self.assertEqual(val.shape, (1, 1))
                self.assertEqual(val[0, 0], i * 10 + j)

        # do a point selection
        sel = selections.select(shape, [(0, 0), (1, 1), (2, 2), (3, 3)])
        val = db.getDatasetValues(dset_id, sel)
        self.assertTrue(isinstance(val, np.ndarray))
        self.assertEqual(val.shape, (4,))
        for i in range(4):
            self.assertEqual(val[i], i * 10 + i)

        # point selection write
        arr = np.zeros((4,), dtype=dtype)
        db.setDatasetValues(dset_id, sel, arr)

        # test select all write
        sel_all = selections.select(shape, ...)
        arr = np.zeros(shape, dtype=dtype)
        arr[...] = 42
        db.setDatasetValues(dset_id, sel_all, arr)
        arr = db.getDatasetValues(dset_id, sel_all)
        for i in range(nrows):
            for j in range(ncols):
                self.assertEqual(arr[i, j], 42)

        # try with broadcasting
        arr_one_value = np.zeros((1, 1), dtype=dtype)
        arr_one_value[0, 0] = 7
        db.setDatasetValues(dset_id, sel_all, arr_one_value)
        # check that entire dataset is updated to the single value
        arr = db.getDatasetValues(dset_id, sel_all)
        self.assertTrue((arr == 7).all())

        db.close()

    def testStringDataset(self):
        nrows = 6
        ncols = 3
        shape = (nrows, ncols)
        dtype = np.dtype("S1")
        data = [[b'a', b'b', b'c'],
                [b'd', b'e', b'f'],
                [b'g', b'h', b'i'],
                [b'j', b'k', b'l'],
                [b'm', b'n', b'o'],
                [b'x', b'y', b'z']]
        init_arr = np.array(data, dtype=dtype)

        db = Hdf5db(app_logger=self.log)
        root_id = db.open()
        dset_id = db.createDataset(shape, dtype=dtype)
        db.createHardLink(root_id, "dset", dset_id)
        sel_all = selections.select(shape, ...)
        arr = db.getDatasetValues(dset_id, sel_all)
        self.assertEqual(arr.dtype, dtype)
        self.assertEqual(arr.shape, shape)

        db.setDatasetValues(dset_id, sel_all, init_arr)

        arr = db.getDatasetValues(dset_id, sel_all)
        self.assertTrue(np.array_equal(arr, init_arr))
        sel_one = selections.select(shape, (slice(5, 6), slice(2, 3)))
        arr = db.getDatasetValues(dset_id, sel_one)
        self.assertEqual(arr.shape, (1, 1))
        self.assertEqual(arr[0, 0], b'z')

        db.close()

    def testBoolDataset(self):
        shape = (10,)
        dtype = np.dtype(bool)

        db = Hdf5db(app_logger=self.log)
        root_id = db.open()
        dset_id = db.createDataset(shape, dtype=dtype)
        db.createHardLink(root_id, "dset", dset_id)
        sel_first = selections.select(shape, slice(0, 1))
        arr = db.getDatasetValues(dset_id, sel_first)
        self.assertEqual(arr.dtype, dtype)
        self.assertEqual(arr.shape, (1,))
        self.assertEqual(arr[0], False)

        # update one element
        sel_second = selections.select(shape, slice(1, 2))
        db.setDatasetValues(dset_id, sel_second, np.array([True,], dtype=dtype))

        # read back three elements
        sel_three = selections.select(shape, slice(0, 3))
        arr = db.getDatasetValues(dset_id, sel_three)
        self.assertEqual(arr.dtype, dtype)
        self.assertEqual(arr.shape, (3,))
        self.assertEqual(list(arr[...]), [False, True, False])

        # read back three elements
        sel_three = selections.select(shape, slice(1, 4))
        arr = db.getDatasetValues(dset_id, sel_three)
        self.assertEqual(arr.dtype, dtype)
        self.assertEqual(arr.shape, (3,))
        self.assertEqual(list(arr[...]), [True, False, False])

        db.close()

    def testVlenStringDataset(self):
        nrows = 4
        shape = (nrows,)
        dtype = special_dtype(vlen=str)
        data = ["Hello", "HDF5", "REST", "API"]
        init_arr = np.array(data, dtype=dtype)

        db = Hdf5db(app_logger=self.log)
        root_id = db.open()
        dset_id = db.createDataset(shape, dtype=dtype)
        db.createHardLink(root_id, "dset", dset_id)
        sel_all = selections.select(shape, ...)
        arr = db.getDatasetValues(dset_id, sel_all)
        self.assertEqual(arr.dtype, dtype)
        self.assertEqual(arr.shape, shape)

        db.setDatasetValues(dset_id, sel_all, init_arr)

        arr = db.getDatasetValues(dset_id, sel_all)
        self.assertTrue(np.array_equal(arr, init_arr))
        sel_one = selections.select(shape, slice(2, 3))
        arr = db.getDatasetValues(dset_id, sel_one)
        self.assertEqual(arr.shape, (1,))
        self.assertEqual(arr[0], 'REST')

        db.close()

    def testVlenIntDataset(self):
        nrows = 4
        shape = (nrows,)
        dtype = special_dtype(vlen=np.int32)

        init_arr = np.empty((nrows,), dtype=dtype)
        for i in range(nrows):
            init_arr[i] = np.array(list(range(i, 2 * i + 1)), dtype=np.int32)

        db = Hdf5db(app_logger=self.log)
        root_id = db.open()
        dset_id = db.createDataset(shape, dtype=dtype)
        db.createHardLink(root_id, "dset", dset_id)
        sel_all = selections.select(shape, ...)
        arr = db.getDatasetValues(dset_id, sel_all)
        self.assertEqual(arr.dtype, dtype)
        self.assertEqual(arr.shape, shape)

        db.setDatasetValues(dset_id, sel_all, init_arr)

        arr = db.getDatasetValues(dset_id, sel_all)
        self.assertTrue(isinstance(arr, np.ndarray))
        self.assertEqual(arr.dtype.kind, 'O')
        self.assertTrue("vlen" in arr.dtype.metadata)
        self.assertEqual(arr.dtype.metadata["vlen"], np.dtype(np.int32))
        for i in range(nrows):
            e = arr[i]
            self.assertTrue(isinstance(e, np.ndarray))
            self.assertEqual(e.dtype, np.int32)
            self.assertTrue(np.array_equal(e, init_arr[i]))

        sel_one = selections.select(shape, slice(2, 3))
        arr = db.getDatasetValues(dset_id, sel_one)
        self.assertEqual(arr.shape, (1,))
        self.assertTrue(np.array_equal(arr[0], init_arr[2]))

        db.close()

    def testScalarDataset(self):
        dtype = np.int32

        db = Hdf5db(app_logger=self.log)
        root_id = db.open()
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

        db.close()

    def testResizableDataset(self):
        nrows = 8
        ncols = 10
        shape = (nrows, ncols)
        dtype = np.int32
        maxdims = (None, ncols * 2)
        layout = {"class": "H5D_CHUNKED", "dims": shape}
        cpl = {"layout": layout}

        db = Hdf5db(app_logger=self.log)

        root_id = db.open()
        dset_id = db.createDataset(shape, maxdims=maxdims, dtype=dtype, cpl=cpl)
        db.createHardLink(root_id, "dset", dset_id)
        db.createAttribute(dset_id, "a1", "Hello, world")

        # resize limited dimension
        db.resizeDataset(dset_id, (nrows, ncols * 2))

        # try to go beyond max extent
        try:
            db.resizeDataset(dset_id, (nrows, ncols * 3))
            self.assertTrue(False)
        except ValueError:
            pass  # expected

        # resize unlimited dimension
        db.resizeDataset(dset_id, (nrows * 10, ncols))

        db.close()

    def testFillValueDataset(self):
        dtype = np.uint32
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()
        cpl = {"fillValue": 0xdeadbeef}
        dset_id = db.createDataset((), dtype=dtype, cpl=cpl)
        db.createHardLink(root_id, "dset", dset_id)
        dset_json = db.getObjectById(dset_id)
        self.assertTrue("creationProperties" in dset_json)
        cpl = dset_json["creationProperties"]
        self.assertTrue("fillValue" in cpl)
        self.assertEqual(cpl["fillValue"], 0xdeadbeef)
        sel_all = selections.select((), ...)
        arr = db.getDatasetValues(dset_id, sel_all)
        self.assertEqual(arr.dtype, dtype)
        self.assertEqual(arr.shape, ())
        self.assertEqual(arr[()], 0xdeadbeef)


if __name__ == "__main__":
    # setup test files

    unittest.main()
