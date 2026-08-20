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
import math
import numpy as np
from h5json import Hdf5db
from h5json import selections
from h5json.hdf5db import ChunkIterator
from h5json.objid import isRootObjId, isValidUuid, isSchema2Id
from h5json.hdf5dtype import special_dtype, Reference, RegionReference
from h5json.storage_plugin import NullPlugin
from h5json.jsonstore.h5json_plugin import H5JsonPlugin


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

    def testArrayTypeAttribute(self):
        # A top-level array-typed attribute (dtype.subdtype is not None) -
        # one scalar attribute whose value is itself a fixed-size array.
        #
        # Regression test: createAttribute() used to convert the value via
        # np.asarray(value, dtype=<array dtype>) *before* unwrapping the
        # array type. numpy silently broadcasts in that case rather than
        # reinterpreting a matching-shape array as a single array-typed
        # element - e.g. np.asarray([0, 1, 2], dtype='(3,)i4') produces a
        # (3, 3) broadcast, not the intended 3-element scalar. That also
        # lost the array type itself: since the (already-broadcast) value's
        # apparent shape no longer matched the expected shape, the "advertised"
        # shape/dtype reduction that follows was skipped, so the stored type
        # ended up as a plain H5T_INTEGER instead of H5T_ARRAY.
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()
        dt = np.dtype("(3,)i4")
        value = np.arange(3, dtype="i4")
        db.createAttribute(root_id, "A1", value, shape=value.shape, dtype=dt)
        item = db.getAttribute(root_id, "A1")

        shape_json = item["shape"]
        # the array's own shape (3,) is entirely absorbed into the type,
        # so the attribute itself is a scalar
        self.assertEqual(shape_json["class"], "H5S_SCALAR")

        item_type = item["type"]
        self.assertEqual(item_type["class"], "H5T_ARRAY")
        self.assertEqual(item_type["dims"], (3,))
        self.assertEqual(item_type["base"]["class"], "H5T_INTEGER")
        self.assertEqual(item_type["base"]["base"], "H5T_STD_I32LE")

        # the stored value must be the plain 3-element array, not a
        # broadcast 3x3 result
        self.assertEqual(item["value"], [0, 1, 2])

        ret_value = db.getAttributeValue(root_id, "A1")
        self.assertTrue(isinstance(ret_value, np.ndarray))
        self.assertEqual(ret_value.shape, (3,))
        self.assertEqual(ret_value.dtype, np.dtype("i4"))
        self.assertTrue((ret_value == value).all())
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
        attr = db.getAttribute(root_id, "A1")
        self.assertTrue("shape" in attr)

        attr_type = attr["type"]
        self.assertEqual(attr_type["class"], "H5T_REFERENCE")
        self.assertEqual(attr_type["base"], "H5T_STD_REF_OBJ")
        attr_value = attr["value"]
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

        # read with a fancy selection
        sel = selections.select(shape, (slice(0, 4), [0, 2, 4, 6, 8]))
        val = db.getDatasetValues(dset_id, sel)
        self.assertTrue(isinstance(val, np.ndarray))
        self.assertEqual(val.shape, (4, 5))
        for i in range(4):
            for j in range(5):
                self.assertEqual(val[i, j], i * 10 + j * 2)

        # read with a point selection with two coordinates
        sel = selections.select(shape, ([1, 3, 5, 7], [0, 2, 4, 6]))
        val = db.getDatasetValues(dset_id, sel)
        self.assertTrue(isinstance(val, np.ndarray))
        self.assertEqual(val.shape, (4,))

        for i in range(4):
            self.assertEqual(val[i], ((i * 2) + 1) * 10 + i * 2)

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
        arr = db.getDatasetValues(dset_id, sel_all)
        for i in range(nrows):
            for j in range(ncols):
                x = arr[i, j]
                if i == j and i < 4:
                    # these are the elements we zeroed out with the point write
                    self.assertEqual(x, 0)
                else:
                    self.assertEqual(x, i * 10 + j)

        # point selection write with broadcasting
        arr = np.array(42, dtype=dtype)
        db.setDatasetValues(dset_id, sel, arr)
        arr = db.getDatasetValues(dset_id, sel_all)
        for i in range(nrows):
            for j in range(ncols):
                x = arr[i, j]
                if i == j and i < 4:
                    # these are the elements were set to 42 with the point write
                    self.assertEqual(x, 42)
                else:
                    self.assertEqual(x, i * 10 + j)

        # test select all write
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

    def test3DDataset(self):

        shape = (5, 1000, 1000)
        dtype = np.int32

        db = Hdf5db(app_logger=self.log)
        db.open()

        dset_id = db.createDataset(shape, dtype=dtype)
        # write some values to the dataset
        sel = selections.select(shape, (slice(0, 5), 1, 10))
        data = np.array([95, 96, 97, 98, 99], dtype=dtype)
        db.setDatasetValues(dset_id, sel, data)

        sel = selections.select(shape, (slice(0, 5), 10, 100))
        data = np.array([195, 196, 197, 198, 199], dtype=dtype)
        db.setDatasetValues(dset_id, sel, data)

        sel = selections.select(shape, (slice(0, 5), 100, 500))
        data = np.array([295, 296, 297, 298, 299], dtype=dtype)
        db.setDatasetValues(dset_id, sel, data)

        # single coordinate, increasing
        sel = selections.select(shape, (slice(0, 5), 10, [10, 100, 500]))
        arr = db.getDatasetValues(dset_id, sel)
        self.assertEqual(arr.shape, (5, 3))
        self.assertTrue((arr[:, 0] == [0, 0, 0, 0, 0]).all())
        self.assertTrue((arr[:, 1] == [195, 196, 197, 198, 199]).all())
        self.assertTrue((arr[:, 2] == [0, 0, 0, 0, 0]).all())

        # non-increasing indexes
        sel = selections.select(shape, (slice(0, 5), 10, [100, 10, 500]))
        arr = db.getDatasetValues(dset_id, sel)
        self.assertEqual(arr.shape, (5, 3))
        self.assertTrue((arr[:, 0] == [195, 196, 197, 198, 199]).all())
        self.assertTrue((arr[:, 1] == [0, 0, 0, 0, 0]).all())
        self.assertTrue((arr[:, 2] == [0, 0, 0, 0, 0]).all())

        # test multiple coordinates
        sel = selections.select(shape, (0, [1, 10, 100], [10, 100, 500]))
        arr = db.getDatasetValues(dset_id, sel)
        self.assertEqual(arr.shape, (3,))
        self.assertTrue((arr[:] == [95, 195, 295]).all())

        # test slice plus two coordinates
        sel = selections.select(shape, (slice(0, 5), [1, 10, 100], [10, 100, 500]))
        arr = db.getDatasetValues(dset_id, sel)
        self.assertEqual(arr.shape, (5, 3))
        self.assertTrue((arr[:, 0] == [95, 96, 97, 98, 99]).all())
        self.assertTrue((arr[:, 1] == [195, 196, 197, 198, 199]).all())
        self.assertTrue((arr[:, 2] == [295, 296, 297, 298, 299]).all())

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

    def testCompoundDataset(self):
        count = 10

        db = Hdf5db(app_logger=self.log)
        db.open()
        dtype = np.dtype([('real', np.float32), ('img', np.float32)])
        dset_id = db.createDataset((count,), dtype=dtype)

        sel_one = selections.select((count,), slice(0, 1))
        val = db.getDatasetValues(dset_id, sel_one)

        for i in range(count):
            theta = (4.0 * math.pi) * (float(i) / float(count))
            val['real'] = math.cos(theta)
            val['img'] = math.sin(theta)
            sel_one = selections.select((count,), slice(i, i + 1))
            db.setDatasetValues(dset_id, sel_one, val)

        sel_one = selections.select((count,), slice(0, 1))
        val = db.getDatasetValues(dset_id, sel_one)
        self.assertEqual(val['real'], 1.0)

        # create a selection to fetch just the real components
        sel_real = selections.select((count,), ..., fields=["real",])
        val = db.getDatasetValues(dset_id, sel_real)

        self.assertTrue(isinstance(val, np.ndarray))
        self.assertEqual(len(val.dtype), 0)
        self.assertEqual(val.dtype, np.float32)

        # zero out the imaginary values
        sel_img = selections.select((count,), ..., fields=["img", ])
        db.setDatasetValues(dset_id, sel_img, np.array(0.0, dtype=np.float32))

        # read the entire dataset
        sel_all = selections.select((count,), ...)
        val = db.getDatasetValues(dset_id, sel_all)
        self.assertTrue(isinstance(val, np.ndarray))
        self.assertEqual(len(val.dtype), 2)
        for i in range(count):
            theta = (4.0 * math.pi) * (float(i) / float(count))
            e = val[i]
            self.assertEqual(e[0], math.cos(theta))
            self.assertEqual(e[1], 0.0)

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

    def _make_tabular_arr(self):
        """Return a 1-D compound ndarray with 12 rows of stock-trade data."""
        value = [
            ("EBAY", "20170102", 3023, 3088),  # 0
            ("AAPL", "20170102", 3054, 2933),  # 1
            ("AMZN", "20170102", 2973, 3011),  # 2
            ("EBAY", "20170103", 3042, 3128),  # 3
            ("AAPL", "20170103", 3182, 3034),  # 4
            ("AMZN", "20170103", 3021, 2788),  # 5
            ("EBAY", "20170104", 2798, 2876),  # 6
            ("AAPL", "20170104", 2834, 2867),  # 7
            ("AMZN", "20170104", 2891, 2978),  # 8
            ("EBAY", "20170105", 2973, 2962),  # 9
            ("AAPL", "20170105", 2934, 3010),  # 10
            ("AMZN", "20170105", 3018, 3086),  # 11
        ]
        dtype = np.dtype([("symbol", "S4"), ("date", "S8"), ("open", "i4"), ("close", "i4")])
        arr = np.zeros((len(value),), dtype=dtype)
        for i, row in enumerate(value):
            for j in range(4):
                arr[i][j] = row[j]
        return arr

    def testQuerySimpleType(self):
        nrows = 10
        ncols = 10
        shape = (nrows, ncols)
        dtype = np.int32
        db = Hdf5db(app_logger=self.log)
        db.open()

        dset_id = db.createDataset(shape, dtype=dtype)
        arr = np.zeros(shape, dtype=dtype)
        for i in range(nrows):
            for j in range(ncols):
                arr[i, j] = i * j
        sel_all = selections.select(shape, ...)
        db.setDatasetValues(dset_id, sel_all, arr)
        # query syntax follows https://hdfgroup.github.io/h5col/queries/syntax.html
        query = "field('_') > 10"
        indices = db.queryDataset(dset_id, query)
        self.assertEqual(indices.shape, (56, 2))

        indices = db.queryDataset(dset_id, query, update_value=0)
        self.assertEqual(indices.shape, (56, 2))

        indices = db.queryDataset(dset_id, query)
        self.assertEqual(indices.shape, (0, 2))

        # query update with limit
        query = "field('_') == 0"
        indices = db.queryDataset(dset_id, query, limit=5, update_value=-99)
        self.assertEqual(indices.shape, (5, 2))

        db.close()

    def testQueryDataset1D(self):
        data_arr = self._make_tabular_arr()
        shape = data_arr.shape

        db = Hdf5db(app_logger=self.log)
        db.open()
        dset_id = db.createDataset(shape, dtype=data_arr.dtype)
        sel_all = selections.select(shape, ...)
        db.setDatasetValues(dset_id, sel_all, data_arr)

        # simple equality query
        query = "field('symbol') == b'AAPL'"
        indices = db.queryDataset(dset_id, query)
        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(indices.dtype, np.dtype("int64"))
        self.assertEqual(indices.shape, (4, 1))
        expected_indexes = {1, 4, 7, 10}
        for idx in indices:
            self.assertIn(int(idx[0]), expected_indexes)

        # isin query
        query = "field('symbol').isin(b'AAPL', b'EBAY')"
        indices = db.queryDataset(dset_id, query)
        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(len(indices), 8)
        expected_indexes = {0, 1, 3, 4, 6, 7, 9, 10}
        for idx in indices:
            self.assertIn(int(idx[0]), expected_indexes)

        # AND ('&') query across two fields
        query = "(field('symbol').isin(b'AAPL')) & (field('date') > 20170102)"
        indices = db.queryDataset(dset_id, query)
        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(len(indices), 3)
        expected_indexes = {4, 7, 10}
        for idx in indices:
            self.assertIn(int(idx[0]), expected_indexes)

        # query with no results
        query = "field('symbol') == b'XYZ'"
        indices = db.queryDataset(dset_id, query)
        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(indices.dtype, np.dtype("int64"))
        self.assertEqual(indices.shape, (0, 1))
        self.assertEqual(len(indices), 0)

        # query with selection (only rows 2-11)
        sel = selections.select(shape, slice(2, 12))
        query = "field('symbol') == b'AAPL'"
        indices = db.queryDataset(dset_id, query, sel=sel)
        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(indices.shape, (3, 1))
        expected_in_order = (4, 7, 10)
        for i, idx in enumerate(indices):
            self.assertEqual(int(idx[0]), expected_in_order[i])

        # invalid query should raise ValueError
        try:
            db.queryDataset(dset_id, "foobar")
            self.fail("Expected ValueError for invalid query field")
        except ValueError:
            pass

        # query with update_value
        indices = db.queryDataset(dset_id, query, update_value={"open": -999, "close": 999})
        self.assertEqual(indices.shape, (4, 1))
        sel = selections.select(shape, indices)
        values = db.getDatasetValues(dset_id, sel)
        for i in range(len(values)):
            self.assertEqual(values[i]["open"], -999)
            self.assertEqual(values[i]["close"], 999)
            self.assertEqual(values[i]["symbol"], b"AAPL")

        db.close()

    def testQueryDataset2D(self):
        data_arr = self._make_tabular_arr()
        nrows = data_arr.shape[0]
        data_arr = data_arr.reshape((nrows // 2, 2))
        shape = data_arr.shape

        db = Hdf5db(app_logger=self.log)
        db.open()
        dset_id = db.createDataset(shape, dtype=data_arr.dtype)
        sel_all = selections.select(shape, ...)
        db.setDatasetValues(dset_id, sel_all, data_arr)

        # AAPL appears at (0,1), (2,0), (3,1), (5,0) in the 6×2 layout
        query = "field('symbol') == b'AAPL'"
        indices = db.queryDataset(dset_id, query)
        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(indices.shape, (4, 2))
        expected_indexes = {(0, 1), (2, 0), (3, 1), (5, 0)}
        for row in indices:
            self.assertIn(tuple(int(x) for x in row), expected_indexes)

        # query with selection (second column only: rows 0-5, col 1)
        slices = (slice(0, 6, 1), slice(1, 2, 1))
        sel = selections.select(shape, slices)
        indices = db.queryDataset(dset_id, query, sel=sel)
        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(indices.shape, (2, 2))
        expected_indexes = [(0, 1), (3, 1)]
        for i, row in enumerate(indices):
            self.assertEqual(tuple(int(x) for x in row), expected_indexes[i])

        # query with update_value
        indices = db.queryDataset(dset_id, query, update_value={"open": -999, "close": 999})
        self.assertEqual(indices.shape, (4, 2))
        sel = selections.select(shape, indices)
        values = db.getDatasetValues(dset_id, sel)
        for i in range(len(values)):
            self.assertEqual(values[i]["open"], -999)
            self.assertEqual(values[i]["close"], 999)
            self.assertEqual(values[i]["symbol"], b"AAPL")

        db.close()

    def testChunkIterator1D(self):
        shape = (10,)
        dtype = np.int32
        data = np.arange(10, dtype=dtype)
        cpl = {"layout": {"class": "H5D_CHUNKED", "dims": (3,)}}

        db = Hdf5db(app_logger=self.log)
        db.open()
        dset_id = db.createDataset(shape, dtype=dtype, cpl=cpl)
        sel_all = selections.select(shape, ...)
        db.setDatasetValues(dset_id, sel_all, data)

        it = db.getChunkIterator(dset_id)
        self.assertIsInstance(it, ChunkIterator)
        chunks = list(it)
        self.assertEqual(len(chunks), 4)  # ceil(10/3)
        for chunk in chunks:
            self.assertIsInstance(chunk, np.ndarray)
        reconstructed = np.concatenate(chunks)
        np.testing.assert_array_equal(reconstructed, data)

        db.close()

    def testChunkIterator2D(self):
        shape = (6, 5)
        dtype = np.int32
        data = np.arange(30, dtype=dtype).reshape(shape)
        cpl = {"layout": {"class": "H5D_CHUNKED", "dims": (4, 3)}}

        db = Hdf5db(app_logger=self.log)
        db.open()
        dset_id = db.createDataset(shape, dtype=dtype, cpl=cpl)
        sel_all = selections.select(shape, ...)
        db.setDatasetValues(dset_id, sel_all, data)

        total_elements = 0
        collected = []
        for chunk in db.getChunkIterator(dset_id):
            self.assertIsInstance(chunk, np.ndarray)
            total_elements += chunk.size
            collected.append(chunk.reshape(-1))
        self.assertEqual(total_elements, data.size)
        # every element should appear exactly once across the chunks
        np.testing.assert_array_equal(np.sort(np.concatenate(collected)), np.sort(data.reshape(-1)))

        db.close()

    def testChunkIteratorWithSelection(self):
        shape = (10,)
        dtype = np.int32
        data = np.arange(10, dtype=dtype)
        cpl = {"layout": {"class": "H5D_CHUNKED", "dims": (3,)}}

        db = Hdf5db(app_logger=self.log)
        db.open()
        dset_id = db.createDataset(shape, dtype=dtype, cpl=cpl)
        sel_all = selections.select(shape, ...)
        db.setDatasetValues(dset_id, sel_all, data)

        sel = selections.select(shape, slice(2, 8))
        reconstructed = np.concatenate(list(db.getChunkIterator(dset_id, sel=sel)))
        np.testing.assert_array_equal(reconstructed, data[2:8])

        db.close()

    def testChunkIteratorNonChunkedLayout(self):
        # with no chunked layout, the whole dataset is treated as a single chunk
        shape = (5, 4)
        dtype = np.int32
        data = np.arange(20, dtype=dtype).reshape(shape)

        db = Hdf5db(app_logger=self.log)
        db.open()
        dset_id = db.createDataset(shape, dtype=dtype)
        sel_all = selections.select(shape, ...)
        db.setDatasetValues(dset_id, sel_all, data)

        chunks = list(db.getChunkIterator(dset_id))
        self.assertEqual(len(chunks), 1)
        np.testing.assert_array_equal(chunks[0], data)

        db.close()

    def testChunkIteratorInvalid(self):
        shape = (10,)
        dtype = np.int32
        cpl = {"layout": {"class": "H5D_CHUNKED", "dims": (3,)}}

        db = Hdf5db(app_logger=self.log)
        db.open()
        dset_id = db.createDataset(shape, dtype=dtype, cpl=cpl)

        # scalar datasets aren't supported
        scalar_dset_id = db.createDataset((), dtype=dtype)
        with self.assertRaises(ValueError):
            db.getChunkIterator(scalar_dset_id)

        # fancy/point selections aren't supported
        fancy_sel = selections.select(shape, [1, 3, 5])
        with self.assertRaises(ValueError):
            db.getChunkIterator(dset_id, sel=fancy_sel)

        # selection shape must match the dataset shape
        mismatched_sel = selections.select((20,), ...)
        with self.assertRaises(TypeError):
            db.getChunkIterator(dset_id, sel=mismatched_sel)

        db.close()

    def testGetDatasetValuesByQuery1D(self):
        data_arr = self._make_tabular_arr()
        shape = data_arr.shape
        # small chunk size so matches span multiple chunks
        cpl = {"layout": {"class": "H5D_CHUNKED", "dims": (3,)}}

        db = Hdf5db(app_logger=self.log)
        db.open()
        dset_id = db.createDataset(shape, dtype=data_arr.dtype, cpl=cpl)
        sel_all = selections.select(shape, ...)
        db.setDatasetValues(dset_id, sel_all, data_arr)

        query = "field('symbol') == b'AAPL'"
        values = db.getDatasetValues(dset_id, sel_all, query=query)
        self.assertIsInstance(values, np.ndarray)
        self.assertEqual(values.dtype, data_arr.dtype)
        self.assertEqual(values.shape, (4,))
        expected_indexes = (1, 4, 7, 10)
        for i, val in enumerate(values):
            self.assertEqual(val, data_arr[expected_indexes[i]])

        # same result as fetching indices via queryDataset and point-reading them
        indices = db.queryDataset(dset_id, query)
        sel_points = selections.select(shape, [int(idx[0]) for idx in indices])
        expected_values = db.getDatasetValues(dset_id, sel_points)
        np.testing.assert_array_equal(values, expected_values)

        # query with no results
        no_match = db.getDatasetValues(dset_id, sel_all, query="field('symbol') == b'XYZ'")
        self.assertIsInstance(no_match, np.ndarray)
        self.assertEqual(no_match.dtype, data_arr.dtype)
        self.assertEqual(no_match.shape, (0,))

        # query with a selection (rows 2-11)
        sel = selections.select(shape, slice(2, 12))
        values = db.getDatasetValues(dset_id, sel, query=query)
        self.assertEqual(values.shape, (3,))
        expected_in_order = (4, 7, 10)
        for i, val in enumerate(values):
            self.assertEqual(val, data_arr[expected_in_order[i]])

        db.close()

    def testGetDatasetValuesByQuery2D(self):
        nrows = 10
        ncols = 10
        shape = (nrows, ncols)
        dtype = np.int32
        cpl = {"layout": {"class": "H5D_CHUNKED", "dims": (4, 3)}}

        db = Hdf5db(app_logger=self.log)
        db.open()
        dset_id = db.createDataset(shape, dtype=dtype, cpl=cpl)
        arr = np.zeros(shape, dtype=dtype)
        for i in range(nrows):
            for j in range(ncols):
                arr[i, j] = i * j
        sel_all = selections.select(shape, ...)
        db.setDatasetValues(dset_id, sel_all, arr)

        query = "field('_') > 10"
        values = db.getDatasetValues(dset_id, sel_all, query=query)
        self.assertIsInstance(values, np.ndarray)
        self.assertEqual(values.shape, (56,))
        for val in values:
            self.assertTrue(val > 10)
        # every matching value should appear exactly once
        expected = sorted(arr[arr > 10].tolist())
        self.assertEqual(sorted(values.tolist()), expected)

        db.close()

    def testGetDatasetValuesByQueryFancySelection(self):
        data_arr = self._make_tabular_arr()
        shape = data_arr.shape

        db = Hdf5db(app_logger=self.log)
        db.open()
        dset_id = db.createDataset(shape, dtype=data_arr.dtype)
        sel_all = selections.select(shape, ...)
        db.setDatasetValues(dset_id, sel_all, data_arr)

        # a point/fancy selection isn't chunk-iterable - exercises the
        # single-fetch fallback path (AAPL is at indices 1, 4, 7 of the six
        # selected rows 1, 2, 4, 5, 7, 8)
        point_sel = selections.select(shape, [1, 2, 4, 5, 7, 8])
        query = "field('symbol') == b'AAPL'"
        values = db.getDatasetValues(dset_id, point_sel, query=query)
        self.assertEqual(values.shape, (3,))
        for val in values:
            self.assertEqual(val["symbol"], b"AAPL")

        db.close()

    def testCreateReferenceDataset(self):
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()

        dset_id = db.createDataset(shape=(), dtype=np.int32)
        db.createHardLink(root_id, "DS1", dset_id)

        dt = special_dtype(ref=Reference)

        # create a ref datsaet
        shape = (4, )
        ref_dset_id = db.createDataset(shape=shape, dtype=dt)

        # assign a ref to ds1
        ref_arr = np.zeros(shape, dtype=dt)
        ds1_ref = "datasets/" + dset_id
        ref_arr[0] = ds1_ref
        sel_all = selections.select(shape, ...)
        db.setDatasetValues(ref_dset_id, sel_all, ref_arr)
        sel = selections.select(shape, (slice(0, 2),))
        arr = db.getDatasetValues(ref_dset_id, sel)
        self.assertEqual(arr.shape, (2, ))
        self.assertEqual(arr.dtype, dt)
        self.assertEqual(arr[0], ds1_ref.encode())
        self.assertEqual(arr[1], b'')

        db.close()

    def testCreateRegionReferenceDataset(self):
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()

        # target dataset that the region reference will point into
        target_shape = (10,)
        target_id = db.createDataset(shape=target_shape, dtype=np.int32)
        db.createHardLink(root_id, "DS1", target_id)

        # build a RegionReference: id of the target dataset + a selection on it
        sel = selections.select(target_shape, slice(2, 8))
        ref = RegionReference("datasets/" + target_id, sel)
        raw = ref.tobytes()

        # RegionReference is a variable-length ("O") type - its size depends
        # on the bound selection, not just the referenced dataset - so
        # (unlike a plain object Reference) it isn't a fixed-width dtype
        dt = special_dtype(ref=RegionReference)
        self.assertEqual(dt.kind, "O")

        # create a ref dataset
        shape = (4, )
        ref_dset_id = db.createDataset(shape=shape, dtype=dt)

        # assign a region ref to element 0, leave the rest empty (no ref)
        ref_arr = np.empty(shape, dtype=dt)
        ref_arr[0] = raw
        for i in range(1, shape[0]):
            ref_arr[i] = b''
        sel_all = selections.select(shape, ...)
        db.setDatasetValues(ref_dset_id, sel_all, ref_arr)
        sel_read = selections.select(shape, (slice(0, 2),))
        arr = db.getDatasetValues(ref_dset_id, sel_read)
        self.assertEqual(arr.shape, (2, ))
        self.assertEqual(arr.dtype, dt)
        self.assertEqual(arr[1], b'')

        # decode the round-tripped region reference and confirm it matches
        round_tripped = RegionReference.frombytes(arr[0])
        self.assertEqual(round_tripped.id, ref.id)
        round_tripped_sel = selections.Selection.frombytes(round_tripped.selection_bytes)
        self.assertEqual(round_tripped_sel, sel)

        db.close()

    def testCreateOpaqueDataset(self):
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()

        dt = np.dtype("V2")
        shape = (4,)
        dset_id = db.createDataset(shape=shape, dtype=dt)
        db.createHardLink(root_id, "DS1", dset_id)

        arr = np.zeros(shape, dtype=dt)
        arr[3] = b'\xfe\xff'
        sel_all = selections.select(shape, ...)
        db.setDatasetValues(dset_id, sel_all, arr)

        result = db.getDatasetValues(dset_id, sel_all)
        self.assertEqual(result.dtype, dt)
        self.assertEqual([v.tobytes() for v in result], [b'\x00\x00'] * 3 + [b'\xfe\xff'])

        dset_json = db.getObjectById(dset_id)
        self.assertEqual(dset_json["type"], {"class": "H5T_OPAQUE", "size": 2})

        db.close()

    def testCreateOpaqueAttribute(self):
        # matches the format used in data/json/opaque_attr.json:
        # {"value": "<base64>", "encoding": "base64"}
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()

        dt = np.dtype("V2")
        value = np.zeros((), dtype=dt)
        value[()] = b'\xfe\xff'
        db.createAttribute(root_id, "A1", value, dtype=dt)

        attr = db.getAttribute(root_id, "A1")
        self.assertEqual(attr["type"], {"class": "H5T_OPAQUE", "size": 2})
        self.assertEqual(attr["shape"], {"class": "H5S_SCALAR"})
        self.assertEqual(attr["value"], "/v8=")
        self.assertEqual(attr["encoding"], "base64")

        attr_value = db.getAttributeValue(root_id, "A1")
        self.assertEqual(attr_value.dtype, dt)
        self.assertEqual(attr_value.tobytes(), b'\xfe\xff')

        db.close()

    def testClosedProperty(self):
        # closed before any plugin is set at all
        db = Hdf5db(app_logger=self.log)
        self.assertFalse(db.closed)

        # set a plugin directly (bypassing db.open())
        plugin = NullPlugin(None, app_logger=self.log)
        db.plugin = plugin
        self.assertTrue(db.closed)  # plugin hasn't been opened yet

        plugin.open()
        self.assertFalse(db.closed)

        plugin.close()
        self.assertTrue(db.closed)

    def testGetDtype(self):
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()

        dt = np.dtype([("x", np.int32), ("y", np.float64)])
        ctype_id = db.createCommittedType(dt)
        db.createHardLink(root_id, "ctype", ctype_id)

        # obj_json whose "type" is a direct reference (by id) to a committed
        # datatype, rather than an inline type description - exercises the
        # committed-type branch of getDtype()
        obj_json = {"type": ctype_id}
        resolved_dtype = db.getDtype(obj_json)
        self.assertEqual(resolved_dtype, dt)

        # sanity check against the more common "inline type json" branch -
        # the committed type's own json describes its type inline
        ctype_json = db.getObjectById(ctype_id)
        self.assertEqual(db.getDtype({"type": ctype_json["type"]}), dt)

        # obj_json with no "type" key at all (e.g. a group)
        with self.assertRaises(TypeError):
            db.getDtype({"links": {}})

        db.close()

    def testGetObjectByPath(self):
        db = Hdf5db(app_logger=self.log)
        root_id = db.open()

        g1_id = db.createGroup()
        db.createHardLink(root_id, "g1", g1_id)
        db.createAttribute(g1_id, "a1", "hello")

        dset_id = db.createDataset((4,), dtype=np.int32)
        db.createHardLink(g1_id, "dset", dset_id)

        root_obj = db.getObjectByPath("/")
        self.assertEqual(root_obj, db.getObjectById(root_id))

        g1_obj = db.getObjectByPath("g1")
        self.assertEqual(g1_obj, db.getObjectById(g1_id))
        self.assertIn("a1", g1_obj["attributes"])

        dset_obj = db.getObjectByPath("/g1/dset")
        self.assertEqual(dset_obj, db.getObjectById(dset_id))
        self.assertEqual(dset_obj["type"]["class"], "H5T_INTEGER")

        # non-existent path raises KeyError (mirrors getObjectIdByPath)
        with self.assertRaises(KeyError):
            db.getObjectByPath("/g1/nosuch")

        db.close()

    def testTrackingSetsAndDeleteObject(self):
        filepath = "test/unit/out/hdf5db_testTrackingSetsAndDeleteObject.json"
        db = Hdf5db(app_logger=self.log)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()

        # fresh db - no dirty/deleted/resized objects tracked yet
        self.assertEqual(db.dirty_objects, set())
        self.assertEqual(db.deleted_objects, set())
        self.assertEqual(db.resized_datasets, set())

        g1_id = db.createGroup()
        db.createHardLink(root_id, "g1", g1_id)
        self.assertIn(g1_id, db.new_objects)

        shape = (4, 4)
        maxdims = (None, 8)
        cpl = {"layout": {"class": "H5D_CHUNKED", "dims": shape}}
        dset_id = db.createDataset(shape, maxdims=maxdims, dtype=np.int32, cpl=cpl)
        db.createHardLink(root_id, "dset", dset_id)
        self.assertIn(dset_id, db.new_objects)

        # flush persists the new objects, clearing the new/dirty/resized sets
        self.assertTrue(db.flush())
        self.assertEqual(db.new_objects, set())
        self.assertEqual(db.dirty_objects, set())
        self.assertEqual(db.resized_datasets, set())

        # modifying a previously-flushed (no longer "new") object marks it dirty
        db.createAttribute(g1_id, "a1", "hello")
        self.assertIn(g1_id, db.dirty_objects)

        # resizing a previously-flushed (no longer "new") dataset marks it resized
        db.resizeDataset(dset_id, (4, 8))
        self.assertIn(dset_id, db.resized_datasets)

        # deleteObject removes the object from the dirty set and adds it to
        # deleted_objects
        db.deleteObject(g1_id)
        self.assertIn(g1_id, db.deleted_objects)
        self.assertNotIn(g1_id, db.dirty_objects)
        self.assertNotIn(g1_id, db.getCollection("groups"))
        self.assertFalse(g1_id in db)

        # deleteObject removes a resized dataset from the resized_datasets set
        db.deleteObject(dset_id)
        self.assertIn(dset_id, db.deleted_objects)
        self.assertNotIn(dset_id, db.resized_datasets)

        # deleting the root group is not allowed
        with self.assertRaises(KeyError):
            db.deleteObject(root_id)

        # deleting an id that was never created should raise
        with self.assertRaises(KeyError):
            db.deleteObject("d-does-not-exist")

        # a freshly created (still "new") object can also be deleted directly
        g2_id = db.createGroup()
        self.assertIn(g2_id, db.new_objects)
        db.deleteObject(g2_id)
        self.assertNotIn(g2_id, db.new_objects)
        self.assertIn(g2_id, db.deleted_objects)

        db.close()

    def testAutoFlushDefaultsAndOverrides(self):
        from h5json.hdf5db import DEFAULT_AUTO_FLUSH_MEMORY, DEFAULT_AUTO_FLUSH_INTERVAL

        db = Hdf5db(app_logger=self.log)
        self.assertEqual(db.auto_flush_memory, DEFAULT_AUTO_FLUSH_MEMORY)
        self.assertEqual(db.auto_flush_interval, DEFAULT_AUTO_FLUSH_INTERVAL)
        self.assertEqual(db.memory_usage, 0)
        # last_flush_time is set at construction, before any flush() has happened
        self.assertTrue(db.last_flush_time > 0)

        db2 = Hdf5db(app_logger=self.log, auto_flush_memory=1024, auto_flush_interval=5)
        self.assertEqual(db2.auto_flush_memory, 1024)
        self.assertEqual(db2.auto_flush_interval, 5)

        db3 = Hdf5db(app_logger=self.log, auto_flush_memory=None, auto_flush_interval=None)
        self.assertIsNone(db3.auto_flush_memory)
        self.assertIsNone(db3.auto_flush_interval)

    def testMemoryUsageTracksDatasetUpdates(self):
        filepath = "test/unit/out/hdf5db_testMemoryUsageTracksDatasetUpdates.json"
        db = Hdf5db(app_logger=self.log, auto_flush_memory=None, auto_flush_interval=None)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()

        self.assertEqual(db.memory_usage, 0)

        shape = (100,)
        dset_id = db.createDataset(shape, dtype=np.int64)
        db.createHardLink(root_id, "dset", dset_id)
        self.assertEqual(db.memory_usage, 0)  # no values written yet

        arr = np.arange(100, dtype=np.int64)
        sel_all = selections.select(shape, ...)
        db.setDatasetValues(dset_id, sel_all, arr)
        self.assertEqual(db.memory_usage, arr.nbytes)

        # a full-coverage rewrite discards (and un-counts) the prior update
        db.setDatasetValues(dset_id, sel_all, arr)
        self.assertEqual(db.memory_usage, arr.nbytes)

        # a partial (hyperslab) update adds to, rather than replaces, the total
        sel_partial = selections.select(shape, slice(0, 10))
        db.setDatasetValues(dset_id, sel_partial, arr[:10])
        self.assertEqual(db.memory_usage, arr.nbytes + arr[:10].nbytes)

        # flush() resets the tracked memory usage back to 0
        db.flush()
        self.assertEqual(db.memory_usage, 0)
        db.close()

    def testAutoFlushOnMemoryThreshold(self):
        filepath = "test/unit/out/hdf5db_testAutoFlushOnMemoryThreshold.json"
        arr = np.zeros((100,), dtype=np.int64)  # 800 bytes
        db = Hdf5db(app_logger=self.log, auto_flush_memory=arr.nbytes, auto_flush_interval=None)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()

        shape = arr.shape
        dset_id = db.createDataset(shape, dtype=np.int64)
        db.createHardLink(root_id, "dset", dset_id)
        db.flush()  # start with a clean slate so createDataset above doesn't count

        sel_all = selections.select(shape, ...)
        # writing an update whose size meets the threshold triggers an
        # automatic flush - without ever calling db.flush() explicitly
        db.setDatasetValues(dset_id, sel_all, arr)
        self.assertEqual(db.memory_usage, 0)
        self.assertEqual(db.dirty_objects, set())
        self.assertEqual(db.new_objects, set())

        db.close()

    def testAutoFlushOnTimeInterval(self):
        filepath = "test/unit/out/hdf5db_testAutoFlushOnTimeInterval.json"
        db = Hdf5db(app_logger=self.log, auto_flush_memory=None, auto_flush_interval=0.05)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()

        g1_id = db.createGroup()
        db.createHardLink(root_id, "g1", g1_id)
        db.flush()  # clean slate, resets last_flush_time

        time.sleep(0.1)  # exceed the 0.05s auto_flush_interval

        # any subsequent mutating call should now trigger an automatic flush
        db.createAttribute(g1_id, "a1", "hello")
        self.assertEqual(db.dirty_objects, set())

        db.close()

    def testAutoFlushDisabled(self):
        filepath = "test/unit/out/hdf5db_testAutoFlushDisabled.json"
        # a tiny memory threshold and interval would normally trigger
        # immediately, but passing None for both disables auto-flush entirely
        db = Hdf5db(app_logger=self.log, auto_flush_memory=None, auto_flush_interval=None)
        db.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = db.open()

        shape = (100,)
        dset_id = db.createDataset(shape, dtype=np.int64)
        db.createHardLink(root_id, "dset", dset_id)
        db.flush()

        time.sleep(0.05)
        arr = np.ones(shape, dtype=np.int64)
        sel_all = selections.select(shape, ...)
        db.setDatasetValues(dset_id, sel_all, arr)

        # nothing should have been auto-flushed
        self.assertEqual(db.memory_usage, arr.nbytes)
        self.assertIn(dset_id, db.dirty_objects)

        db.close()

    def testAutoFlushJsonRoundTrip(self):
        # confirm data written via an automatic (not explicit) flush is
        # actually persisted correctly - not just that in-memory tracking
        # state looks right
        filepath = "test/unit/out/hdf5db_testAutoFlushJsonRoundTrip.json"
        shape = (50,)
        arr = np.arange(50, dtype=np.int64)  # 400 bytes

        wdb = Hdf5db(app_logger=self.log, auto_flush_memory=arr.nbytes, auto_flush_interval=None)
        wdb.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = wdb.open()

        dset_id = wdb.createDataset(shape, dtype=np.int64)
        wdb.createHardLink(root_id, "dset", dset_id)
        wdb.flush()  # clean slate

        sel_all = selections.select(shape, ...)
        wdb.setDatasetValues(dset_id, sel_all, arr)  # crosses memory threshold
        self.assertEqual(wdb.memory_usage, 0)  # confirms auto-flush already ran
        wdb.close()

        rdb = Hdf5db(app_logger=self.log)
        rdb.plugin = H5JsonPlugin(filepath, read_only=True, app_logger=self.log)
        rdb.open()
        read_dset_id = rdb.getObjectIdByPath("/dset")
        result = rdb.getDatasetValues(read_dset_id, selections.select(shape, ...))
        self.assertTrue(np.array_equal(result, arr))
        rdb.close()

    def testReadAll(self):
        filepath = "test/unit/out/hdf5db_testReadAll.json"

        wdb = Hdf5db(app_logger=self.log)
        wdb.plugin = H5JsonPlugin(filepath, app_logger=self.log)
        root_id = wdb.open()

        g1_id = wdb.createGroup()
        wdb.createHardLink(root_id, "g1", g1_id)
        g2_id = wdb.createGroup()
        wdb.createHardLink(g1_id, "g2", g2_id)
        dset_id = wdb.createDataset((4,), dtype=np.int32)
        wdb.createHardLink(g2_id, "dset", dset_id)
        wdb.createAttribute(g1_id, "a1", "hello")
        wdb.close()

        rdb = Hdf5db(app_logger=self.log)
        rdb.plugin = H5JsonPlugin(filepath, read_only=True, app_logger=self.log)
        reopened_root_id = rdb.open()
        self.assertEqual(reopened_root_id, root_id)

        # before readAll, nothing has been pulled into the in-memory db yet
        self.assertEqual(rdb.getCollection(), [])

        rdb.readAll()

        obj_ids = rdb.getCollection()
        self.assertEqual(len(obj_ids), 4)  # root, g1, g2, dset
        for expected_id in (root_id, g1_id, g2_id, dset_id):
            self.assertIn(expected_id, obj_ids)

        groups = rdb.getCollection("groups")
        self.assertEqual(len(groups), 3)
        datasets = rdb.getCollection("datasets")
        self.assertEqual(datasets, [dset_id])

        # attributes on objects pulled in via readAll should be usable too
        self.assertEqual(rdb.getAttributes(g1_id), ["a1"])

        # readAll should raise once the db is closed
        rdb.close()
        with self.assertRaises(IOError):
            rdb.readAll()


if __name__ == "__main__":
    # setup test files

    unittest.main()
