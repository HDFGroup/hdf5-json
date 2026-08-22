##############################################################################
# Copyright by The HDF Group.                                                #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HSDS (HDF5 Scalable Data Service), Libraries and      #
# Utilities.  The full HSDS copyright notice, including                      #
# terms governing use, modification, and redistribution, is contained in     #
# the file COPYING, which can be found at the root of the source code        #
# distribution tree.  If you do not have access to this file, you may        #
# request a copy from help@hdfgroup.org.                                     #
##############################################################################
import unittest
import logging
import numpy as np

from h5json import hdf5dtype
from h5json import selections
from h5json.hdf5dtype import special_dtype
from h5json.hdf5dtype import check_dtype
from h5json.hdf5dtype import Reference
from h5json.hdf5dtype import RegionReference
from h5json.hdf5dtype import isOpaqueDtype
from h5json.hdf5dtype import isVlen
from h5json.objid import createObjId, getUuidFromId


class Hdf5dtypeTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Hdf5dtypeTest, self).__init__(*args, **kwargs)
        # main
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    def testGetBaseTypeJson(self):
        type_json = hdf5dtype.getBaseTypeJson("H5T_IEEE_F64LE")
        self.assertTrue("class" in type_json)
        self.assertEqual(type_json["class"], "H5T_FLOAT")
        self.assertTrue("base" in type_json)
        self.assertEqual(type_json["base"], "H5T_IEEE_F64LE")

        type_json = hdf5dtype.getBaseTypeJson("H5T_IEEE_F16LE")
        self.assertTrue("class" in type_json)
        self.assertEqual(type_json["class"], "H5T_FLOAT")
        self.assertTrue("base" in type_json)
        self.assertEqual(type_json["base"], "H5T_IEEE_F16LE")

        type_json = hdf5dtype.getBaseTypeJson("H5T_STD_I32LE")
        self.assertTrue("class" in type_json)
        self.assertEqual(type_json["class"], "H5T_INTEGER")
        self.assertTrue("base" in type_json)
        self.assertEqual(type_json["base"], "H5T_STD_I32LE")

        try:
            hdf5dtype.getBaseTypeJson("foobar")
            self.assertTrue(False)
        except TypeError:
            pass  # expected

    def testBaseIntegerTypeItem(self):
        dt = np.dtype("<i1")
        typeItem = hdf5dtype.getTypeItem(dt)
        self.assertEqual(typeItem["class"], "H5T_INTEGER")
        self.assertEqual(typeItem["base"], "H5T_STD_I8LE")
        typeItem = hdf5dtype.getTypeResponse(typeItem)  # non-verbose format
        self.assertEqual(typeItem["class"], "H5T_INTEGER")
        self.assertEqual(typeItem["base"], "H5T_STD_I8LE")

    def testBaseFloatTypeItem(self):
        dt = np.dtype("<f8")
        typeItem = hdf5dtype.getTypeItem(dt)
        self.assertEqual(typeItem["class"], "H5T_FLOAT")
        self.assertEqual(typeItem["base"], "H5T_IEEE_F64LE")
        typeItem = hdf5dtype.getTypeResponse(typeItem)  # non-verbose format
        self.assertEqual(typeItem["class"], "H5T_FLOAT")
        self.assertEqual(typeItem["base"], "H5T_IEEE_F64LE")

    def testBaseFloat16TypeItem(self):
        dt = np.dtype("<f2")
        typeItem = hdf5dtype.getTypeItem(dt)
        self.assertEqual(typeItem["class"], "H5T_FLOAT")
        self.assertEqual(typeItem["base"], "H5T_IEEE_F16LE")
        typeItem = hdf5dtype.getTypeResponse(typeItem)  # non-verbose format
        self.assertEqual(typeItem["class"], "H5T_FLOAT")
        self.assertEqual(typeItem["base"], "H5T_IEEE_F16LE")

    def testBaseStringTypeItem(self):
        dt = np.dtype("S3")
        typeItem = hdf5dtype.getTypeItem(dt)
        self.assertEqual(typeItem["class"], "H5T_STRING")
        self.assertEqual(typeItem["length"], 3)
        self.assertEqual(typeItem["strPad"], "H5T_STR_NULLPAD")
        self.assertEqual(typeItem["charSet"], "H5T_CSET_ASCII")

    def testBaseStringUTFTypeItem(self):
        dt = np.dtype("U3")
        typeItem = hdf5dtype.getTypeItem(dt)
        self.assertEqual(typeItem["class"], "H5T_STRING")
        # type item length in bytes (may no actual be enough space for some UTF strings)
        self.assertEqual(typeItem["length"], 12)
        self.assertEqual(typeItem["strPad"], "H5T_STR_NULLPAD")
        self.assertEqual(typeItem["charSet"], "H5T_CSET_UTF8")

    def testBaseVLenAsciiTypeItem(self):
        dt = special_dtype(vlen=bytes)
        typeItem = hdf5dtype.getTypeItem(dt)
        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeItem["class"], "H5T_STRING")
        self.assertEqual(typeItem["length"], "H5T_VARIABLE")
        self.assertEqual(typeItem["strPad"], "H5T_STR_NULLTERM")
        self.assertEqual(typeItem["charSet"], "H5T_CSET_ASCII")
        self.assertEqual(typeSize, "H5T_VARIABLE")

    def testBaseVLenUnicodeTypeItem(self):
        dt = special_dtype(vlen=str)
        typeItem = hdf5dtype.getTypeItem(dt)
        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeItem["class"], "H5T_STRING")
        self.assertEqual(typeItem["length"], "H5T_VARIABLE")
        self.assertEqual(typeItem["strPad"], "H5T_STR_NULLTERM")
        self.assertEqual(typeItem["charSet"], "H5T_CSET_UTF8")
        self.assertEqual(typeSize, "H5T_VARIABLE")

    def testBaseStringDTypeTypeItem(self):
        # numpy's StringDType (kind == "T") is accepted as a convenience
        # input, but reported identically to the "O"-kind vlen str case -
        # neither h5json's type descriptor nor HDF5 itself can distinguish
        # the two, so there's nothing to preserve by treating them differently
        if not hasattr(np, "dtypes") or not hasattr(np.dtypes, "StringDType"):
            self.skipTest("numpy.dtypes.StringDType not available in this numpy version")
        dt = np.dtypes.StringDType()
        typeItem = hdf5dtype.getTypeItem(dt)
        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeItem["class"], "H5T_STRING")
        self.assertEqual(typeItem["length"], "H5T_VARIABLE")
        self.assertEqual(typeItem["strPad"], "H5T_STR_NULLTERM")
        self.assertEqual(typeItem["charSet"], "H5T_CSET_UTF8")
        self.assertEqual(typeSize, "H5T_VARIABLE")

    def testBaseEnumTypeItem(self):
        mapping = {"RED": 0, "GREEN": 1, "BLUE": 2}
        dt = special_dtype(enum=(np.int8, mapping))
        typeItem = hdf5dtype.getTypeItem(dt)
        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeItem["class"], "H5T_ENUM")
        baseItem = typeItem["base"]
        self.assertEqual(baseItem["class"], "H5T_INTEGER")
        self.assertEqual(baseItem["base"], "H5T_STD_I8LE")
        self.assertTrue("members" in typeItem)
        members = typeItem["members"]
        expected = [{'name': 'RED', 'value': 0}, {'name': 'GREEN', 'value': 1}, {'name': 'BLUE', 'value': 2}]
        self.assertEqual(members, expected)
        self.assertEqual(typeSize, 1)

    def testBaseBoolTypeItem(self):
        typeItem = hdf5dtype.getTypeItem(np.dtype("bool"))
        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeItem["class"], "H5T_ENUM")
        baseItem = typeItem["base"]
        self.assertEqual(baseItem["class"], "H5T_INTEGER")
        self.assertEqual(baseItem["base"], "H5T_STD_I8LE")
        self.assertTrue("members" in typeItem)
        members = typeItem["members"]
        self.assertEqual(len(members), 2)
        self.assertEqual(members[0], {"name": "FALSE", "value": 0})
        self.assertEqual(members[1], {"name": "TRUE", "value": 1})
        self.assertEqual(typeSize, 1)

    def testBaseArrayTypeItem(self):
        dt = np.dtype("(2,2)<int32")
        typeItem = hdf5dtype.getTypeItem(dt)
        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeItem["class"], "H5T_ARRAY")
        self.assertTrue("dims" in typeItem)
        self.assertEqual(typeItem["dims"], (2, 2,))
        baseItem = typeItem["base"]
        self.assertEqual(baseItem["class"], "H5T_INTEGER")
        self.assertEqual(baseItem["base"], "H5T_STD_I32LE")
        self.assertEqual(typeSize, 16)

    def testObjReferenceTypeItem(self):
        dt = special_dtype(ref=Reference)
        typeItem = hdf5dtype.getTypeItem(dt)
        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeItem["class"], "H5T_REFERENCE")
        self.assertEqual(typeItem["base"], "H5T_STD_REF_OBJ")
        # length of obj id, e.g.:
        # g-b2c9a750-a557-11e7-ab09-0242ac110009
        self.assertEqual(typeSize, 48)

    def testRegionReferenceTypeItem(self):
        dt = special_dtype(ref=RegionReference)
        # unlike an object ref (a fixed "S48" type), a region ref's size
        # depends on the bound selection, not just the referenced dataset,
        # so it's a variable-length ("O") type - same as vlen strings/types
        self.assertEqual(dt.kind, "O")
        typeItem = hdf5dtype.getTypeItem(dt)
        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeItem["class"], "H5T_REFERENCE")
        self.assertEqual(typeItem["base"], "H5T_STD_REF_DSETREG")
        self.assertEqual(typeSize, "H5T_VARIABLE")

    def testRegionReferenceTypeItemRoundTrip(self):
        dt = special_dtype(ref=RegionReference)
        typeItem = hdf5dtype.getTypeItem(dt)
        dtRoundTrip = hdf5dtype.createDataType(typeItem)
        self.assertEqual(dtRoundTrip.kind, "O")
        self.assertTrue(dtRoundTrip.metadata["ref"] is RegionReference)

    def testCompoundArrayTypeItem(self):
        dt = np.dtype([("a", "<i1"), ("b", "S1", (10,))])
        typeItem = hdf5dtype.getTypeItem(dt)
        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeItem["class"], "H5T_COMPOUND")
        fields = typeItem["fields"]
        field_a = fields[0]
        self.assertEqual(field_a["name"], "a")
        field_a_type = field_a["type"]
        self.assertEqual(field_a_type["class"], "H5T_INTEGER")
        self.assertEqual(field_a_type["base"], "H5T_STD_I8LE")
        field_b = fields[1]
        self.assertEqual(field_b["name"], "b")
        field_b_type = field_b["type"]
        self.assertEqual(field_b_type["class"], "H5T_ARRAY")
        self.assertEqual(field_b_type["dims"], (10,))
        field_b_basetype = field_b_type["base"]
        self.assertEqual(field_b_basetype["class"], "H5T_STRING")
        self.assertEqual(typeSize, 11)

    def testEnumArrayTypeItem(self):
        mapping = {"RED": 0, "GREEN": 1, "BLUE": 2}
        dt_enum = special_dtype(enum=(np.int8, mapping))
        typeItem = hdf5dtype.getTypeItem(dt_enum)
        dt_array = np.dtype("(2,3)" + dt_enum.str, metadata=dict(dt_enum.metadata))

        typeItem = hdf5dtype.getTypeItem(dt_array)

        self.assertEqual(typeItem["class"], "H5T_ARRAY")
        self.assertTrue("dims" in typeItem)
        self.assertEqual(typeItem["dims"], (2, 3))
        baseItem = typeItem["base"]
        self.assertEqual(baseItem["class"], "H5T_ENUM")
        self.assertTrue("members" in baseItem)
        members = baseItem["members"]
        self.assertEqual(len(members), 3)
        self.assertEqual(members[0], {"name": "RED", "value": 0})
        self.assertEqual(members[1], {"name": "GREEN", "value": 1})
        self.assertEqual(members[2], {"name": "BLUE", "value": 2})
        self.assertTrue("base" in baseItem)
        basePrim = baseItem["base"]
        self.assertEqual(basePrim["class"], "H5T_INTEGER")
        self.assertEqual(basePrim["base"], "H5T_STD_I8LE")
        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeSize, 6)  # one-byte for base enum type * shape of (2,3)

    def testCompoundArrayVlenIntTypeItem(self):
        dt_vlen = special_dtype(vlen=np.int32)
        dt_arr = np.dtype((dt_vlen, (4,)))
        dt_compound = np.dtype(
            [("VALUE1", np.float64), ("VALUE2", np.int64), ("VALUE3", dt_arr)]
        )
        typeItem = hdf5dtype.getTypeItem(dt_compound)

        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeSize, "H5T_VARIABLE")
        self.assertEqual(typeItem["class"], "H5T_COMPOUND")
        fields = typeItem["fields"]
        field_a = fields[0]
        self.assertEqual(field_a["name"], "VALUE1")
        field_a_type = field_a["type"]
        self.assertEqual(field_a_type["class"], "H5T_FLOAT")
        self.assertEqual(field_a_type["base"], "H5T_IEEE_F64LE")
        field_b = fields[1]
        self.assertEqual(field_b["name"], "VALUE2")
        field_b_type = field_b["type"]
        self.assertEqual(field_b_type["class"], "H5T_INTEGER")
        self.assertEqual(field_b_type["base"], "H5T_STD_I64LE")
        field_c = fields[2]
        field_c_type = field_c["type"]
        self.assertEqual(field_c_type["class"], "H5T_ARRAY")
        self.assertEqual(field_c_type["dims"], (4,))
        field_c_type_base = field_c_type["base"]
        self.assertEqual(field_c_type_base["class"], "H5T_VLEN")
        self.assertEqual(field_c_type_base["size"], "H5T_VARIABLE")
        field_c_type_base_base = field_c_type_base["base"]
        self.assertEqual(field_c_type_base_base["class"], "H5T_INTEGER")
        self.assertEqual(field_c_type_base_base["base"], "H5T_STD_I32LE")

    def testCompoundArrayVlenStringTypeItem(self):
        dt_vlen = special_dtype(vlen=bytes)
        dt_arr = np.dtype((dt_vlen, (4,)))
        dt_compound = np.dtype(
            [("VALUE1", np.float64), ("VALUE2", np.int64), ("VALUE3", dt_arr)]
        )
        typeItem = hdf5dtype.getTypeItem(dt_compound)

        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeSize, "H5T_VARIABLE")
        self.assertEqual(typeItem["class"], "H5T_COMPOUND")
        fields = typeItem["fields"]
        field_a = fields[0]
        self.assertEqual(field_a["name"], "VALUE1")
        field_a_type = field_a["type"]
        self.assertEqual(field_a_type["class"], "H5T_FLOAT")
        self.assertEqual(field_a_type["base"], "H5T_IEEE_F64LE")
        field_b = fields[1]
        self.assertEqual(field_b["name"], "VALUE2")
        field_b_type = field_b["type"]
        self.assertEqual(field_b_type["class"], "H5T_INTEGER")
        self.assertEqual(field_b_type["base"], "H5T_STD_I64LE")
        field_c = fields[2]
        field_c_type = field_c["type"]

        self.assertEqual(field_c_type["class"], "H5T_ARRAY")
        self.assertEqual(field_c_type["dims"], (4,))
        field_c_base_type = field_c_type["base"]
        self.assertEqual(field_c_base_type["class"], "H5T_STRING")
        self.assertEqual(field_c_base_type["length"], "H5T_VARIABLE")
        self.assertEqual(field_c_base_type["charSet"], "H5T_CSET_ASCII")

    def testCompoundArrayVlenStr(self):
        dt_str = special_dtype(vlen=str)
        dt_arr_str = np.dtype((dt_str, (3, 2)))
        dt_compound = np.dtype([("VALUE1", "i4"), ("VALUE2", dt_arr_str)])
        self.assertTrue(isVlen(dt_compound))
        type_item = hdf5dtype.getTypeItem(dt_compound)
        typeSize = hdf5dtype.getItemSize(type_item)
        self.assertEqual(typeSize, "H5T_VARIABLE")
        self.assertEqual(type_item["class"], "H5T_COMPOUND")
        fields = type_item["fields"]
        field_a = fields[0]
        self.assertEqual(field_a["name"], "VALUE1")
        field_a_type = field_a["type"]
        self.assertEqual(field_a_type["class"], "H5T_INTEGER")
        self.assertEqual(field_a_type["base"], "H5T_STD_I32LE")

        field_b = fields[1]
        field_b_type = field_b["type"]

        self.assertEqual(field_b_type["class"], "H5T_ARRAY")
        self.assertEqual(field_b_type["dims"], (3, 2))
        field_b_base_type = field_b_type["base"]
        self.assertEqual(field_b_base_type["class"], "H5T_STRING")
        self.assertEqual(field_b_base_type["length"], "H5T_VARIABLE")
        self.assertEqual(field_b_base_type["charSet"], "H5T_CSET_UTF8")

    def testOpaqueTypeItem(self):
        dt = np.dtype("V200")
        self.assertTrue(isOpaqueDtype(dt))
        typeItem = hdf5dtype.getTypeItem(dt)
        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeItem["class"], "H5T_OPAQUE")
        self.assertTrue("base" not in typeItem)
        self.assertEqual(typeSize, 200)

    def testVlenDataItem(self):
        dt = special_dtype(vlen=np.dtype("int32"))
        typeItem = hdf5dtype.getTypeItem(dt)
        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeItem["class"], "H5T_VLEN")
        self.assertEqual(typeItem["size"], "H5T_VARIABLE")
        baseItem = typeItem["base"]
        self.assertEqual(baseItem["base"], "H5T_STD_I32LE")
        self.assertEqual(typeSize, "H5T_VARIABLE")

    def testVlenReferenceDataItem(self):
        ref_dt = special_dtype(ref=Reference)
        dt = special_dtype(vlen=ref_dt)
        typeItem = hdf5dtype.getTypeItem(dt)
        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeItem["class"], "H5T_VLEN")
        self.assertEqual(typeItem["size"], "H5T_VARIABLE")
        baseItem = typeItem["base"]
        self.assertEqual(baseItem["base"], "H5T_STD_REF_OBJ")
        self.assertEqual(typeSize, "H5T_VARIABLE")

    def testCompoundTypeItem(self):
        dt = np.dtype(
            [("temp", np.float32), ("pressure", np.float32), ("wind", np.int16)]
        )
        typeItem = hdf5dtype.getTypeItem(dt)
        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeItem["class"], "H5T_COMPOUND")
        self.assertTrue("fields" in typeItem)
        fields = typeItem["fields"]
        self.assertEqual(len(fields), 3)
        tempField = fields[0]
        self.assertEqual(tempField["name"], "temp")
        self.assertTrue("type" in tempField)
        tempFieldType = tempField["type"]
        self.assertEqual(tempFieldType["class"], "H5T_FLOAT")
        self.assertEqual(tempFieldType["base"], "H5T_IEEE_F32LE")
        self.assertEqual(typeSize, 10)

        typeItem = hdf5dtype.getTypeResponse(typeItem)  # non-verbose format
        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeItem["class"], "H5T_COMPOUND")
        self.assertTrue("fields" in typeItem)
        fields = typeItem["fields"]
        self.assertEqual(len(fields), 3)
        tempField = fields[0]
        self.assertEqual(tempField["name"], "temp")
        self.assertTrue("type" in tempField)
        tempFieldType = tempField["type"]
        self.assertEqual(tempFieldType["class"], "H5T_FLOAT")
        self.assertEqual(tempFieldType["base"], "H5T_IEEE_F32LE")
        self.assertEqual(typeSize, 10)

    def testCompoundOnfFieldTypeItem(self):
        dt = np.dtype([("temp", np.float32),])
        typeItem = hdf5dtype.getTypeItem(dt)
        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeItem["class"], "H5T_COMPOUND")
        self.assertTrue("fields" in typeItem)
        fields = typeItem["fields"]
        self.assertEqual(len(fields), 1)
        tempField = fields[0]
        self.assertEqual(tempField["name"], "temp")
        self.assertTrue("type" in tempField)
        tempFieldType = tempField["type"]
        self.assertEqual(tempFieldType["class"], "H5T_FLOAT")
        self.assertEqual(tempFieldType["base"], "H5T_IEEE_F32LE")
        self.assertEqual(typeSize, 4)

    def testCompoundOfCompoundTypeItem(self):
        dt1 = np.dtype([("x", np.float32), ("y", np.float32)])
        dt2 = np.dtype([("a", np.float32), ("b", np.float32), ("c", np.float32)])
        dt = np.dtype([("field1", dt1), ("field2", dt2)])
        typeItem = hdf5dtype.getTypeItem(dt)

        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeSize, 20)
        self.assertEqual(typeItem["class"], "H5T_COMPOUND")
        self.assertTrue("fields" in typeItem)
        fields = typeItem["fields"]
        self.assertEqual(len(fields), 2)
        field1 = fields[0]

        self.assertEqual(field1["name"], "field1")
        field1_type = field1["type"]
        self.assertEqual(field1_type["class"], "H5T_COMPOUND")
        field2 = fields[1]

        self.assertEqual(field2["name"], "field2")
        field2_type = field2["type"]
        self.assertEqual(field2_type["class"], "H5T_COMPOUND")

    def testCreateBaseType(self):
        dt = hdf5dtype.createDataType("H5T_STD_U32BE")
        self.assertEqual(dt.name, "uint32")
        self.assertEqual(dt.byteorder, ">")
        self.assertEqual(dt.kind, "u")
        self.assertFalse(isVlen(dt))

        dt = hdf5dtype.createDataType("H5T_STD_I16LE")
        self.assertEqual(dt.name, "int16")
        self.assertEqual(dt.kind, "i")

        dt = hdf5dtype.createDataType("H5T_IEEE_F64LE")
        self.assertEqual(dt.name, "float64")
        self.assertEqual(dt.kind, "f")
        self.assertFalse(isVlen(dt))

        dt = hdf5dtype.createDataType("H5T_IEEE_F32LE")
        self.assertEqual(dt.name, "float32")
        self.assertEqual(dt.kind, "f")
        self.assertFalse(isVlen(dt))

        typeItem = {"class": "H5T_INTEGER", "base": "H5T_STD_I32BE"}
        typeSize = hdf5dtype.getItemSize(typeItem)
        dt = hdf5dtype.createDataType(typeItem)
        self.assertEqual(dt.name, "int32")
        self.assertEqual(dt.kind, "i")
        self.assertEqual(typeSize, 4)
        self.assertFalse(isVlen(dt))

    def testCreateBaseStringType(self):
        typeItem = {"class": "H5T_STRING", "charSet": "H5T_CSET_ASCII", "length": 6}
        typeSize = hdf5dtype.getItemSize(typeItem)
        dt = hdf5dtype.createDataType(typeItem)
        self.assertEqual(dt.name, "bytes48")
        self.assertEqual(dt.kind, "S")
        self.assertEqual(typeSize, 6)
        self.assertFalse(isVlen(dt))

    def testCreateBaseUnicodeType(self):
        typeItem = {"class": "H5T_STRING", "charSet": "H5T_CSET_UTF8", "length": 6}

        dt = hdf5dtype.createDataType(typeItem)
        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertTrue(dt is not None)
        self.assertEqual(dt.name, "bytes48")
        self.assertEqual(dt.kind, "S")  # uses byte
        self.assertEqual(typeSize, 6)
        self.assertFalse(isVlen(dt))

    def testCreateNullTermStringType(self):
        typeItem = {
            "class": "H5T_STRING",
            "charSet": "H5T_CSET_ASCII",
            "length": 6,
            "strPad": "H5T_STR_NULLTERM",
        }
        typeSize = hdf5dtype.getItemSize(typeItem)
        dt = hdf5dtype.createDataType(typeItem)

        self.assertEqual(dt.name, "bytes48")
        self.assertEqual(dt.kind, "S")
        self.assertEqual(typeSize, 6)
        self.assertFalse(isVlen(dt))

    def testCreateVLenStringType(self):
        typeItem = {
            "class": "H5T_STRING",
            "charSet": "H5T_CSET_ASCII",
            "length": "H5T_VARIABLE",
        }
        typeSize = hdf5dtype.getItemSize(typeItem)
        dt = hdf5dtype.createDataType(typeItem)
        self.assertEqual(dt.name, "object")
        self.assertEqual(dt.kind, "O")
        self.assertEqual(check_dtype(vlen=dt), bytes)
        self.assertEqual(typeSize, "H5T_VARIABLE")
        self.assertTrue(isVlen(dt))

    def testCreateVLenStringArrayType(self):
        typeItem = {
            "class": "H5T_ARRAY",
            "dims": (2, 2),
            "base": {
                "class": "H5T_STRING",
                "charSet": "H5T_CSET_ASCII",
                "length": "H5T_VARIABLE",
            }
        }
        typeSize = hdf5dtype.getItemSize(typeItem)
        dt = hdf5dtype.createDataType(typeItem)
        self.assertEqual(dt.name, "void256")  # assuming 8-byte pointers
        self.assertEqual(dt.kind, "V")
        self.assertEqual(dt.shape, (2, 2))
        self.assertEqual(check_dtype(vlen=dt), None)
        self.assertEqual(check_dtype(vlen=dt.base), bytes)
        self.assertEqual(typeSize, "H5T_VARIABLE")
        self.assertEqual(dt.base.kind, 'O')
        self.assertTrue(isVlen(dt))

    def testCreateVLenUTF8Type(self):
        typeItem = {
            "class": "H5T_STRING",
            "charSet": "H5T_CSET_UTF8",
            "length": "H5T_VARIABLE",
        }
        typeSize = hdf5dtype.getItemSize(typeItem)
        dt = hdf5dtype.createDataType(typeItem)
        self.assertEqual(dt.name, "object")
        self.assertEqual(dt.kind, "O")
        self.assertEqual(check_dtype(vlen=dt), str)
        self.assertEqual(typeSize, "H5T_VARIABLE")
        self.assertTrue(isVlen(dt))

    def testCreateVLenDataType(self):
        typeItem = {"class": "H5T_VLEN", "base": "H5T_STD_I32BE"}
        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeSize, "H5T_VARIABLE")
        dt = hdf5dtype.createDataType(typeItem)
        self.assertEqual(dt.name, "object")
        self.assertEqual(dt.kind, "O")
        self.assertTrue(isVlen(dt))

    def testCreateOpaqueType(self):
        typeItem = {"class": "H5T_OPAQUE", "size": 200}
        typeSize = hdf5dtype.getItemSize(typeItem)
        dt = hdf5dtype.createDataType(typeItem)
        self.assertEqual(dt.name, "void1600")
        self.assertEqual(dt.kind, "V")
        self.assertEqual(typeSize, 200)
        self.assertFalse(isVlen(dt))

    def testCreateEnumType(self):
        typeItem = {
            "class": "H5T_ENUM",
            "base": {"base": "H5T_STD_I16LE", "class": "H5T_INTEGER"},
            "mapping": {"GAS": 2, "LIQUID": 1, "PLASMA": 3, "SOLID": 0},
        }

        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeSize, 2)
        dt = hdf5dtype.createDataType(typeItem)
        self.assertEqual(dt.name, "int16")
        self.assertEqual(dt.kind, "i")
        mapping = check_dtype(enum=dt)
        self.assertTrue(isinstance(mapping, dict))
        self.assertEqual(mapping["SOLID"], 0)
        self.assertEqual(mapping["LIQUID"], 1)
        self.assertEqual(mapping["GAS"], 2)
        self.assertEqual(mapping["PLASMA"], 3)
        self.assertFalse(isVlen(dt))

    def testCreateBoolType(self):
        typeItem = {
            "class": "H5T_ENUM",
            "base": {"base": "H5T_STD_I8LE", "class": "H5T_INTEGER"},
            "mapping": {"TRUE": 1, "FALSE": 0},
        }

        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeSize, 1)
        dt = hdf5dtype.createDataType(typeItem)
        self.assertEqual(dt.name, "bool")
        self.assertEqual(dt.kind, "b")
        self.assertEqual(typeSize, hdf5dtype.getDtypeItemSize(dt))
        self.assertFalse(isVlen(dt))

    def testCreateReferenceType(self):
        typeItem = {
            "class": "H5T_REFERENCE",
            "base": "H5T_STD_REF_OBJ",
            "length": 48,
            "charSet": "H5T_CSET_ASCII",
            "strPad": "H5T_STR_NULLPAD"
        }
        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeSize, 48)
        dt = hdf5dtype.createDataType(typeItem)
        self.assertEqual(dt.kind, "S")
        self.assertTrue(dt.metadata['ref'] is Reference)
        self.assertEqual(check_dtype(ref=dt), Reference)
        self.assertFalse(isVlen(dt))

    def testCreateVlenReferenceType(self):
        typeItem = {
            'class': 'H5T_VLEN',
            'base': {'class': 'H5T_REFERENCE', 'base': 'H5T_STD_REF_OBJ'}
        }
        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeSize, 'H5T_VARIABLE')
        dt = hdf5dtype.createDataType(typeItem)
        self.assertEqual(dt.kind, "O")
        base = dt.metadata['vlen']
        self.assertTrue(base.metadata['ref'] is Reference)
        self.assertEqual(check_dtype(ref=base), Reference)
        self.assertTrue(isVlen(dt))

    def testCreateCompoundType(self):
        typeItem = {
            "class": "H5T_COMPOUND",
            "fields": [
                {"name": "temp", "type": "H5T_IEEE_F32LE"},
                {"name": "pressure", "type": "H5T_IEEE_F32LE"},
                {
                    "name": "location",
                    "type": {
                        "length": "H5T_VARIABLE",
                        "charSet": "H5T_CSET_ASCII",
                        "class": "H5T_STRING",
                        "strPad": "H5T_STR_NULLTERM",
                    },
                },
                {"name": "wind", "type": "H5T_STD_I16LE"},
            ],
        }
        typeSize = hdf5dtype.getItemSize(typeItem)
        dt = hdf5dtype.createDataType(typeItem)
        self.assertEqual(dt.name, "void144")
        self.assertEqual(dt.kind, "V")
        self.assertEqual(len(dt.fields), 4)
        self.assertEqual(typeSize, hdf5dtype.getDtypeItemSize(dt))
        self.assertTrue(isVlen(dt))

        dtLocation = dt[2]
        self.assertEqual(dtLocation.name, "object")
        self.assertEqual(dtLocation.kind, "O")
        self.assertEqual(check_dtype(vlen=dtLocation), bytes)
        self.assertEqual(typeSize, "H5T_VARIABLE")
        self.assertEqual(typeSize, hdf5dtype.getDtypeItemSize(dtLocation))

    def testCreateCompoundInvalidFieldName(self):
        typeItem = {
            "class": "H5T_COMPOUND",
            "fields": [
                {
                    "name": "\u03b1",
                    "type": {"base": "H5T_STD_I32LE", "class": "H5T_INTEGER"},
                },
                {
                    "name": "\u03c9",
                    "type": {"base": "H5T_STD_I32LE", "class": "H5T_INTEGER"},
                },
            ],
        }
        try:
            hdf5dtype.createDataType(typeItem)
            self.assertTrue(False)
        except TypeError:
            pass  # expected

    def testCreateCompoundOfCompoundType(self):
        typeItem = {
            "class": "H5T_COMPOUND",
            "fields": [
                {
                    "name": "field1",
                    "type": {
                        "class": "H5T_COMPOUND",
                        "fields": [
                            {
                                "name": "x",
                                "type": {
                                    "class": "H5T_FLOAT",
                                    "base": "H5T_IEEE_F32LE",
                                },
                            },
                            {
                                "name": "y",
                                "type": {
                                    "class": "H5T_FLOAT",
                                    "base": "H5T_IEEE_F32LE",
                                },
                            },
                        ],
                    },
                },
                {
                    "name": "field2",
                    "type": {
                        "class": "H5T_COMPOUND",
                        "fields": [
                            {
                                "name": "a",
                                "type": {
                                    "class": "H5T_FLOAT",
                                    "base": "H5T_IEEE_F32LE",
                                },
                            },
                            {
                                "name": "b",
                                "type": {
                                    "class": "H5T_FLOAT",
                                    "base": "H5T_IEEE_F32LE",
                                },
                            },
                            {
                                "name": "c",
                                "type": {
                                    "class": "H5T_FLOAT",
                                    "base": "H5T_IEEE_F32LE",
                                },
                            },
                        ],
                    },
                },
            ],
        }
        dt = hdf5dtype.createDataType(typeItem)
        self.assertEqual(dt.name, "void160")
        self.assertEqual(dt.kind, "V")
        self.assertEqual(len(dt.fields), 2)
        self.assertFalse(isVlen(dt))
        dt_field1 = dt[0]
        self.assertEqual(dt_field1.name, "void64")
        self.assertEqual(dt_field1.kind, "V")
        self.assertEqual(len(dt_field1.fields), 2)
        dt_field2 = dt[1]
        self.assertEqual(dt_field2.name, "void96")
        self.assertEqual(dt_field2.kind, "V")
        self.assertEqual(len(dt_field2.fields), 3)

    def testCreateCompoundTypeUnicodeFields(self):
        typeItem = {
            "class": "H5T_COMPOUND",
            "fields": [
                {"name": u"temp", "type": "H5T_IEEE_F32LE"},
                {"name": u"pressure", "type": "H5T_IEEE_F32LE"},
                {"name": u"wind", "type": "H5T_STD_I16LE"},
            ],
        }
        typeSize = hdf5dtype.getItemSize(typeItem)
        dt = hdf5dtype.createDataType(typeItem)
        self.assertEqual(dt.name, "void80")
        self.assertEqual(dt.kind, "V")
        self.assertEqual(len(dt.fields), 3)
        self.assertEqual(typeSize, 10)
        self.assertEqual(typeSize, hdf5dtype.getDtypeItemSize(dt))
        self.assertFalse(isVlen(dt))

    def testCreateArrayType(self):
        typeItem = {"class": "H5T_ARRAY", "base": "H5T_STD_I64LE", "dims": (3, 5)}
        typeSize = hdf5dtype.getItemSize(typeItem)
        dt = hdf5dtype.createDataType(typeItem)
        self.assertEqual(dt.name, "void960")
        self.assertEqual(dt.kind, "V")
        self.assertEqual(dt.base.kind, "i")
        self.assertEqual(typeSize, 120)
        self.assertEqual(typeSize, hdf5dtype.getDtypeItemSize(dt))
        self.assertFalse(isVlen(dt))

    def testCreateCompoundArrayVlenType(self):
        typeItem = {
            "fields": [
                {"type": {"class": "H5T_INTEGER", "base": "H5T_STD_U64BE"}, "name": "VALUE"},
                {"type": {"class": "H5T_FLOAT", "base": "H5T_IEEE_F64BE"}, "name": "VALUE2"},
                {"type": {"class": "H5T_ARRAY", "dims": [8],
                          "base": {
                              "class": "H5T_STRING",
                              "charSet": "H5T_CSET_ASCII",
                              "strPad": "H5T_STR_NULLTERM",
                              "length": "H5T_VARIABLE"
                            }  # noqa: E126
                          },
                 "name": "VALUE3"}
                ],  # noqa: E123
            "class": "H5T_COMPOUND"
        }
        typeSize = hdf5dtype.getItemSize(typeItem)
        dt = hdf5dtype.createDataType(typeItem)
        self.assertEqual(dt.name, "void640")
        self.assertEqual(dt.kind, "V")
        self.assertEqual(typeSize, "H5T_VARIABLE")
        self.assertEqual(typeSize, hdf5dtype.getDtypeItemSize(dt))
        self.assertTrue(isVlen(dt))
        dt_arr = dt["VALUE3"]
        self.assertEqual(dt_arr.kind, "V")
        self.assertEqual(dt_arr.shape, (8,))
        self.assertEqual(dt_arr.metadata, None)

    def testCreateArrayIntegerType(self):
        typeItem = {"class": "H5T_INTEGER", "base": "H5T_STD_I64LE", "dims": (3, 5)}

        try:
            hdf5dtype.createDataType(typeItem)
            self.assertTrue(False)  # expected exception - dims used with non-array type
        except TypeError:
            pass  # should get exception

    def testCreateVlenObjRefType(self):
        typeItem = {
            "class": "H5T_VLEN",
            "base": {"class": "H5T_REFERENCE", "base": "H5T_STD_REF_OBJ"},
        }
        dt = hdf5dtype.createDataType(typeItem)
        self.assertEqual(dt.name, "object")
        self.assertEqual(dt.kind, "O")
        self.assertTrue(check_dtype(ref=dt) is None)
        self.assertTrue(isVlen(dt))
        dt_base = check_dtype(vlen=dt)
        self.assertTrue(dt_base is not None)
        self.assertTrue(check_dtype(ref=dt_base) is Reference)

    def testCreateCompoundArrayType(self):
        typeItem = {
            "class": "H5T_COMPOUND",
            "fields": [
                {"type": {"base": "H5T_STD_I8LE", "class": "H5T_INTEGER"}, "name": "a"},
                {
                    "type": {
                        "dims": [10],
                        "base": {
                            "length": 1,
                            "charSet": "H5T_CSET_ASCII",
                            "class": "H5T_STRING",
                            "strPad": "H5T_STR_NULLPAD",
                        },
                        "class": "H5T_ARRAY",
                    },
                    "name": "b",
                },
            ],
        }
        typeSize = hdf5dtype.getItemSize(typeItem)
        dt = hdf5dtype.createDataType(typeItem)
        self.assertEqual(len(dt.fields), 2)
        self.assertTrue("a" in dt.fields.keys())
        self.assertTrue("b" in dt.fields.keys())
        self.assertEqual(typeSize, 11)
        self.assertEqual(typeSize, hdf5dtype.getDtypeItemSize(dt))
        self.assertFalse(isVlen(dt))

    def testCompoundArrayType(self):
        typeItem = {
            "class": "H5T_COMPOUND",
            "fields": [
                {
                    "type": {"class": "H5T_INTEGER", "base": "H5T_STD_U64BE"},
                    "name": "VALUE1",
                },
                {
                    "type": {"class": "H5T_FLOAT", "base": "H5T_IEEE_F64BE"},
                    "name": "VALUE2",
                },
                {
                    "type": {
                        "class": "H5T_ARRAY",
                        "dims": [2],
                        "base": {
                            "class": "H5T_STRING",
                            "charSet": "H5T_CSET_ASCII",
                            "strPad": "H5T_STR_NULLTERM",
                            "length": "H5T_VARIABLE",
                        },
                    },
                    "name": "VALUE3",
                },
            ],
        }
        dt = hdf5dtype.createDataType(typeItem)
        typeSize = hdf5dtype.getItemSize(typeItem)
        self.assertEqual(typeSize, "H5T_VARIABLE")
        self.assertTrue(isVlen(dt))
        self.assertEqual(len(dt), 3)
        self.assertTrue("VALUE1" in dt.fields.keys())
        self.assertTrue("VALUE2" in dt.fields.keys())
        self.assertTrue("VALUE3" in dt.fields.keys())
        self.assertEqual(typeSize, hdf5dtype.getDtypeItemSize(dt))

    def testGetDtypeItemSizeRegionReference(self):
        # Regression test: getDtypeItemSize() used to only recognize the
        # "vlen" metadata key, so a RegionReference-tagged dtype (metadata
        # key "ref", not "vlen") fell through to the plain dtype.itemsize
        # branch, reporting a fixed 8 bytes instead of "H5T_VARIABLE".  HSDS
        # uses this function to size a dataset's chunk layout, so a region
        # reference dataset's stored value got silently truncated to the
        # object-pointer itemsize rather than the actual (much larger)
        # serialized RegionReference.tobytes() blob.
        dt = special_dtype(ref=RegionReference)
        self.assertEqual(hdf5dtype.getDtypeItemSize(dt), "H5T_VARIABLE")

        # a plain object Reference is a fixed-format id string, not a
        # length-prefixed blob, so it's unaffected by this fix
        dt_ref = special_dtype(ref=Reference)
        self.assertEqual(hdf5dtype.getDtypeItemSize(dt_ref), dt_ref.itemsize)

        # also check a RegionReference nested in a compound field
        dt_compound = np.dtype([("a", "i4"), ("b", dt)])
        self.assertEqual(hdf5dtype.getDtypeItemSize(dt_compound), "H5T_VARIABLE")

    def testFindItemType(self):
        # simple scalar python types
        self.assertEqual(hdf5dtype.find_item_type(5), int)
        self.assertEqual(hdf5dtype.find_item_type(5.0), float)
        self.assertEqual(hdf5dtype.find_item_type("abc"), str)
        self.assertEqual(hdf5dtype.find_item_type(b"abc"), bytes)
        self.assertEqual(hdf5dtype.find_item_type(True), bool)

        # lists/tuples - uniform item type (possibly nested)
        self.assertEqual(hdf5dtype.find_item_type([1, 2, 3]), int)
        self.assertEqual(hdf5dtype.find_item_type([[1, 2], [3, 4]]), int)
        self.assertEqual(hdf5dtype.find_item_type((1, 2, 3)), int)

        # mixed item types -> None
        self.assertIsNone(hdf5dtype.find_item_type([1, "a"]))

        # empty collection -> no common type -> None
        self.assertIsNone(hdf5dtype.find_item_type([]))

        # numpy array with a specific (non-object) dtype -> None
        arr_int = np.array([1, 2, 3])
        self.assertIsNone(hdf5dtype.find_item_type(arr_int))

        # numpy object arrays are treated like plain python collections
        arr_obj_str = np.array(["a", "b"], dtype=object)
        self.assertEqual(hdf5dtype.find_item_type(arr_obj_str), str)
        arr_obj_int = np.array([1, 2, 3], dtype=object)
        self.assertEqual(hdf5dtype.find_item_type(arr_obj_int), int)

        # a numpy array using the h5py vlen extension is not treated as
        # a plain object collection - falls into the "return None" branch
        dt_vlen = special_dtype(vlen=str)
        arr_vlen = np.array(["a", "b"], dtype=dt_vlen)
        self.assertIsNone(hdf5dtype.find_item_type(arr_vlen))

    def testGuessDtype(self):
        # non-str/bytes item types -> None (left to array constructor)
        self.assertIsNone(hdf5dtype.guess_dtype(5))
        self.assertIsNone(hdf5dtype.guess_dtype([1, 2, 3]))
        self.assertIsNone(hdf5dtype.guess_dtype([]))

        # bytes items -> vlen bytes special dtype
        dt_bytes = hdf5dtype.guess_dtype([b"a", b"b"])
        self.assertEqual(dt_bytes.kind, "O")
        self.assertEqual(check_dtype(vlen=dt_bytes), bytes)

        # str items -> vlen str special dtype
        dt_str = hdf5dtype.guess_dtype(["a", "b"])
        self.assertEqual(dt_str.kind, "O")
        self.assertEqual(check_dtype(vlen=dt_str), str)

        # nested lists of str also resolve to a uniform item type
        dt_nested = hdf5dtype.guess_dtype([["a", "b"], ["c", "d"]])
        self.assertEqual(check_dtype(vlen=dt_nested), str)

    def testIsFloat16Dtype(self):
        self.assertFalse(hdf5dtype.is_float16_dtype(None))
        self.assertTrue(hdf5dtype.is_float16_dtype(np.float16))
        self.assertTrue(hdf5dtype.is_float16_dtype("f2"))  # normalizes strings
        self.assertTrue(hdf5dtype.is_float16_dtype(np.dtype("<f2")))
        self.assertFalse(hdf5dtype.is_float16_dtype(np.float32))
        # same itemsize as float16, but different kind
        self.assertFalse(hdf5dtype.is_float16_dtype(np.int16))

    def testValidateTypeItem(self):
        # valid type item - should pass without raising
        typeItem = {"class": "H5T_INTEGER", "base": "H5T_STD_I32LE"}
        hdf5dtype.validateTypeItem(typeItem)

        # valid predefined type name string
        hdf5dtype.validateTypeItem("H5T_STD_I16LE")

        # missing 'base' for H5T_INTEGER -> KeyError
        with self.assertRaises(KeyError):
            hdf5dtype.validateTypeItem({"class": "H5T_INTEGER"})

        # missing 'class' key entirely -> KeyError
        with self.assertRaises(KeyError):
            hdf5dtype.validateTypeItem({"base": "H5T_STD_I32LE"})

        # unrecognized predefined type name -> TypeError
        with self.assertRaises(TypeError):
            hdf5dtype.validateTypeItem("foobar")

        # not a dict or string -> TypeError
        with self.assertRaises(TypeError):
            hdf5dtype.validateTypeItem(42)

    def testGetSubType(self):
        dt_compound = np.dtype([("a", "<i4"), ("b", "<f8"), ("c", "S10")])

        sub = hdf5dtype.getSubType(dt_compound, ["a", "c"])
        self.assertEqual(sub.names, ("a", "c"))
        self.assertEqual(sub["a"], np.dtype("<i4"))
        self.assertEqual(sub["c"], np.dtype("S10"))

        # a single field name given as a (non-list) string is accepted
        sub_single = hdf5dtype.getSubType(dt_compound, "b")
        self.assertEqual(sub_single.names, ("b",))
        self.assertEqual(sub_single["b"], np.dtype("<f8"))

        # parent must be a compound type
        with self.assertRaises(TypeError):
            hdf5dtype.getSubType(np.dtype("<i4"), ["a"])

        # null/empty field specification
        with self.assertRaises(TypeError):
            hdf5dtype.getSubType(dt_compound, None)
        with self.assertRaises(TypeError):
            hdf5dtype.getSubType(dt_compound, [])

        # requested field not present in the parent type
        with self.assertRaises(TypeError):
            hdf5dtype.getSubType(dt_compound, ["z"])


class ReferenceTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ReferenceTest, self).__init__(*args, **kwargs)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    def testBindSchema2Id(self):
        root_id = createObjId("groups")
        dset_id = createObjId("datasets", root_id=root_id)
        ref = Reference("datasets/" + dset_id)
        self.assertEqual(ref.id, dset_id)

    def testBindSchema1Uuid(self):
        # a plain (schema 1 style, no "d-" prefix) uuid, as used in
        # data/json/regionref_dset.json / regionref_attr.json
        ref = Reference("datasets/a296b77d-83f8-11e5-815f-3c15c2da029e")
        self.assertEqual(ref.id, "d-a296b77d-83f8-11e5-815f-3c15c2da029e")

    def testBindAlreadyHashTagId(self):
        # a bare hashtag id (no "/" at all) passes straight through
        ref = Reference("d-a296b77d-83f8-11e5-815f-3c15c2da029e")
        self.assertEqual(ref.id, "d-a296b77d-83f8-11e5-815f-3c15c2da029e")

    def testBindNull(self):
        ref = Reference(None)
        self.assertIsNone(ref.id)

    def testTolistDataset(self):
        root_id = createObjId("groups")
        dset_id = createObjId("datasets", root_id=root_id)
        ref = Reference("datasets/" + dset_id)
        self.assertEqual(ref.tolist(), ["datasets/" + dset_id])

    def testTolistGroup(self):
        root_id = createObjId("groups")
        group_id = createObjId("groups", root_id=root_id)
        ref = Reference("groups/" + group_id)
        self.assertEqual(ref.tolist(), ["groups/" + group_id])

    def testTolistDatatype(self):
        root_id = createObjId("groups")
        dtype_id = createObjId("datatypes", root_id=root_id)
        ref = Reference("datatypes/" + dtype_id)
        self.assertEqual(ref.tolist(), ["datatypes/" + dtype_id])

    def testTolistUnboundRaises(self):
        # an unbound reference has self._id is None (not a string)
        ref = Reference(None)
        with self.assertRaises(TypeError):
            ref.tolist()

    def testTolistEmptyStringId(self):
        # simulate a bound reference whose id happens to be an empty string
        root_id = createObjId("groups")
        dset_id = createObjId("datasets", root_id=root_id)
        ref = Reference("datasets/" + dset_id)
        ref._id = ""
        self.assertEqual(ref.tolist(), [("",)])

    def testTolistUnexpectedIdTypeRaises(self):
        root_id = createObjId("groups")
        dset_id = createObjId("datasets", root_id=root_id)
        ref = Reference("datasets/" + dset_id)
        ref._id = "x-1234"  # unrecognized collection prefix
        with self.assertRaises(TypeError):
            ref.tolist()


class RegionReferenceTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(RegionReferenceTest, self).__init__(*args, **kwargs)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    def _dset_id(self):
        root_id = createObjId("groups")
        return createObjId("datasets", root_id=root_id)

    def testNull(self):
        ref = RegionReference()
        self.assertIsNone(ref.id)
        self.assertIsNone(ref.selection_bytes)

    def testBindWithSelectionInstance(self):
        dset_id = self._dset_id()
        sel = selections.select((100,), slice(2, 10))
        ref = RegionReference()
        ref.bind("datasets/" + dset_id, sel)
        self.assertIsNotNone(ref.id)
        self.assertIsInstance(ref.selection_bytes, bytes)
        self.assertEqual(selections.Selection.frombytes(ref.selection_bytes), sel)

    def testBindWithSerializedBytes(self):
        dset_id = self._dset_id()
        sel = selections.select((100,), slice(2, 10))
        data = sel.tobytes()
        ref = RegionReference()
        ref.bind(dset_id, data)
        self.assertEqual(ref.selection_bytes, bytes(data))
        self.assertEqual(selections.Selection.frombytes(ref.selection_bytes), sel)

    def testBindWithoutCollectionPrefix(self):
        dset_id = self._dset_id()
        sel = selections.select((100,), slice(2, 10))
        ref_prefixed = RegionReference()
        ref_prefixed.bind("datasets/" + dset_id, sel)
        ref_bare = RegionReference()
        ref_bare.bind(dset_id, sel)
        self.assertEqual(ref_prefixed.id, ref_bare.id)

    def testBindSchema1Uuid(self):
        # a plain (schema 1 style, no "d-" prefix) uuid, as used in
        # data/json/regionref_dset.json / regionref_attr.json
        sel = selections.select((100,), slice(2, 10))
        ref = RegionReference()
        ref.bind("datasets/a296b77d-83f8-11e5-815f-3c15c2da029e", sel)
        self.assertEqual(ref.id, "d-a296b77d-83f8-11e5-815f-3c15c2da029e")

    def testBindWithDatasetObject(self):
        dset_id = self._dset_id()
        sel = selections.select((100,), slice(2, 10))

        class FakeDatasetObj:
            def __init__(self, _id):
                self._id = _id

        ref_via_obj = RegionReference()
        ref_via_obj.bind(FakeDatasetObj(dset_id), sel)
        ref_via_str = RegionReference()
        ref_via_str.bind(dset_id, sel)
        self.assertEqual(ref_via_obj.id, ref_via_str.id)

    def testConstructorBinds(self):
        dset_id = self._dset_id()
        sel = selections.select((100,), slice(2, 10))
        ref = RegionReference(dset_id, sel)
        self.assertIsNotNone(ref.id)
        self.assertEqual(selections.Selection.frombytes(ref.selection_bytes), sel)

    def testBindWrongCollectionRaises(self):
        root_id = createObjId("groups")
        group_id = createObjId("groups", root_id=root_id)
        sel = selections.select((100,), slice(2, 10))
        ref = RegionReference()
        with self.assertRaises(TypeError):
            ref.bind("groups/" + group_id, sel)

    def testBindInvalidSelectionRaises(self):
        dset_id = self._dset_id()
        ref = RegionReference()
        with self.assertRaises(TypeError):
            ref.bind(dset_id, 42)

    def testTobytesFrombytesRoundtrip(self):
        dset_id = self._dset_id()
        sel = selections.select((100,), slice(2, 10))
        ref = RegionReference(dset_id, sel)

        raw = ref.tobytes()
        self.assertIsInstance(raw, bytes)

        ref2 = RegionReference.frombytes(raw)
        self.assertEqual(ref2.id, ref.id)
        self.assertEqual(selections.Selection.frombytes(ref2.selection_bytes), sel)

    def testTobytesFrombytesNullRef(self):
        ref = RegionReference()
        raw = ref.tobytes()
        ref2 = RegionReference.frombytes(raw)
        self.assertIsNone(ref2.id)
        self.assertIsNone(ref2.selection_bytes)

    def testTobytesSurvivesFixedWidthStoragePadding(self):
        # a RegionReference stored as a raw value in a numpy fixed-length
        # byte-string ("S<n>") array element must survive being embedded in
        # an oversized field (which numpy NUL-pads) and read back - this is
        # exactly the storage strategy used for H5T_REFERENCE dataset values.
        dset_id = self._dset_id()
        sel = selections.select((100,), slice(2, 10))
        ref = RegionReference(dset_id, sel)
        raw = ref.tobytes()

        dt = np.dtype(f"S{len(raw) + 32}")  # deliberately oversized
        arr = np.zeros((1,), dtype=dt)
        arr[0] = raw
        stored = bytes(arr[0])

        ref2 = RegionReference.frombytes(stored)
        self.assertEqual(ref2.id, ref.id)
        self.assertEqual(selections.Selection.frombytes(ref2.selection_bytes), sel)

    def testToJson(self):
        # matches the format used in data/json/regionref_dset.json /
        # regionref_attr.json: {"id": <bare uuid>, "select_type": ..., "selection": [...]}
        dset_id = self._dset_id()
        sel = selections.select((3, 16), ([0, 2, 1, 2], [1, 11, 0, 4]))
        ref = RegionReference("datasets/" + dset_id, sel)

        d = ref.to_json()
        self.assertEqual(d["id"], getUuidFromId(dset_id))
        self.assertEqual(d["select_type"], "H5S_SEL_POINTS")
        self.assertEqual(d["selection"], [[0, 1], [2, 11], [1, 0], [2, 4]])

    def testToJsonHyperslab(self):
        dset_id = self._dset_id()
        sel = selections.select((3, 16), (slice(0, 2), slice(0, 4)))
        ref = RegionReference(dset_id, sel)

        d = ref.to_json()
        self.assertEqual(d["id"], getUuidFromId(dset_id))
        self.assertEqual(d["select_type"], "H5S_SEL_HYPERSLABS")
        self.assertEqual(d["selection"], [[[0, 0], [1, 3]]])

    def testToJsonNullRefRaises(self):
        ref = RegionReference()
        with self.assertRaises(ValueError):
            ref.to_json()

    def testToJsonNoSelection(self):
        # e.g. a region reference read from an actual HDF5 file, where only
        # the target dataset's identity can be recovered, not its selection
        dset_id = self._dset_id()
        ref = RegionReference(dset_id)
        d = ref.to_json()
        self.assertEqual(d, {"id": getUuidFromId(dset_id)})

        ref2 = RegionReference.from_json(d)
        self.assertEqual(ref2.id, ref.id)
        self.assertIsNone(ref2.selection_bytes)

    def testFromJsonRoundTrip(self):
        dset_id = self._dset_id()
        sel = selections.select((3, 16), ([0, 2, 1, 2], [1, 11, 0, 4]))
        ref = RegionReference("datasets/" + dset_id, sel)
        d = ref.to_json()

        ref2 = RegionReference.from_json(d)
        self.assertEqual(ref2.id, ref.id)
        sel2 = selections.Selection.frombytes(ref2.selection_bytes)
        self.assertEqual(sel2.select_type, sel.select_type)
        self.assertEqual(sel2.to_region_json(), sel.to_region_json())

    def testFromJsonBareSchema1Uuid(self):
        # matches data/json/regionref_dset.json / regionref_attr.json, which
        # use plain (schema 1 style, no "d-" prefix) uuids for "id"
        d = {
            "id": "a296b77d-83f8-11e5-815f-3c15c2da029e",
            "select_type": "H5S_SEL_POINTS",
            "selection": [[0, 1], [2, 11], [1, 0], [2, 4]],
        }
        ref = RegionReference.from_json(d)
        self.assertEqual(ref.id, "d-a296b77d-83f8-11e5-815f-3c15c2da029e")
        self.assertEqual(ref.to_json(), d)

    def testFromJsonNull(self):
        ref = RegionReference.from_json(None)
        self.assertIsNone(ref.id)
        self.assertIsNone(ref.selection_bytes)

    def testFromJsonMissingIdRaises(self):
        with self.assertRaises(KeyError):
            RegionReference.from_json({"select_type": "H5S_SEL_POINTS", "selection": [[0]]})

    def testToJsonFancySelectionFallsBackToSelectionDict(self):
        # H5S_SEL_FANCY has no points/hyperslab equivalent, so to_json()
        # embeds the fully general Selection.to_dict() instead of raising
        dset_id = self._dset_id()
        sel = selections.select((6, 10), (slice(0, 4), [1, 3, 7]))
        ref = RegionReference("datasets/" + dset_id, sel)

        d = ref.to_json()
        self.assertEqual(d["id"], getUuidFromId(dset_id))
        self.assertNotIn("select_type", d)
        self.assertNotIn("selection", d)
        self.assertEqual(d["selection_dict"], sel.to_dict())

    def testFromJsonSelectionDictRoundTrip(self):
        dset_id = self._dset_id()
        sel = selections.select((6, 10), (slice(0, 4), [1, 3, 7]))
        ref = RegionReference("datasets/" + dset_id, sel)
        d = ref.to_json()

        ref2 = RegionReference.from_json(d)
        self.assertEqual(ref2.id, ref.id)
        sel2 = selections.Selection.frombytes(ref2.selection_bytes)
        self.assertEqual(sel2, sel)
        self.assertEqual(ref2.to_json(), d)

    def testToJsonSteppedHyperslabFallsBackToSelectionDict(self):
        # stepped hyperslabs also have no region-reference equivalent
        dset_id = self._dset_id()
        sel = selections.select((10,), slice(0, 10, 2))
        ref = RegionReference("datasets/" + dset_id, sel)

        d = ref.to_json()
        self.assertIn("selection_dict", d)
        ref2 = RegionReference.from_json(d)
        sel2 = selections.Selection.frombytes(ref2.selection_bytes)
        self.assertEqual(sel2, sel)


if __name__ == "__main__":
    # setup test files

    unittest.main()
