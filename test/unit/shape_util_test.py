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

from h5json.shape_util import getShapeClass, getShapeDims, getNumElements, getRank, getShapeJson
from h5json.shape_util import isNullSpace, isScalar, getDataSize, isExtensible, getMaxDims


class ShapeUtilTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ShapeUtilTest, self).__init__(*args, **kwargs)
        # main
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.WARNING)

    def testGetShape(self):

        null_shape = getShapeJson("H5S_NULL")
        self.assertTrue("class" in null_shape)
        self.assertEqual(null_shape["class"], "H5S_NULL")
        self.assertFalse("dims" in null_shape)
        self.assertFalse("maxdims" in null_shape)

        null_shape = getShapeJson(None)
        self.assertTrue("class" in null_shape)
        self.assertEqual(null_shape["class"], "H5S_NULL")
        self.assertFalse("dims" in null_shape)
        self.assertFalse("maxdims" in null_shape)

        scalar_shape = getShapeJson(())
        self.assertTrue("class" in scalar_shape)
        self.assertEqual(scalar_shape["class"], "H5S_SCALAR")
        self.assertTrue("dims" not in scalar_shape)
        self.assertFalse("maxdims" in scalar_shape)

        simple_shape = getShapeJson(42)
        self.assertTrue("class" in simple_shape)
        self.assertEqual(simple_shape["class"], "H5S_SIMPLE")
        self.assertTrue("dims" in simple_shape)
        self.assertEqual(simple_shape["dims"], (42, ))
        self.assertFalse("maxdims" in simple_shape)

        simple_shape = getShapeJson((42, ))
        self.assertTrue("class" in simple_shape)
        self.assertEqual(simple_shape["class"], "H5S_SIMPLE")
        self.assertTrue("dims" in simple_shape)
        self.assertEqual(simple_shape["dims"], (42, ))
        self.assertFalse("maxdims" in simple_shape)

        extendable_shape = getShapeJson((4, 5), maxdims=("H5S_UNLIMITED", 10))
        self.assertTrue("class" in extendable_shape)
        self.assertEqual(extendable_shape["class"], "H5S_SIMPLE")
        self.assertTrue("dims" in extendable_shape)
        self.assertEqual(extendable_shape["dims"], (4, 5))
        self.assertTrue("maxdims" in extendable_shape)
        self.assertTrue(extendable_shape["maxdims"], ("H5S_UNLIMITED", 10))

        extendable_shape = getShapeJson((4, 5), maxdims=(None, 10))
        self.assertTrue("class" in extendable_shape)
        self.assertEqual(extendable_shape["class"], "H5S_SIMPLE")
        self.assertTrue("dims" in extendable_shape)
        self.assertEqual(extendable_shape["dims"], (4, 5))
        self.assertTrue("maxdims" in extendable_shape)
        self.assertTrue(extendable_shape["maxdims"], ("H5S_UNLIMITED", 10))

        extendable_shape = getShapeJson((4, 5), maxdims=(0, 10))
        self.assertTrue("class" in extendable_shape)
        self.assertEqual(extendable_shape["class"], "H5S_SIMPLE")
        self.assertTrue("dims" in extendable_shape)
        self.assertEqual(extendable_shape["dims"], (4, 5))
        self.assertTrue("maxdims" in extendable_shape)
        self.assertTrue(extendable_shape["maxdims"], ("H5S_UNLIMITED", 10))

    def testSimple(self):

        type_json = {
            "base": "H5T_STD_I32BE",
            "class": "H5T_INTEGER"
        }
        vstr_json = {
            "charSet": "H5T_CSET_ASCII",
            "class": "H5T_STRING",
            "length": "H5T_VARIABLE",
            "strPad": "H5T_STR_NULLTERM"
        }
        null_shape_json = {"class": "H5S_NULL"}
        null_shape_obj = {"type": type_json, "shape": null_shape_json}
        scalar_shape_json = {"class": "H5S_SCALAR"}
        scalar_shape_obj = {"type": type_json, "shape": scalar_shape_json}
        vstr_scalar_shape_obj = {"type": vstr_json, "shape": scalar_shape_json}

        simple_shape_json = {"class": "H5S_SIMPLE", "dims": [5, 7]}
        simple_shape_obj = {"type": type_json, "shape": simple_shape_json}
        vstr_simple_shape_obj = {"type": vstr_json, "shape": simple_shape_json}
        resizable_shape_obj = {'class': 'H5S_SIMPLE', 'dims': [10], 'maxdims': [20]}
        unlimited_shape_obj = {'class': 'H5S_SIMPLE', 'dims': [0, 20], 'maxdims': ["H5S_UNLIMITED", 40]}

        self.assertEqual(getShapeClass(null_shape_json), "H5S_NULL")
        self.assertEqual(getShapeClass(null_shape_obj), "H5S_NULL")
        self.assertEqual(getShapeClass(scalar_shape_json), "H5S_SCALAR")
        self.assertEqual(getShapeClass(scalar_shape_obj), "H5S_SCALAR")
        self.assertEqual(getShapeClass(vstr_scalar_shape_obj), "H5S_SCALAR")
        self.assertEqual(getShapeClass(simple_shape_json), "H5S_SIMPLE")
        self.assertEqual(getShapeClass(simple_shape_obj), "H5S_SIMPLE")
        self.assertEqual(getShapeClass(vstr_simple_shape_obj), "H5S_SIMPLE")
        self.assertEqual(getShapeClass(resizable_shape_obj), "H5S_SIMPLE")
        self.assertEqual(getShapeClass(unlimited_shape_obj), "H5S_SIMPLE")

        self.assertEqual(getShapeDims(null_shape_json), None)
        self.assertEqual(getShapeDims(null_shape_obj), None)
        self.assertEqual(getShapeDims(scalar_shape_json), ())
        self.assertEqual(getShapeDims(scalar_shape_obj), ())
        self.assertEqual(getShapeDims(vstr_scalar_shape_obj), ())
        self.assertEqual(getShapeDims(simple_shape_json), (5, 7))
        self.assertEqual(getShapeDims(simple_shape_obj), (5, 7))
        self.assertEqual(getShapeDims(vstr_simple_shape_obj), (5, 7))
        self.assertEqual(getShapeDims(12), (12,))
        self.assertEqual(getShapeDims(resizable_shape_obj), (10,))
        self.assertEqual(getShapeDims(unlimited_shape_obj), (0, 20))

        self.assertEqual(getMaxDims(null_shape_json), None)
        self.assertEqual(getMaxDims(null_shape_obj), None)
        self.assertEqual(getMaxDims(scalar_shape_json), ())
        self.assertEqual(getMaxDims(scalar_shape_obj), ())
        self.assertEqual(getMaxDims(vstr_scalar_shape_obj), ())
        self.assertEqual(getMaxDims(simple_shape_json), (5, 7))
        self.assertEqual(getMaxDims(simple_shape_obj), (5, 7))
        self.assertEqual(getMaxDims(vstr_simple_shape_obj), (5, 7))
        self.assertEqual(getMaxDims(resizable_shape_obj), (20,))
        self.assertEqual(getMaxDims(unlimited_shape_obj), ("H5S_UNLIMITED", 40))

        self.assertEqual(getRank(null_shape_json), 0)
        self.assertEqual(getRank(null_shape_obj), 0)
        self.assertEqual(getRank(scalar_shape_json), 0)
        self.assertEqual(getRank(scalar_shape_obj), 0)
        self.assertEqual(getRank(vstr_scalar_shape_obj), 0)
        self.assertEqual(getRank(simple_shape_json), 2)
        self.assertEqual(getRank(simple_shape_obj), 2)
        self.assertEqual(getRank(vstr_simple_shape_obj), 2)
        self.assertEqual(getRank(resizable_shape_obj), 1)
        self.assertEqual(getRank(unlimited_shape_obj), 2)

        self.assertEqual(getNumElements(null_shape_json), 0)
        self.assertEqual(getNumElements(null_shape_obj), 0)
        self.assertEqual(getNumElements(scalar_shape_json), 1)
        self.assertEqual(getNumElements(scalar_shape_obj), 1)
        self.assertEqual(getNumElements(vstr_scalar_shape_obj), 1)
        self.assertEqual(getNumElements(simple_shape_json), 35)
        self.assertEqual(getNumElements(simple_shape_obj), 35)
        self.assertEqual(getNumElements(vstr_simple_shape_obj), 35)
        self.assertEqual(getNumElements(resizable_shape_obj), 10)
        self.assertEqual(getNumElements(unlimited_shape_obj), 0)
        self.assertEqual(getNumElements(()), 1)
        self.assertEqual(getNumElements([1, 2, 3]), 6)

        self.assertEqual(isNullSpace(null_shape_json), True)
        self.assertEqual(isNullSpace(null_shape_obj), True)
        self.assertEqual(isNullSpace(scalar_shape_json), False)
        self.assertEqual(isNullSpace(scalar_shape_obj), False)
        self.assertEqual(isNullSpace(vstr_scalar_shape_obj), False)
        self.assertEqual(isNullSpace(simple_shape_json), False)
        self.assertEqual(isNullSpace(simple_shape_obj), False)
        self.assertEqual(isNullSpace(vstr_simple_shape_obj), False)
        self.assertEqual(isNullSpace(resizable_shape_obj), False)
        self.assertEqual(isNullSpace(unlimited_shape_obj), False)

        self.assertEqual(isScalar(null_shape_json), False)
        self.assertEqual(isScalar(null_shape_obj), False)
        self.assertEqual(isScalar(scalar_shape_json), True)
        self.assertEqual(isScalar(scalar_shape_obj), True)
        self.assertEqual(isScalar(vstr_scalar_shape_obj), True)
        self.assertEqual(isScalar(simple_shape_json), False)
        self.assertEqual(isScalar(simple_shape_obj), False)
        self.assertEqual(isScalar(vstr_simple_shape_obj), False)
        self.assertEqual(isScalar(resizable_shape_obj), False)
        self.assertEqual(isScalar(unlimited_shape_obj), False)

        self.assertEqual(getDataSize(null_shape_json, 4), 0)
        self.assertEqual(getDataSize(null_shape_obj, 4), 0)
        self.assertEqual(getDataSize(scalar_shape_json, 4), 4)
        self.assertEqual(getDataSize(scalar_shape_obj, 4), 4)
        self.assertEqual(getDataSize(vstr_scalar_shape_obj, 4), 4)
        self.assertEqual(getDataSize(simple_shape_json, 4), 140)
        self.assertEqual(getDataSize(simple_shape_obj, 4), 140)
        self.assertEqual(getDataSize(vstr_simple_shape_obj, 4), 140)
        self.assertEqual(getDataSize(resizable_shape_obj, 4), 40)
        self.assertEqual(getDataSize(unlimited_shape_obj, 4), 0)

        self.assertEqual(getDataSize((), 4), 4)
        self.assertEqual(getDataSize([1, 2, 3], 4), 24)

        self.assertEqual(isScalar(null_shape_json), False)
        self.assertEqual(isScalar(null_shape_obj), False)
        self.assertEqual(isScalar(scalar_shape_json), True)
        self.assertEqual(isScalar(scalar_shape_obj), True)
        self.assertEqual(isScalar(vstr_scalar_shape_obj), True)
        self.assertEqual(isScalar(simple_shape_json), False)
        self.assertEqual(isScalar(simple_shape_obj), False)
        self.assertEqual(isScalar(vstr_simple_shape_obj), False)
        self.assertEqual(isScalar(resizable_shape_obj), False)
        self.assertEqual(isScalar(unlimited_shape_obj), False)

        self.assertEqual(isExtensible(null_shape_json), False)
        self.assertEqual(isExtensible(null_shape_obj), False)
        self.assertEqual(isExtensible(scalar_shape_json), False)
        self.assertEqual(isExtensible(scalar_shape_obj), False)
        self.assertEqual(isExtensible(vstr_scalar_shape_obj), False)
        self.assertEqual(isExtensible(simple_shape_json), False)
        self.assertEqual(isExtensible(simple_shape_obj), False)
        self.assertEqual(isExtensible(vstr_simple_shape_obj), False)
        self.assertEqual(isExtensible(resizable_shape_obj), True)
        self.assertEqual(isExtensible(unlimited_shape_obj), True)


if __name__ == "__main__":
    # setup test files

    unittest.main()
