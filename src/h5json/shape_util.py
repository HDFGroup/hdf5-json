##############################################################################
# Copyright by The HDF Group.                                                #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HSDS (HDF5 REST Server) Service, Libraries and        #
# Utilities.  The full HDF5 REST Server copyright notice, including          #
# terms governing use, modification, and redistribution, is contained in     #
# the file COPYING, which can be found at the root of the source code        #
# distribution tree.  If you do not have access to this file, you may        #
# request a copy from help@hdfgroup.org.                                     #
##############################################################################

import numpy as np


def getShapeClass(shape):
    """ Return shape class of the given data shape """

    if not isinstance(shape, dict):
        raise TypeError("expected dict object")

    if shape.get("class") in ("H5S_NULL", "H5S_SCALAR", "H5S_SIMPLE"):
        # this is a shape_json obj
        shape_json = shape
    elif "shape" in shape:
        # dataset or attribute json
        shape_json = shape["shape"]
    else:
        raise ValueError(f"Unknown shape: {shape}")

    if "class" not in shape_json:
        raise KeyError("expected 'class' key for data shape")\

    return shape_json["class"]


def getShapeDims(shape):
    """
    Get dims from a given shape json.  Return [1,] for Scalar datasets,
    None for null data spaces
    """
    dims = None
    if isinstance(shape, int):
        dims = (shape, )
    elif isinstance(shape, list):
        dims = tuple(shape)
    elif isinstance(shape, tuple):
        dims = shape  # can use as is
    elif isinstance(shape, str):
        # only valid string value is H5S_NULL
        if shape != "H5S_NULL":
            raise ValueError("Invalid value for shape")
        dims = None
    elif isinstance(shape, dict):
        if shape.get("class") in ("H5S_NULL", "H5S_SCALAR", "H5S_SIMPLE"):
            # this is a shape_json obj
            shape_json = shape
        elif "shape" in shape:
            # dataset or attribute json
            shape_json = shape["shape"]
        else:
            raise ValueError(f"Unknown shape: {shape}")

        if "class" not in shape_json:
            raise ValueError("'class' key not found in shape")
        shape_class = shape_json["class"]
        if shape_class == "H5S_NULL":
            dims = None
        elif shape_class == "H5S_SCALAR":
            dims = ()
        elif shape_class == "H5S_SIMPLE":
            if "dims" not in shape_json:
                raise ValueError("'dims' key expected for shape")
            dims = tuple(shape_json["dims"])
        else:
            raise ValueError(f"Unknown shape: {shape_json}")
    else:
        raise ValueError(f"Unexpected shape class: {type(shape)}")
    return dims


def getNumElements(obj_json):
    """ return the number of elements defined by the dataset's shape
        returns None for null shape, 1 for scalar shape, and product of
        extents otherwise """

    dims = getShapeDims(obj_json)
    if dims is None:
        return 0
    else:
        return int(np.prod(dims))


def getRank(shape):
    """ Return rank of given data shape """

    dims = getShapeDims(shape)
    if dims is None:
        return 0
    else:
        return len(dims)


def isNullSpace(shape):
    """Return true if this dataset is a null data space"""

    shape_class = getShapeClass(shape)
    if shape_class == "H5S_NULL":
        return True
    else:
        return False


def isScalar(shape):
    """ return true if this is a scalar dataset """

    shape_class = getShapeClass(shape)
    if shape_class == "H5S_SCALAR":
        return True
    else:
        return False


def getDataSize(shape, type_size: int = 1):
    """Return the size of the dataspace.  For
    any unlimited dimensions, assume a value of 1.
    (so the return size will be the absolute minimum)
    """

    if isinstance(shape, dict) and isNullSpace(shape):
        return 0

    if isinstance(shape, dict) and isScalar(shape):
        return type_size  # just return size for one item

    dims = getShapeDims(shape)

    if dims is None:
        return 0
    else:
        return type_size * int(np.prod(dims))
