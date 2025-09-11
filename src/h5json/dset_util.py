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

import time
import numpy as np


def resize_dataset(dset_json, shape):
    shape_json = dset_json["shape"]
    shape_class = shape_json["class"]
    if shape_class != "H5S_SIMPLE":
        raise TypeError(f"dataset with shape class: {shape_class} cannot be resized")
    if len(shape_json["dims"]) != len(shape):
        raise ValueError("Resize shape parameter doesn't match dataset's rank")
    if "maxdims" not in shape_json:
        raise ValueError("Dataset is not resizable")
    dims = shape_json["dims"]
    maxdims = shape_json["maxdims"]

    if shape_json["dims"] == list(shape):
        # no change, just return
        return
    for i in range(len(dims)):
        extent = shape[i]
        if extent < 0:
            raise ValueError("dimensions can't be negative")
        if maxdims[i] == "H5S_UNLIMITED":
            # any positive extent is ok
            continue
        if extent > maxdims[i]:
            raise ValueError(f"extent for dimension {i} can't be larger than {maxdims[i]}")

    shape_json["dims"] = list(shape)


def getDims(dset_json):
    """ return extents of the dataset shape as a tuple """
    shape_json = dset_json["shape"]
    shape_class = shape_json["class"]
    if shape_class == "H5S_NULL":
        dims = None
    elif shape_class == "H5S_SCALAR":
        dims = ()
    elif shape_class == "H5S_SIMPLE":
        dims = tuple(shape_json["dims"])
    else:
        raise ValueError(f"Unexpected shape class: {shape_class}")
    return dims


def getNumElements(dset_json):
    """ return the number of elements defined by the dataset's shape
        returns None for null shape, 1 for scalar shape, and product of
        extents otherwise """

    return int(np.prod(getDims(dset_json)))


def getDatasetLayout(dset_json):
    """ Return layout json from creation property list or layout json """
    layout = None

    if "creationProperties" in dset_json:
        cp = dset_json["creationProperties"]
        if "layout" in cp:
            layout = cp["layout"]
    if not layout and "layout" in dset_json:
        layout = dset_json["layout"]
    if not layout:
        # no layout for {dset_json
        return None
    return layout


def getDatasetLayoutClass(dset_json):
    """ return layout class """
    layout = getDatasetLayout(dset_json)
    if layout and "class" in layout:
        layout_class = layout["class"]
    else:
        layout_class = None
    return layout_class
