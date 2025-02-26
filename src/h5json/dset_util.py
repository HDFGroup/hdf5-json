##############################################################################
# Copyright by The HDF Group.                                                #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HSDS (HDF5 REST Server) Service, Libraries and      #
# Utilities.  The full HDF5 REST Server copyright notice, including          #
# terms governing use, modification, and redistribution, is contained in     #
# the file COPYING, which can be found at the root of the source code        #
# distribution tree.  If you do not have access to this file, you may        #
# request a copy from help@hdfgroup.org.                                     #
##############################################################################

import time
from .hdf5dtype import getTypeItem

"""
# standard compress filters
_HDF_FILTERS = {
    1: {"class": "H5Z_FILTER_DEFLATE", "alias": "gzip", "options": ["level"]},
    2: {"class": "H5Z_FILTER_SHUFFLE", "alias": "shuffle"},
    3: {"class": "H5Z_FILTER_FLETCHER32", "alias": "fletcher32"},
    4: {
        "class": "H5Z_FILTER_SZIP",
        "alias": "szip",
        "options": ["bitsPerPixel", "coding", "pixelsPerBlock", "pixelsPerScanLine"],
    },
    5: {"class": "H5Z_FILTER_NBIT"},
    6: {
        "class": "H5Z_FILTER_SCALEOFFSET",
        "alias": "scaleoffset",
        "options": ["scaleType", "scaleOffset"],
    },
    32000: {"class": "H5Z_FILTER_LZF", "alias": "lzf"},
}

_HDF_FILTER_OPTION_ENUMS = {
    "coding": {
        h5py.h5z.SZIP_EC_OPTION_MASK: "H5_SZIP_EC_OPTION_MASK",
        h5py.h5z.SZIP_NN_OPTION_MASK: "H5_SZIP_NN_OPTION_MASK",
    },
    "scaleType": {
        h5py.h5z.SO_FLOAT_DSCALE: "H5Z_SO_FLOAT_DSCALE",
        h5py.h5z.SO_FLOAT_ESCALE: "H5Z_SO_FLOAT_ESCALE",
        h5py.h5z.SO_INT: "H5Z_SO_INT",
    },
}

# h5py supported filters
_H5PY_FILTERS = {
    "gzip": 1,
    "shuffle": 2,
    "fletcher32": 3,
    "szip": 4,
    "scaleoffset": 6,
    "lzf": 32000,
}

_H5PY_COMPRESSION_FILTERS = ("gzip", "lzf", "szip")
"""

def resize_dataset(dset_json, shape):
    shape_json = dset_json["shape"]
    shape_class = shape_json["class"]
    if shape_class != "H5S_SIMPLE":
        raise TypeError(f"dataset with shape class: {shape_class} cannot be resized")
    if len(shape_class["dims"]) != len(shape):
        raise ValueError("Resize shape parameter doesn't match dataset's rank")
    if shape_json["dims"] == list(shape):
        # no change, just return
        return
    shape_json["dims"] = list(shape)
    dset_json["modified"] = time.time()
        
         