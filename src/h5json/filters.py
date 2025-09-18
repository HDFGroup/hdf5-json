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

import h5py

from .hdf5dtype import isVlen

# List of registered filters.  Not all are supported by every reader and writer.
#
#
# tuple of filter key, filter id, and options,
FILTER_DEFS = (
    ("H5Z_FILTER_NONE", 0, "none", ()),
    ("H5Z_FILTER_DEFLATE", 1, "gzip", ("level",)),  # aka as "zlib" for blosc
    ("H5Z_FILTER_SHUFFLE", 2, "shuffle", ()),
    ("H5Z_FILTER_FLETCHER32", 3, "fletcher32", ()),
    ("H5Z_FILTER_SZIP", 4, "szip", ("bitsPerPixel", "coding", "pixelsPerBlock", "pixelsPerScanLine")),
    ("H5Z_FILTER_NBIT", 5, "nbit", ()),
    ("H5Z_FILTER_SCALEOFFSET", 6, "scaleoffset", ("scaleType", "scaleOffset")),
    ("H5Z_FILTER_LZF", 32000, "lzf", ()),
    ("H5Z_FILTER_BLOSC", 32001, "blosclz", ()),
    ("H5Z_FILTER_SNAPPY", 32003, "snappy", ()),
    ("H5Z_FILTER_LZ4", 32004, "lz4", ()),
    ("H5Z_FILTER_LZ4HC", 32005, "lz4hc", ()),
    ("H5Z_FILTER_BITSHUFFLE", 32008, "bitshuffle", ()),
    ("H5Z_FILTER_ZSTD", 32015, "zstd", ()),
)

HDF_FILTER_OPTION_ENUMS = {
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

COMPRESSION_FILTER_IDS = (
    "H5Z_FILTER_DEFLATE",
    "H5Z_FILTER_SZIP",
    "H5Z_FILTER_SCALEOFFSET",
    "H5Z_FILTER_LZF",
    "H5Z_FILTER_BLOSC",
    "H5Z_FILTER_SNAPPY",
    "H5Z_FILTER_LZ4",
    "H5Z_FILTER_LZ4HC",
    "H5Z_FILTER_ZSTD",
)

COMPRESSION_FILTER_NAMES = (
    "gzip",
    "szip",
    "lzf",
    "blosclz",
    "snappy",
    "lz4",
    "lz4hc",
    "zstd",
)


def getFilterItem(key):
    """
    Return filter code, id, and name, based on an id, a name or a code.
    """

    if key == "deflate":
        key = "gzip"  # use gzip as equivalent
    for item in FILTER_DEFS:
        # check for a match by key, id, or alias (the first three elements)
        for i in range(3):
            if key == item[i]:
                return {"class": item[0], "id": item[1], "name": item[2], "options": item[3]}
    return None  # not found


def getFiltersJson(create_props, supported_filters=None):
    """ return standardized filter representation from creation properties
        raise bad request if invalid """

    # refer to https://hdf5-json.readthedocs.io/en/latest/bnf/\
    # filters.html#grammar-token-filter_list

    if "filters" not in create_props:
        return {}  # null set

    f_in = create_props["filters"]

    if not isinstance(f_in, list):
        msg = "Expected filters in creation_props to be a list"
        raise TypeError(msg)

    f_out = []
    for filter in f_in:
        if isinstance(filter, int) or isinstance(filter, str):
            item = getFilterItem(filter)
            if not item:
                msg = f"filter {filter} not recognized"
                raise ValueError(msg)

            if item["name"] not in supported_filters:
                msg = f"filter {filter} is not supported"
                raise ValueError(msg)
            f_out.append(item)
        elif isinstance(filter, dict):
            if "class" not in filter:
                msg = "expected 'class' key for filter property"
                raise KeyError(msg)
            if filter["class"] != "H5Z_FILTER_USER":
                item = getFilterItem(filter["class"])
            elif "id" in filter:
                item = getFilterItem(filter["id"])
            elif "name" in filter:
                item = getFilterItem(filter["name"])
            else:
                item = None
            if not item:
                msg = f"filter {filter['class']} not recognized"
                raise ValueError(msg)
            if "id" not in filter:
                filter["id"] = item["id"]
            elif item["id"] != filter["id"]:
                msg = f"Expected {filter['class']} to have id: "
                msg += f"{item['id']} but got {filter['id']}"
                raise ValueError(msg)
            if "name" not in filter:
                filter["name"] = item["name"]
            if filter["name"] not in supported_filters:
                msg = f"filter {filter} is not supported"
                raise KeyError(msg)

            f_out.append(filter)
        else:
            msg = f"Unexpected type for filter: {filter}"
            raise ValueError(msg)

    # return standardized filter representation
    return f_out


def getFilters(dset_json):
    """Return list of filters, or empty list"""
    if "creationProperties" not in dset_json:
        return []
    creationProperties = dset_json["creationProperties"]
    if "filters" not in creationProperties:
        return []
    filters = creationProperties["filters"]
    return filters


def getCompressionFilter(filters):
    """Return compression filter from filters, or None"""
    for filter in filters:
        if "class" not in filter:
            # expected class key - malformed filter def
            continue
        filter_class = filter["class"]
        if filter_class in COMPRESSION_FILTER_IDS:
            return filter
        if all(
            (
                filter_class == "H5Z_FILTER_USER",
                "name" in filter,
                filter["name"] in COMPRESSION_FILTER_NAMES,
            )
        ):
            return filter
    return None


def getShuffleFilter(filters):
    """Return shuffle filter, or None"""
    FILTER_CLASSES = ("H5Z_FILTER_SHUFFLE", "H5Z_FILTER_BITSHUFFLE")
    for filter in filters:
        if "class" not in filter:
            # invalid filter def?
            continue
        filter_class = filter["class"]
        if filter_class in FILTER_CLASSES:
            return filter

    return None
