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


def getAllFilterNames():
    """ Return list of all recognized filter names """

    names = set()
    for item in FILTER_DEFS:
        filter_id = item[1]
        filter_name = item[2]
        if filter_id > 0 and filter_name:
            names.add(filter_name)
    names = list(names)
    names.sort()
    return tuple(names)


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

    if not supported_filters:
        supported_filters = getAllFilterNames()

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
            if filter.get("class") == "H5Z_FILTER_USER":
                # user filter - must have either id or name
                if "id" not in filter and "name" not in filter:
                    msg = "user filter must have either 'id' or 'name' key"
                    raise KeyError(msg)
                item = filter
            elif "id" in filter:
                item = getFilterItem(filter["id"])
            elif "name" in filter:
                item = getFilterItem(filter["name"])
            else:
                item = None
            if not item:
                msg = f"filter {filter} not recognized"
                raise ValueError(msg)

            # copy any filter specified options
            filter_class = item["class"]
            if filter_class == "H5Z_FILTER_DEFLATE":
                if "level" in filter:
                    level_val = filter["level"]
                    if not isinstance(level_val, int):
                        msg = "Expected integer level for deflate filter"
                        raise TypeError(msg)
                    if level_val < 0 or level_val > 9:
                        msg = "Deflate filter level must be between 0 and 9"
                        raise ValueError(msg)
                    item["level"] = level_val
            elif filter_class == "H5Z_FILTER_SHUFFLE":
                pass  # no options
            elif filter_class == "H5Z_FILTER_FLETCHER32":
                pass  # no options
            elif filter_class == "H5Z_FILTER_SZIP":
                for key in ("bitsPerPixel", "coding", "pixelsPerBlock", "pixelsPerScanLine"):
                    if key in filter:
                        val = filter[key]
                        if key == "coding":
                            if val not in HDF_FILTER_OPTION_ENUMS["coding"].values():
                                msg = f"Invalid coding option for szip filter: {val}"
                                raise ValueError(msg)
                            else:
                                # other options need to be positivie integers
                                if not isinstance(val, int) or val <= 0:
                                    msg = f"Expected positive integer for szip filter option {key}"
                                    raise ValueError(msg)
                        item[key] = val
            elif filter_class == "H5Z_FILTER_NBIT":
                pass  # no options
            elif filter_class == "H5Z_FILTER_SCALEOFFSET":
                if "scaleType" in filter:
                    val = filter["scaleType"]
                    if val not in HDF_FILTER_OPTION_ENUMS["scaleType"].values():
                        msg = f"Invalid scaleType option for scaleoffset filter: {val}"
                        raise ValueError(msg)
                    else:
                        item["scaleType"] = val
                if "scaleOffset" in filter:
                    val = filter["scaleOffset"]
                    if not isinstance(val, int) or val < 0:
                        msg = "Expected non-negative integer for scaleOffset option"
                        raise ValueError(msg)
                    else:
                        item["scaleOffset"] = val
            elif filter_class == "H5Z_FILTER_LZF":
                pass  # no options
            elif filter_class == "H5Z_FILTER_BLOSC":
                pass  # no options
            elif filter_class == "H5Z_FILTER_SNAPPY":
                pass  # no options
            elif filter_class == "H5Z_FILTER_LZ4":
                pass  # no options
            elif filter_class == "H5Z_FILTER_LZ4HC":
                pass  # no options
            elif filter_class == "H5Z_FILTER_BITSHUFFLE":
                pass  # no options
            elif filter_class == "H5Z_FILTER_ZSTD":
                pass  # no options
            else:
                msg = f"filter class {filter_class} is not supported"
                raise KeyError(msg)
            f_out.append(item)
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


def getFilterOps(filters, dtype=None):
    """Get list of filter operations to be used for this dataset"""

    compressionFilter = getCompressionFilter(filters)

    filter_ops = {}

    shuffleFilter = getShuffleFilter(filters)

    if shuffleFilter and not isVlen(dtype):
        shuffle_name = shuffleFilter["name"]
        if shuffle_name == "shuffle":
            filter_ops["shuffle"] = 1  # use regular shuffle
        elif shuffle_name == "bitshuffle":
            filter_ops["shuffle"] = 2  # use bitshuffle
        else:
            filter_ops["shuffle"] = 0  # no shuffle
    else:
        filter_ops["shuffle"] = 0  # no shuffle

    """ return list of filter operations for this dataset """
    if compressionFilter:
        if compressionFilter["class"] == "H5Z_FILTER_DEFLATE":
            filter_ops["compressor"] = "zlib"  # blosc compressor
        else:
            if "name" in compressionFilter:
                filter_ops["compressor"] = compressionFilter["name"]
            else:
                filter_ops["compressor"] = "lz4"  # default to lz4
        if "level" not in compressionFilter:
            filter_ops["level"] = 5  # medium level
        else:
            filter_ops["level"] = int(compressionFilter["level"])

    return filter_ops
