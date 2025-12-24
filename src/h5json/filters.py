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

DEFAULT_GZIP = 4
DEFAULT_SZIP = 4
SO_INT_MINBITS_DEFAULT = 0

# List of registered filters.  Not all are supported by every reader and writer.
#
#
# tuple of filter key, filter id, and options,
FILTER_DEFS = (
    ("H5Z_FILTER_NONE", 0, "none", ()),
    ("H5Z_FILTER_DEFLATE", 1, "gzip", ("level",)),  # aka as "default" or "zlib" for blosc
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


def getFilterItem(name, options={}):
    """
    Return filter code, id, and name, based on an id, a name or a code.
    """
    # is key is dict, just verify it's a valid filter and return
    filter_json = None

    if isinstance(name, dict):
        filter_json = name
        base_keys = ("class", "id", "name")
        for key in base_keys:
            if key not in filter_json:
                raise KeyError(f"Expected {key} for filter")
        # use class key to look up options
        name = filter_json["class"]
    elif name in ("deflate", "zlib"):
        name = "gzip"  # use gzip as equivalent

    option_set = None
    for item in FILTER_DEFS:
        # check for a match by key, id, or alias (the first three elements)
        for i in range(3):
            if name == item[i]:
                if filter_json is None:
                    filter_json = {"class": item[0], "id": item[1], "name": item[2]}
                option_set = set(item[3])
                break

    if not filter_json and isinstance(name, int) and name > 32000:
        filter_json = {"class": "H5Z_FILTER_USER", "id": name, "name": f"user filter {name}"}

    if not filter_json:
        raise KeyError(f"filter {name} is unknown")

    filter_class = filter_json["class"]
    if filter_class == "H5Z_FILTER_USER":
        option_set = set()
        option_set.add("parameters")

    # check that any option supplied is supported by the filter
    for key in options:
        if key not in option_set:
            msg = f"Option {key} is not supported by the {filter_class} filter"
            raise KeyError(msg)

    # for any supplied options verify they are correct type and range
    # (raise Type or Value error if not).  If option is not given, use
    # the default value if not.  Finally add options to the filter_json

    if filter_class == "H5Z_FILTER_DEFLATE":
        if "level" in options:
            level_val = options["level"]
            if not isinstance(level_val, int):
                msg = "Expected integer level for deflate filter"
                raise TypeError(msg)
            if level_val < 0 or level_val > 9:
                msg = "Deflate filter level must be between 0 and 9"
                raise ValueError(msg)
            filter_json["level"] = level_val
        else:
            filter_json["level"] = DEFAULT_GZIP

    elif filter_class == "H5Z_FILTER_SHUFFLE":
        pass  # no options
    elif filter_class == "H5Z_FILTER_FLETCHER32":
        pass  # no options
    elif filter_class == "H5Z_FILTER_SZIP":
        for key in option_set:        # option set("bitsPerPixel", "coding", "pixelsPerBlock", "pixelsPerScanLine"):
            if key in options:
                val = options[key]
                if key == "coding":
                    if val not in HDF_FILTER_OPTION_ENUMS["coding"].values():
                        msg = f"Invalid coding option for szip filter: {val}"
                        raise ValueError(msg)
                else:
                    # other options need to be positivie integers
                    if not isinstance(val, int) or val <= 0:
                        msg = f"Expected positive integer for szip filter option {key}"
                        raise ValueError(msg)
                filter_json[key] = val
            else:
                pass  # no defaults for szip
    elif filter_class == "H5Z_FILTER_NBIT":
        pass  # no options
    elif filter_class == "H5Z_FILTER_SCALEOFFSET":
        if "scaleType" in options:
            val = options["scaleType"]
            if val not in HDF_FILTER_OPTION_ENUMS["scaleType"].values():
                msg = f"Invalid scaleType option for scaleoffset filter: {val}"
                raise ValueError(msg)

            filter_json["scaleType"] = val
        if "scaleOffset" in options:
            val = options["scaleOffset"]
            if not isinstance(val, int) or val < 0:
                msg = "Expected non-negative integer for scaleOffset option"
                raise ValueError(msg)
            filter_json["scaleOffset"] = val
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
    elif filter_class == "H5Z_FILTER_NONE":
        pass  # no options
    elif filter_class == "H5Z_FILTER_USER":
        if "parameters" in options:
            parameters = options["parameters"]
            # expecting a positive integer array
            if not isinstance(parameters, (list, tuple)):
                raise TypeError(f"filter {filter_class} parameters option should be a list")
            vals = []
            for val in parameters:
                if not isinstance(val, int):
                    raise TypeError(f"filter {filter_class} parameters expected integer value")
                if val <= 0:
                    raise TypeError(f"filter {filter_class} parameters option should be a positive int")
                vals.append(val)
            filter_json["parameters"] = val
    else:
        msg = f"filter class {filter_class} is not supported"
        raise KeyError(msg)

    return filter_json


def validateFilter(filter_json, supported_filters=None):
    """ Check the given the given filter for create format,
        required options set.  Raise TypeError, KeyError or ValueError if not.
        If supported_filters is supplied, raise KeyError if a non-supported
        filter is supplied. """

    if not isinstance(filter_json, dict):
        raise TypeError(f"Expected dict for filter but got {type(filter_json)}")
    base_keys = ("class", "id", "name")
    for key in base_keys:
        if key not in filter_json:
            raise KeyError(f"Expected {key} for filter")
    filter_class = filter_json["class"]
    filter_id = filter_json["id"]
    # check that the filter_class agrees with the id in FILTER_DEFS
    options = None
    for filter_def in FILTER_DEFS:
        if filter_def[0] == filter_class:
            if filter_id != filter_def[1]:
                msg = f"Incorrect filter_id: {filter_id} for filter: {filter_class}"
                raise ValueError(msg)
            # collect any filter options to check later
            options = {}
            for key in filter_json:
                if key in base_keys:
                    continue
                options[key] = filter_json[key]
            break

    if options is None and filter_class == "H5Z_FILTER_USER":
        # custom filter, id should be > 32000
        if filter_id <= 32000:
            raise ValueError(f"Unexpected filter id: {filter_id} for user filter")
        options = {}
        for key in filter_json:
            if key in base_keys:
                continue
            options[key] = filter_json[key]

    if options is None:
        raise KeyError(f"Unknown filter: {filter_class}")

    # will raise error if any option is invalid
    getFilterItem(filter_json, options)


def validateFilters(filters, supported_filters=None):
    """ validate each filter in the filter list """

    # TBD: check given order of filters is supported
    for filter_json in filters:
        validateFilter(filter_json, supported_filters=supported_filters)


def getFilters(dset_json):
    """Return list of filters, or empty list"""
    if "creationProperties" not in dset_json:
        return []
    creationProperties = dset_json["creationProperties"]
    if "filters" not in creationProperties:
        return []
    filters = creationProperties["filters"]
    return filters


def isCompressionFilter(filter):
    filter_json = getFilterItem(filter)
    return filter_json["class"] in COMPRESSION_FILTER_IDS


def getCompressionFilter(filters):
    """Return compression filter ids from filters, or None"""
    return COMPRESSION_FILTER_IDS
