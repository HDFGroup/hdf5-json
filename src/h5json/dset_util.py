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

import math
from .hdf5dtype import getItemSize
from .shape_util import getDataSize
from .objid import isValidUuid

CHUNK_MIN = 512 * 1024  # Soft lower limit (512k)
CHUNK_MAX = 2048 * 1024  # Hard upper limit (2M)


LAYOUT_CLASSES = (
    "H5D_COMPACT",
    "H5D_CONTIGUOUS",
    "H5D_CONTIGUOUS_REF",
    "H5D_CHUNKED",
    "H5D_CHUNKED_REF",
    "H5D_CHUNKED_REF_INDIRECT",
)


def getDatasetLayout(dset_json):
    """ Return layout json from creation property list or layout json """
    layout = None

    if "creationProperties" in dset_json:
        cp = dset_json["creationProperties"]
        if "layout" in cp:
            layout = cp["layout"]

    return layout


def getDatasetLayoutClass(dset_json):
    """ return layout class """
    layout = getDatasetLayout(dset_json)
    if layout and "class" in layout:
        layout_class = layout["class"]
    else:
        layout_class = None
    return layout_class


def resize_dataset(dset_json, shape):
    """ Update shape dims to the given shape provided new shape is valid for maxdims """
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


def getContiguousLayout(shape_json, item_size, chunk_min=None, chunk_max=None):
    """
    create a chunk layout for datasets use contiguous storage.
    """
    if not isinstance(item_size, int):
        msg = "ContiguousLayout can only be used with fixed-length types"
        raise ValueError(msg)

    if chunk_min is None:
        msg = "chunk_min not set"
        raise ValueError(msg)
    if chunk_max is None:
        msg = "chunk_max not set"
        raise ValueError(msg)

    if chunk_max < chunk_min:
        raise ValueError("chunk_max cannot be less than chunk_min")

    if shape_json is None or shape_json["class"] == "H5S_NULL":
        return None
    if shape_json["class"] == "H5S_SCALAR":
        return (1,)  # just enough to store one item
    dims = shape_json["dims"]
    rank = len(dims)
    if rank == 0:
        raise ValueError("rank must be positive for Contiguous Layout")
    for dim in dims:
        if dim < 0:
            raise ValueError("extents must be positive for Contiguous Layout")
        if dim == 0:
            # data shape with no elements, just return dims as layout
            return dims

    n_size = item_size
    layout = [1,] * rank

    for i in range(rank):
        dim = rank - i - 1
        extent = dims[dim]
        if extent * n_size < chunk_max:
            # just use the full extent as layout
            layout[dim] = extent
            n_size *= extent
        else:
            n = extent
            while n > 1:
                n = -(-n // 2)  # use negatives so we round up on odds
                if n * n_size < chunk_max:
                    break
            layout[dim] = n
            break  # just use 1's for the rest of the layout

    return layout


def getChunkSize(chunk_dims, type_size: int = 1):
    """Return chunk size given layout.
    i.e. just the product of the values in the list.
    """

    chunk_size = type_size
    for n in chunk_dims:
        if n <= 0:
            raise ValueError("Invalid chunk layout")
        chunk_size *= n
    return chunk_size


def isExtensible(dims, maxdims):
    """
    Determine if the dataset can be extended
    """
    if maxdims is None or len(dims) == 0:
        return False
    rank = len(dims)
    if len(maxdims) != rank:
        raise ValueError("rank of maxdims does not match dataset")
    for n in range(rank):
        if maxdims[n] in (0, "H5S_UNLIMITED") or maxdims[n] > dims[n]:
            return True
    return False


def getDsetMaxDims(dset_json):
    """
    Get maxdims from a given shape.  Return [1,] for Scalar datasets

    Use with H5S_NULL datasets will throw a ValueError
    """
    if "shape" not in dset_json:
        msg = "No shape found in dset_json"
        raise KeyError(msg)
    shape_json = dset_json["shape"]
    shape_class = shape_json["class"]
    maxdims = None
    if shape_class == "H5S_NULL":
        msg = "Expected shape class other than H5S_NULL"
        raise ValueError(msg)
    elif shape_class == "H5S_SCALAR":
        maxdims = [1,]
    elif shape_class == "H5S_SIMPLE":
        if "maxdims" in shape_json:
            maxdims = shape_json["maxdims"]
        else:
            maxdims = shape_json["dims"]
    else:
        msg = f"Unexpected shape class: {shape_class}"
        raise ValueError(msg)
    return tuple(maxdims)


def getChunkDims(dset_json):
    """Get chunk layout.  Return shape dims for non-chunked layout"""

    shape_json = dset_json["shape"]
    if shape_json["class"] == "H5S_NULL":
        return None
    if shape_json["class"] == "H5S_SCALAR":
        return (1, )
    shape_dims = shape_json["dims"]
    layout_class = getDatasetLayoutClass(dset_json)
    if not layout_class:
        return tuple(shape_dims)

    if not layout_class.startswith("H5D_CHUNKED"):
        # for non-chunked layouts, just return the shape as the chunk dim
        return tuple(shape_dims)

    layout_json = getDatasetLayout(dset_json)
    if "dims" not in layout_json:
        msg = f"Expected dims key in layout: {layout_json}"
        raise KeyError(msg)
    chunk_dims = tuple(layout_json["dims"])
    return chunk_dims


def validateChunkLayout(shape_json, item_size, layout, chunk_table=None):
    """
    Use chunk layout given in the creationPropertiesList (if defined and
    layout is valid).
    Return chunk_layout_json
    """

    rank = 0
    space_dims = None
    chunk_dims = None
    max_dims = None

    if "dims" in shape_json:
        space_dims = shape_json["dims"]
        rank = len(space_dims)

    if "maxdims" in shape_json:
        max_dims = shape_json["maxdims"]
    if "dims" in layout:
        chunk_dims = layout["dims"]

    if chunk_dims:
        # validate that the chunk_dims are valid and correlates with the
        # dataset shape
        if isinstance(chunk_dims, int):
            chunk_dims = [chunk_dims, ]  # promote to array
        if len(chunk_dims) != rank:
            msg = "Layout rank does not match shape rank"
            raise ValueError(msg)
        for i in range(rank):
            dim_extent = space_dims[i]
            chunk_extent = chunk_dims[i]
            if not isinstance(chunk_extent, int):
                msg = "Layout dims must be integer or integer array"
                raise ValueError(msg)
            if chunk_extent <= 0:
                msg = "Invalid layout value"
                raise ValueError(msg)
            if max_dims is None:
                if chunk_extent > dim_extent:
                    msg = "Invalid layout value"
                    raise ValueError(reason=msg)
            elif max_dims[i] != 0:
                if chunk_extent > max_dims[i]:
                    msg = "Invalid layout value for extensible dimension"
                    raise ValueError(msg)
            else:
                pass  # allow any positive value for unlimited dimensions

    if "class" not in layout:
        msg = "class key not found in layout for creation property list"
        raise ValueError(msg)

    layout_class = layout["class"]
    if layout_class == "H5D_CONTIGUOUS_REF":
        # reference to a dataset in a traditional HDF5 files with
        # contiguous storage
        if item_size == "H5T_VARIABLE":
            # can't be used with variable types...
            msg = "Datasets with variable types cannot be used with "
            msg += "reference layouts"
            raise ValueError(msg)
        if "file_uri" not in layout:
            # needed for H5D_CONTIGUOUS_REF
            msg = "'file_uri' key must be provided for "
            msg += "H5D_CONTIGUOUS_REF layout"
            raise ValueError(msg)
        if "offset" not in layout:
            # needed for H5D_CONTIGUOUS_REF
            msg = "'offset' key must be provided for "
            msg += "H5D_CONTIGUOUS_REF layout"
            raise ValueError(msg)
        if "size" not in layout:
            # needed for H5D_CONTIGUOUS_REF
            msg = "'size' key must be provided for "
            msg += "H5D_CONTIGUOUS_REF layout"
            raise ValueError(msg)
        if "dims" in layout:
            # used defined chunk layout not allowed for H5D_CONTIGUOUS_REF
            msg = "'dims' key can not be provided for "
            msg += "H5D_CONTIGUOUS_REF layout"
            raise ValueError(msg)
        if "maxdims" in shape_json:
            # maxdims not allowed for H5D_CONTIGUOUS_REF
            msg = "'maxdims' key can not be provided for "
            msg += "H5D_CONTIGUOUS_REF layout"
            raise ValueError(msg)
    elif layout_class == "H5D_CHUNKED_REF":
        # reference to a dataset in a traditional HDF5 files with
        # chunked storage
        if item_size == "H5T_VARIABLE":
            # can't be used with variable types..
            msg = "Datasets with variable types cannot be used with "
            msg += "reference layouts"
            raise ValueError(msg)
        if "file_uri" not in layout:
            # needed for H5D_CHUNKED_REF
            msg = "'file_uri' key must be provided for "
            msg += "H5D_CHUNKED_REF layout"
            raise ValueError(msg)
        if "dims" not in layout:
            # needed for H5D_CHUNKED_REF
            msg = "'dimns' key must be provided for "
            msg += "H5D_CHUNKED_REF layout"
            raise ValueError(msg)
        if "chunks" not in layout:
            msg = "'chunks' key must be provided for "
            msg += "H5D_CHUNKED_REF layout"
            raise ValueError(msg)
    elif layout_class == "H5D_CHUNKED_REF_INDIRECT":
        # reference to a dataset in a traditional HDF5 files with chunked
        # storage using an auxiliary dataset
        if item_size == "H5T_VARIABLE":
            # can't be used with variable types..
            msg = "Datasets with variable types cannot be used with "
            msg += "reference layouts"
            raise ValueError(msg)
        if "dims" not in layout:
            # needed for H5D_CHUNKED_REF_INDIRECT
            msg = "'dims' key must be provided for "
            msg += "H5D_CHUNKED_REF_INDIRECT layout"
            raise ValueError(msg)
        if "chunk_table" not in layout:
            msg = "'chunk_table' key must be provided for "
            msg += "H5D_CHUNKED_REF_INDIRECT layout"
            raise ValueError(msg)
        chunk_table_id = layout["chunk_table"]
        if not isValidUuid(chunk_table_id, "Dataset"):
            msg = f"Invalid chunk table id: {chunk_table_id}"
            raise ValueError(msg)

    elif layout_class == "H5D_CHUNKED":
        if "dims" not in layout:
            msg = "dims key not found in layout for creation property list"
            raise ValueError(msg)
        if shape_json["class"] != "H5S_SIMPLE":
            msg = "Bad Request: chunked layout not valid with shape class: "
            msg += f"{shape_json['class']}"
            raise ValueError(msg)
    elif layout_class == "H5D_CONTIGUOUS":
        if "dims" in layout:
            msg = "dims key found in layout for creation property list "
            msg += "for H5D_CONTIGUOUS storage class"
            raise ValueError(msg)
        if "maxdims" in shape_json:
            msg = "maxdims found in shape for creation property list "
            msg += "for H5D_CONTIGUOUS storage class"
            raise ValueError(msg)
    elif layout_class == "H5D_COMPACT":
        if "dims" in layout:
            msg = "dims key found in layout for creation property list "
            msg += "for H5D_COMPACT storage class"
            raise ValueError(msg)
        if "maxdims" in shape_json:
            msg = "maxdims found in shape for creation property list "
            msg += "for H5D_COMPACT storage class"
            raise ValueError(msg)
    else:
        msg = f"Unexpected layout: {layout_class}"
        raise ValueError(msg)


def expandChunk(layout, typesize, shape_json, chunk_min=CHUNK_MIN):
    """Compute an increased chunk shape with a size in bytes greater than chunk_min."""
    if shape_json is None or shape_json["class"] == "H5S_NULL":
        return None
    if shape_json["class"] == "H5S_SCALAR":
        return (1,)  # just enough to store one item

    layout = list(layout)
    dims = shape_json["dims"]
    rank = len(dims)
    extendable_dims = 0  # number of dimensions that are extendable
    maxdims = None
    if "maxdims" in shape_json:
        maxdims = shape_json["maxdims"]
        for n in range(rank):
            if maxdims[n] == 0 or maxdims[n] > dims[n]:
                extendable_dims += 1

    dset_size = getDataSize(shape_json, typesize)
    if dset_size <= chunk_min and extendable_dims == 0:
        # just use the entire dataspace shape as one big chunk
        return tuple(dims)

    chunk_size = getChunkSize(layout, typesize)
    if chunk_size >= chunk_min:
        return tuple(layout)  # good already
    while chunk_size < chunk_min:
        # just adjust along extendable dimensions first
        old_chunk_size = chunk_size
        for n in range(rank):
            dim = rank - n - 1  # start from last dim

            if extendable_dims > 0:
                if maxdims[dim] == 0:
                    # infinitely extendable dimensions
                    layout[dim] *= 2
                    chunk_size = getChunkSize(layout, typesize)
                    if chunk_size > chunk_min:
                        break
                elif maxdims[dim] > layout[dim]:
                    # can only be extended so much
                    layout[dim] *= 2
                    if layout[dim] >= dims[dim]:
                        layout[dim] = maxdims[dim]  # trim back
                        extendable_dims -= 1  # one less extendable dimension

                    chunk_size = getChunkSize(layout, typesize)
                    if chunk_size > chunk_min:
                        break
                    else:
                        pass  # ignore non-extensible for now
            else:
                # no extendable dimensions
                if dims[dim] > layout[dim]:
                    # can expand chunk along this dimension
                    layout[dim] *= 2
                    if layout[dim] > dims[dim]:
                        layout[dim] = dims[dim]  # trim back
                    chunk_size = getChunkSize(layout, typesize)
                    if chunk_size > chunk_min:
                        break
                else:
                    pass  # can't extend chunk along this dimension
        if chunk_size <= old_chunk_size:
            # stop iteration if we haven't increased the chunk size
            break
        elif chunk_size > chunk_min:
            break  # we're good
        else:
            pass  # do another round
    return tuple(layout)


def shrinkChunk(layout, typesize, chunk_max=CHUNK_MAX):
    """Compute a reduced chunk shape with a size in bytes less than chunk_max."""

    layout = list(layout)
    chunk_size = getChunkSize(layout, typesize)
    if chunk_size <= chunk_max:
        return tuple(layout)  # good already
    rank = len(layout)

    while chunk_size > chunk_max:
        # just adjust along extendable dimensions first
        old_chunk_size = chunk_size
        for dim in range(rank):
            if layout[dim] > 1:
                # tricky way to do  x // 2 with ceil
                layout[dim] = -(-layout[dim] // 2)
                chunk_size = getChunkSize(layout, typesize)
                if chunk_size <= chunk_max:
                    break
            else:
                pass  # can't shrink chunk along this dimension
        if chunk_size >= old_chunk_size:
            # reality check to see if we'll ever break out of the while loop
            break
        elif chunk_size <= chunk_max:
            break  # we're good
        else:
            pass  # do another round
    return tuple(layout)


def guessChunk(shape_json, typesize, chunk_min=None, chunk_max=None):
    """Guess an appropriate chunk layout for a dataset, given its shape and
    the size of each element in bytes.  Will allocate chunks only as large
    as MAX_SIZE.  Chunks are generally close to some power-of-2 fraction of
    each axis, slightly favoring bigger values for the last index.

    Undocumented and subject to change without warning.
    """
    if shape_json is None or shape_json["class"] == "H5S_NULL":
        return None
    if shape_json["class"] == "H5S_SCALAR":
        return (1,)  # just enough to store one item

    if "maxdims" in shape_json:
        shape = shape_json["maxdims"]
    else:
        shape = shape_json["dims"]

    if typesize == "H5T_VARIABLE":
        typesize = 128  # just take a guess at the item size

    # For unlimited dimensions we have to guess. use 1024
    shape = tuple((x if x != 0 else 1024) for i, x in enumerate(shape))

    chunk_size = getChunkSize(shape, typesize)
    if chunk_min and chunk_size < chunk_min:
        shape = expandChunk(shape, typesize, shape_json, chunk_min=chunk_min)
    elif chunk_max and chunk_size > chunk_max:
        shape = shrinkChunk(shape, typesize, chunk_max=chunk_max)
    else:
        pass  # good already

    return shape


def getLayoutJson(creation_props,
                  shape=None,
                  type_json=None,
                  chunk_min=CHUNK_MIN,
                  chunk_max=CHUNK_MAX,
                  max_chunks_per_folder=0):
    """ Get the layout json given by creation_props.
        Raise value error if invalid """

    item_size = getItemSize(type_json)

    if chunk_min > chunk_max:
        msg = "chunk_max must be larger than chunk_min"
        raise ValueError(msg)

    layout = None
    if "layout" in creation_props:
        layout_props = creation_props["layout"]
    else:
        layout_props = None

    if layout_props:
        if "class" not in layout_props:
            msg = "expected class key in layout props"
            raise KeyError(msg)
        layout_class = layout_props["class"]
        if layout_class == "H5D_CONTIGUOUS":
            # treat contiguous as chunked
            layout_class = "H5D_CHUNKED"
        else:
            layout_class = layout_props["class"]
    elif shape["class"] != "H5S_NULL":
        layout_class = "H5D_CHUNKED"
    else:
        layout_class = None

    if layout_class == "H5D_COMPACT":
        layout = {"class": "H5D_COMPACT"}
    elif layout_class:
        # initialize to H5D_CHUNKED
        layout = {"class": "H5D_CHUNKED"}
    else:
        # null space - no layout
        layout = None

    if layout_props and "dims" in layout_props:
        chunk_dims = layout_props["dims"]
    else:
        chunk_dims = None

    if layout_class == "H5D_CONTIGUOUS_REF":
        kwargs = {"chunk_min": chunk_min, "chunk_max": chunk_max}
        chunk_dims = getContiguousLayout(shape, item_size, **kwargs)
        layout["dims"] = chunk_dims

    if layout_class == "H5D_CHUNKED" and chunk_dims is None:
        # do auto-chunking
        chunk_dims = guessChunk(shape, item_size)

    if layout_class == "H5D_CHUNKED":
        chunk_size = getChunkSize(chunk_dims, item_size)

        # adjust the chunk shape if chunk size is too small or too big
        adjusted_chunk_dims = None
        if chunk_size < chunk_min:
            kwargs = {"chunk_min": chunk_min, "layout_class": layout_class}
            adjusted_chunk_dims = expandChunk(chunk_dims, item_size, shape, **kwargs)
        elif chunk_size > chunk_max:
            kwargs = {"chunk_max": chunk_max}
            adjusted_chunk_dims = shrinkChunk(chunk_dims, item_size, **kwargs)
        if adjusted_chunk_dims:
            layout["dims"] = adjusted_chunk_dims
        else:
            layout["dims"] = chunk_dims  # don't need to adjust chunk size

        # set partition_count if needed:
        set_partition = False
        if max_chunks_per_folder > 0:
            if "dims" in shape and "dims" in layout:
                set_partition = True

        if set_partition:
            chunk_dims = layout["dims"]
            shape_dims = shape["dims"]
            if "maxdims" in shape:
                max_dims = shape["maxdims"]
            else:
                max_dims = None
            num_chunks = 1
            rank = len(shape_dims)
            unlimited_count = 0
            if max_dims:
                for i in range(rank):
                    if max_dims[i] == 0:
                        unlimited_count += 1
            for i in range(rank):
                max_dim = 1
                if max_dims:
                    max_dim = max_dims[i]
                    if max_dim == 0:
                        # don't really know what the ultimate extent
                        # could be, but assume 10^6 for total number of
                        # elements and square-shaped array...
                        MAX_ELEMENT_GUESS = 10.0 ** 6
                        exp = 1 / unlimited_count
                        max_dim = int(math.pow(MAX_ELEMENT_GUESS, exp))
                else:
                    max_dim = shape_dims[i]
                num_chunks *= math.ceil(max_dim / chunk_dims[i])

            if num_chunks > max_chunks_per_folder:
                partition_count = math.ceil(num_chunks / max_chunks_per_folder)
                msg = f"set partition count to: {partition_count}, "
                msg += f"num_chunks: {num_chunks}"
                layout["partition_count"] = partition_count
            else:
                pass  # partition not needed

    if layout_class in ("H5D_CHUNKED_REF", "H5D_CHUNKED_REF_INDIRECT"):
        chunk_size = getChunkSize(chunk_dims, item_size)

        # nothing to do about inefficiently small chunks, but large chunks
        # can be subdivided
        if chunk_size < chunk_min:
            pass  # too small
        elif chunk_size > chunk_max:
            pass  # too large
        layout["dims"] = chunk_dims
