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
import numpy as np

from . import hdf5dtype

def is_reference(val):
    """ Return True if the type or value is a Reference """

    if isinstance(val, object) and val.__class__.__name__ == "Reference":
        return True
    elif isinstance(val, type) and val.__name__ == "Reference":
        return True
 
    return False


def is_regionreference(val):
    """ Return True if the type or value is a RegionReference """

    if isinstance(val, object) and val.__class__.__name__ == "RegionReference":
        return True
    elif isinstance(val, type) and val.__name__ == "RegionReference":
        return True

    return False


def has_reference(dtype):
    """ return True if the dtype (or a sub-type) is a Reference type """
    has_ref = False
    if not isinstance(dtype, np.dtype):
        return False
    if len(dtype) > 0:
        for name in dtype.fields:
            item = dtype.fields[name]
            if has_reference(item[0]):
                has_ref = True
                break
    elif dtype.metadata and "ref" in dtype.metadata:
        basedt = dtype.metadata["ref"]
        has_ref = is_reference(basedt)
    elif dtype.metadata and "vlen" in dtype.metadata:
        basedt = dtype.metadata["vlen"]
        has_ref = has_reference(basedt)
    return has_ref


def convert_dtype(srcdt, to_h5py=True):
    """Return a dtype based on input dtype, converting any Reference types from
    h5py style to h5pyd and vice-versa.
    """

    if len(srcdt) > 0:
        fields = []
        for name in srcdt.fields:
            item = srcdt.fields[name]
            # item is a tuple of dtype and integer offset
            field_dt = convert_dtype(item[0], to_h5py=to_h5py)
            fields.append((name, field_dt))
        tgt_dt = np.dtype(fields)
    else:
        # check if this a "special dtype"
        if srcdt.metadata and "ref" in srcdt.metadata:
            ref = srcdt.metadata["ref"]
            if is_reference(ref):
                if to_h5py:
                    tgt_dt = h5py.special_dtype(ref=h5py.Reference)
                else:
                    tgt_dt = hdf5dtype.special_dtype(ref=hdf5dtype.Reference)
            elif is_regionreference(ref):
                if to_h5py:
                    tgt_dt = h5py.special_dtype(ref=h5py.RegionReference)
                else:
                    tgt_dt = hdf5dtype.special_dtype(ref=hdf5dtype.RegionReference)
            else:
                msg = f"Unexpected ref type: {srcdt}"
                raise TypeError(msg)
        elif srcdt.metadata and "vlen" in srcdt.metadata:
            src_vlen = srcdt.metadata["vlen"]
            if isinstance(src_vlen, np.dtype):
                tgt_base = convert_dtype(src_vlen, to_h5py=to_h5py)
            else:
                tgt_base = src_vlen
            if to_h5py:
                tgt_dt = h5py.special_dtype(vlen=tgt_base)
            else:
                tgt_dt = h5pyd.special_dtype(vlen=tgt_base)
        elif srcdt.kind == "U":
            # use vlen for unicode strings
            if to_h5py:
                tgt_dt = h5py.special_dtype(vlen=str)
            else:
                tgt_dt = hdf5dtype.special_dtype(vlen=str)
        else:
            tgt_dt = srcdt
    return tgt_dt
