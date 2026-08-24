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
# is_reference/is_regionreference/has_reference are pure duck-typing helpers on
# h5json's own Reference/RegionReference classes (no h5py dependency), so they
# live in hdf5dtype.py alongside those classes - re-exported here since this
# module's own convert_dtype() (which does need h5py) uses them too, and
# existing code imports them from here.
from .hdf5dtype import is_reference, is_regionreference, has_reference  # noqa: F401


def convert_dtype(srcdt, to_h5py=True):
    """Return a dtype based on input dtype, converting any Reference types from
    h5py style to h5json and vice-versa.
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
                tgt_dt = hdf5dtype.special_dtype(vlen=tgt_base)
        elif srcdt.kind == "U":
            # use vlen for unicode strings
            if to_h5py:
                tgt_dt = h5py.special_dtype(vlen=str)
            else:
                tgt_dt = hdf5dtype.special_dtype(vlen=str)
        else:
            tgt_dt = srcdt
    return tgt_dt
