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
        
         