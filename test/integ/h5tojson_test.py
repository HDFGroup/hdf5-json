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
import sys
import os
import stat
from shutil import copyfile




"""
main
"""
top_dir = os.path.abspath(os.path.join("..",".."))

data_dir = os.path.join(top_dir, "data","hdf5")

out_dir = os.path.join(top_dir, "test","integ","json_out")

test_files = (
    "array_dset.h5",
    "arraytype.h5",
    # bitfields not supported yet
    # "bitfield_attr.h5",
    # "bitfield_dset.h5",
    "bool_attr.h5",
    "bool_dset.h5",
    "committed_type.h5",
    "compound.h5",
    "compound_array.h5",
    "compound_array_attr.h5",
    #"compound_array_vlen_string.h5",  # crashes python w/ Linux!
    "compound_array_dset.h5",
    "compound_attr.h5",
    "compound_committed.h5",
    "dim_scale.h5",
    "dim_scale_data.h5",
    "dset_creationprop.h5",
    "dset_gzip.h5",
    "empty.h5",
    "enum_attr.h5",
    "enum_dset.h5",
    "fillvalue.h5",
    "fixed_string_attr.h5",  # temp for trying travis
    "fixed_string_dset.h5",  # temp for trying travis
    "h5ex_d_alloc.h5",
    "h5ex_d_checksum.h5",
    "h5ex_d_chunk.h5",
    "h5ex_d_compact.h5",
    # "h5ex_d_extern.h5",
    "h5ex_d_fillval.h5",
    "h5ex_d_gzip.h5",
    "h5ex_d_hyper.h5",
    "h5ex_d_nbit.h5",
    "h5ex_d_rdwr.h5",
    "h5ex_d_shuffle.h5",
    "h5ex_d_sofloat.h5",
    "h5ex_d_soint.h5",
    "h5ex_d_transform.h5",
    "h5ex_d_unlimadd.h5",
    "h5ex_d_unlimgzip.h5",
    "h5ex_d_unlimmod.h5",
    "namedtype.h5",
    "null_objref_dset.h5",
    "null_space_attr.h5",
    "null_space_dset.h5",
    "objref_attr.h5",
    "objref_dset.h5",
    "opaque_attr.h5",
    "opaque_dset.h5",
    "regionref_attr.h5",
    "regionref_dset.h5",
    "resizable.h5",
    "sample.h5",
    "scalar.h5",
    "scalar_array_dset.h5",
    "scalar_attr.h5",
    "tall.h5",
    "tall_with_udlink.h5",
    "tgroup.h5",
    "tref.h5",
    "tstr.h5",
    "types_attr.h5",
    "types_dset.h5",
    "vlen_attr.h5",
    "vlen_dset.h5",
    "vlen_string_attr.h5",
    "vlen_string_dset.h5",
    # "vlen_string_dset_utc.h5",
    "vlen_string_nullterm_attr.h5",
    "vlen_string_nullterm_dset.h5",
    "vlen_unicode_attr.h5",
    "zerodim.h5"
)

# mkdir for output files
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# delete any output files from previous run
for out_file in os.listdir(out_dir):
    split_ext = os.path.splitext(out_file)
    if split_ext[1] == '.json':
        os.unlink(os.path.join(out_dir, out_file))



# convert test files to json
for test_file in test_files:
    split_ext = os.path.splitext(test_file)
    file_path = os.path.join(data_dir, test_file)
    out_file = os.path.join(out_dir, split_ext[0] + ".json")
    if not os.path.exists(file_path):
        sys.exit("file: " + file_path + " not found")
    cmd = "python ../../h5json/h5tojson/h5tojson.py " + file_path + " >" + out_file
    print("cmd:", cmd)
    
    rc = os.system(cmd)
    if rc != 0:
        sys.exit("h5tojson failed converting: " + test_file)
