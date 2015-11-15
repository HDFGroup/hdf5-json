##############################################################################
# Copyright by The HDF Group.                                                #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of jsonServ (HDF5 REST Server) Service, Libraries and      #
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
top_dir = os.path.abspath("../..")

data_dir = os.path.join(top_dir, "data/json")

util_dir = os.path.join(top_dir, "util")

out_dir = os.path.join(top_dir, "test/integ/h5_out")

test_files = (
    # "array_dset.json",
    # "arraytype.json",
    # bitfields not supported yet
    # "bitfield_attr.json",
    # "bitfield_dset.json",
    "committed_type.json",
    "compound.json",
    # "compound_array.json",
    # "compound_array_attr.json",
    # "compound_array_dset.json",
    "compound_attr.json",
    "compound_committed.json",
    "dim_scale.json",
    "dim_scale_data.json",
    "dset_creationprop.json",
    # "dset_gzip.json",
    "empty.json",
    # "enum_attr.json",
    # "enum_dset.json",
    "fillvalue.json",
    #"fixed_string_attr.json",  # temp removal for travis test
    #"fixed_string_dset.json",  # temp removal for travis test
    "h5ex_d_alloc.json",
    "h5ex_d_checksum.json",
    "h5ex_d_chunk.json",
    "h5ex_d_compact.json",
    # "h5ex_d_extern.json",
    "h5ex_d_fillval.json",
    "h5ex_d_gzip.json",
    "h5ex_d_hyper.json",
    "h5ex_d_nbit.json",
    "h5ex_d_rdwr.json",
    "h5ex_d_shuffle.json",
    # "h5ex_d_sofloat.json",
    # "h5ex_d_soint.json",
    "h5ex_d_transform.json",
    "h5ex_d_unlimadd.json",
    "h5ex_d_unlimgzip.json",
    # "h5ex_d_unlimmod.json",
    "namedtype.json",
    # "null_objref_dset.json",
    #"null_space_attr.json",  # temp removal for travis test
    #"null_space_dset.json",  # temp removal for travis test
    "objref_attr.json",
    # "objref_dset.json",
    # "opaque_attr.json",
    # "opaque_dset.json",
    "regionref_attr.json",
    # "regionref_dset.json",
    "resizable.json",
    # "sample.json",
    "scalar.json",
    "scalar_attr.json",
    "tall.json",
    "tall_with_udlink.json",
    "tgroup.json",
    # "tref.json",
    # "tstr.json",
    "types_attr.json",
    "types_dset.json",
    "vlen_attr.json",
    "vlen_dset.json",
    "vlen_string_attr.json",
    "vlen_string_dset.json",
    # "vlen_string_dset_utc.json",
    "vlen_string_nullterm_attr.json",
    "vlen_string_nullterm_dset.json",
    "vlen_unicode_attr.json",
    "zerodim.json"
)

# mkdir for output files
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# delete any output files from previous run
for out_file in os.listdir(out_dir):
    split_ext = os.path.splitext(out_file)
    if split_ext[1] == '.h5':
        os.unlink(os.path.join(out_dir, out_file))

# convert test files to json
for test_file in test_files:
    split_ext = os.path.splitext(test_file)
    file_path = os.path.join(data_dir, test_file)
    out_file = os.path.join(out_dir, split_ext[0] + ".h5")
    if not os.path.exists(file_path):
        sys.exit("file: " + file_path + " not found")
    cmd = "python jsontoh5.py " + file_path + " " + out_file
    print("cmd:", cmd)
    rc = os.system("cd " + util_dir + "; " + cmd  )
    if rc != 0:
        sys.exit("jsontoh5 failed converting: " + test_file)
