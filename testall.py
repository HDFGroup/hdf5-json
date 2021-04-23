##############################################################################
# Copyright by The HDF Group.                                                #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of H5Serv (HDF5 REST Server) Service, Libraries and      #
# Utilities.  The full HDF5 REST Server copyright notice, including       s   #
# terms governing use, modification, and redistribution, is contained in     #
# the file COPYING, which can be found at the root of the source code        #
# distribution tree.  If you do not have access to this file, you may        #
# request a copy from help@hdfgroup.org.                                     #
##############################################################################

import os
import sys
import shutil
import h5py

unit_tests = ("hdf5dtypeTest", "hdf5dbTest")
integ_tests = ("h5tojson_test", "jsontoh5_test")

# verify the hdf5 lib version is recent
if h5py.version.hdf5_version_tuple < (1, 10, 4):
    print(h5py.version.info)
    sys.exit("Need HDF5 library 1.10.4 or later")

# verify we have a recent version of h5py
if h5py.version.version_tuple < (3, 0, 0):
    print(h5py.version.info)
    sys.exit("Need h5py version 3.0 or later")

# Run all hdf5-json tests
# Run this script before running any integ tests
os.chdir("test/unit")
for file_name in unit_tests:
    print(file_name)
    rc = os.system("python " + file_name + ".py")
    if rc != 0:
        sys.exit("FAILED")
shutil.rmtree("./out", ignore_errors=True)
os.remove("hdf5dbtest.log")

os.chdir("../integ")
for file_name in integ_tests:
    print(file_name)
    rc = os.system("python " + file_name + ".py")
    if rc != 0:
        sys.exit("FAILED")
shutil.rmtree("./h5_out", ignore_errors=True)
shutil.rmtree("./json_out", ignore_errors=True)
os.remove("h5tojson.log")
os.remove("jsontoh5.log")

os.chdir("..")
print("Testing suite: Success!")
