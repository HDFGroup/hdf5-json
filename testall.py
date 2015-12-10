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
import h5py

unit_tests = ( 'hdf5dtypeTest', 'hdf5dbTest' )
integ_tests = ( 'h5tojson_test', 'jsontoh5_test' )

# verify the hdf5 lib version is recent
hdf5_version = h5py.version.hdf5_version_tuple
print("hdf5_version:", hdf5_version)
if hdf5_version[1] < 8:
    sys.exit("Need hdf5 lib 1.8 or later")
if hdf5_version[1] == 8 and hdf5_version[2] < 4:
    sys.exit("Need hdf5 lib 1.8.4 or later")
# verify we have a recent version of h5py

h5py_version = h5py.version.version_tuple
print("h5py_version:", h5py_version)
if h5py_version[0] != 2 or h5py_version[1] < 5:
    sys.exit("Need h5py version 2.5 or later")
    
#
#
# Run all hdf5-json tests
# Run this script before running any integ tests
#
os.chdir('test/unit')
for file_name in unit_tests:
    print(file_name)
    rc = os.system('python ' + file_name + '.py')
    if rc != 0:
        sys.exit("Failed")

os.chdir('../integ')
for file_name in integ_tests:
    print(file_name)
    rc = os.system('python ' + file_name + '.py')
    if rc != 0:
        sys.exit("failed")
os.chdir('..')
print("Done!")
