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
import os.path as op
import argparse

from h5json import Hdf5db
from h5json.h5pystore.h5py_plugin import H5pyPlugin
from h5json.jsonstore.h5json_plugin import H5JsonPlugin
from h5json.apps.applog import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Create an HDF5 file from an HDF5/JSON file.")
    parser.add_argument("--nodata", action="store_true", help="don't write dataset values")
    parser.add_argument("--log", metavar="FILE", help="append DEBUG output to FILE")
    parser.add_argument("json_file", help="HDF5/JSON file to convert")
    parser.add_argument("h5_file", help="HDF5 file to create")
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    no_data = args.nodata
    json_filename = args.json_file
    hdf5_filename = args.h5_file

    # create logger
    log = setup_logging(args.log)

    # check that the input file exists
    if not op.isfile(json_filename):
        sys.exit(f"Cannot find file: {json_filename}")
    if op.exists(hdf5_filename) and op.samefile(json_filename, hdf5_filename):
        parser.error(f"input and output are the same file: {hdf5_filename}")

    log.info(f"jsontoh5 {json_filename} to {hdf5_filename}")

    # read_only=True: src_db never creates/modifies anything, and read_only
    # guarantees flush() can never write back to json_filename even if it
    # somehow did (append alone would still permit a write)
    src_db = Hdf5db(plugin=H5JsonPlugin(json_filename, read_only=True, app_logger=log), app_logger=log)
    src_db.open()  # read json data into src_db

    dst_db = Hdf5db(plugin=H5pyPlugin(hdf5_filename, no_data=no_data, app_logger=log), app_logger=log)
    dst_db.open()

    src_db.copy(dst_db)  # write everything src_db read to the output file

    dst_db.close()
    src_db.close()


if __name__ == "__main__":
    main()
