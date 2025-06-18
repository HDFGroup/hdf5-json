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
import logging

from h5json import Hdf5db
from h5json.h5pystore.h5py_writer import H5pyWriter
from h5json.jsonstore.h5json_reader import H5JsonReader


def main():
    if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
        print(f"usage: {sys.argv[0]} [-h] [--nodata] <json_file> <h5_file>")
        sys.exit(0)

    no_data = False
    json_filename = None
    hdf5_filename = None
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "--nodata":
            no_data = True
        elif not json_filename:
            json_filename = sys.argv[i]
        else:
            hdf5_filename = sys.argv[i]

    # create logger
    logfname = "jsontoh5.log"
    loglevel = logging.DEBUG
    logging.basicConfig(filename=logfname, format='%(levelname)s %(asctime)s %(message)s', level=loglevel)
    log = logging.getLogger()

    # check that the input file exists
    if not op.isfile(json_filename):
        sys.exit(f"Cannot find file: {json_filename}")

    log.info(f"jsontoh5 {json_filename} to {hdf5_filename}")

    db = Hdf5db(app_logger=log)
    db.reader = H5JsonReader(json_filename, app_logger=log)
    db.writer = H5pyWriter(hdf5_filename, no_data=no_data, app_logger=log)
    db.open()  # read json data
    # close should create everything the json reader read to the output file
    db.close()


if __name__ == "__main__":
    main()
