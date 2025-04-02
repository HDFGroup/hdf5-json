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
import logging.handlers

from h5json import Hdf5db
from h5json.writer.h5json_writer import H5JsonWriter
from h5json.reader.h5py_reader import H5pyReader


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(f"usage: {sys.argv[0]} [-h] [--nodata] <hdf5_file>")
        sys.exit(0)

    no_data = False
    filename = None
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "--nodata":
            no_data = True
        else:
            filename = sys.argv[i]

    # create logger
    log = logging.getLogger("h5tojson")
    # log.setLevel(logging.WARN)
    log.setLevel(logging.INFO)
    # add log handler
    handler = logging.FileHandler("./h5tojson.log")

    # add handler to logger
    log.addHandler(handler)

    if not op.isfile(filename):
        sys.exit(f"Cannot find file: {filename}")

    log.info(f"h5tojson {filename}")

    kwargs = {"app_logger": log}
    reader = H5pyReader(filename, **kwargs)
    writer = H5JsonWriter(None, no_data=no_data, **kwargs)
    kwargs["h5_reader"] = reader
    kwargs["h5_writer"] = writer

    with Hdf5db(**kwargs) as db:
        db.flush()


if __name__ == "__main__":
    main()
