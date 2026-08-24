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
from h5json.jsonstore.h5json_plugin import H5JsonPlugin
from h5json.h5pystore.h5py_plugin import H5pyPlugin


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(f"usage: {sys.argv[0]} [-h] [--nodata] [--data-limit n] <hdf5_file>")
        sys.exit(0)

    data_limit = None
    filename = None
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "--nodata":
            data_limit = 0
        elif sys.argv[i] == "--data-limit":
            i += 1
            if i >= len(sys.argv):
                sys.exit("Error: --data-limit requires a numeric argument")
            try:
                data_limit = int(sys.argv[i])
            except ValueError:
                sys.exit("Error: --data-limit requires a numeric argument")
        else:
            filename = sys.argv[i]

    # create logger
    logfname = "h5tojson.log"
    loglevel = logging.DEBUG
    logging.basicConfig(filename=logfname, format='%(levelname)s %(asctime)s %(message)s', level=loglevel)
    log = logging.getLogger()

    # check that the input file exists
    if not op.isfile(filename):
        sys.exit(f"Cannot find file: {filename}")

    log.info(f"h5tojson {filename}")

    # read_only=True: open the source file in h5py mode='r' - src_db never
    # creates/modifies anything, and read_only guarantees that even if it
    # somehow did, nothing could actually be written back to the source file
    src_db = Hdf5db(plugin=H5pyPlugin(filename, read_only=True, app_logger=log), app_logger=log)
    src_db.open()  # read HDF5 data into src_db

    dst_db = Hdf5db(plugin=H5JsonPlugin(None, data_limit=data_limit, app_logger=log), app_logger=log)
    dst_db.open()

    src_db.copy(dst_db)  # write src_db's content into dst_db

    dst_db.close()  # triggers write to json file (stdout, since filepath is None)
    src_db.close()


if __name__ == "__main__":
    main()
