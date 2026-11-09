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
from h5json.jsonstore.h5json_plugin import H5JsonPlugin
from h5json.h5pystore.h5py_plugin import H5pyPlugin
from h5json.apps.applog import setup_logging


def _non_negative_int(text):
    """Parse a non-negative integer argument."""
    try:
        value = int(text)
    except ValueError:
        value = -1
    if value < 0:
        raise argparse.ArgumentTypeError(f"invalid non-negative integer: {text!r}")
    return value


def main():
    parser = argparse.ArgumentParser(description="Print the HDF5/JSON representation of an HDF5 file to stdout.")
    parser.add_argument("--nodata", dest="data_limit", action="store_const", const=0,
                        help="omit dataset values (same as --data-limit 0)")
    parser.add_argument("--data-limit", dest="data_limit", type=_non_negative_int, metavar="N",
                        help="omit the values of datasets larger than about N bytes")
    parser.add_argument("--log", metavar="FILE", help="append DEBUG output to FILE")
    parser.add_argument("hdf5_file", help="HDF5 file to convert")
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    data_limit = args.data_limit
    filename = args.hdf5_file

    # create logger
    log = setup_logging(args.log)

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
