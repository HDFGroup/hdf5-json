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
import argparse
import os.path as op
import logging
import logging.handlers

from h5json import Hdf5db
from h5json.writer.h5json_writer import H5JsonWriter
from h5json.reader.h5py_reader import H5pyReader
 

def main():
    parser = argparse.ArgumentParser(usage="%(prog)s [-h] [-D|-d] <hdf5_file>")
    parser.add_argument("-D", action="store_true", help="suppress all data output")
    parser.add_argument(
        "-d",
        action="store_true",
        help="suppress data output for" + " datasets (but not attribute values)",
    )
    parser.add_argument("filename", nargs="+", help="HDF5 to be converted to json")
    args = parser.parse_args()

    # create logger
    log = logging.getLogger("h5tojson")
    # log.setLevel(logging.WARN)
    log.setLevel(logging.INFO)
    # add log handler
    handler = logging.FileHandler("./h5tojson.log")

    # add handler to logger
    log.addHandler(handler)

    filename = args.filename[0]
    if not op.isfile(filename):
        sys.exit(f"Cannot find file: {filename}")

    log.info(f"h5tojson {filename}")

    kwargs = {"app_logger": log}
    
    with Hdf5db(h5_reader=H5pyReader(filename, **kwargs), h5_writer=H5JsonWriter("/tmp/foo.json", no_data=False, **kwargs), **kwargs) as db:
        pass

if __name__ == "__main__":
    main()
