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
from h5json.hsdsstore.hsds_writer import HSDSWriter
from h5json.h5pystore.h5py_reader import H5pyReader

def usage():
    print(f"usage: {sys.argv[0]} [-h] [--nodata] <hdf5_file> <hsds_domain>")
    sys.exit(0)

def main():
    no_data = False
    filename = None
    domain = None
    for i in range(1, len(sys.argv)):
        if sys.argv[i] in ("-h", "--help"):
            usage()
        elif sys.argv[i] == "--nodata":
            no_data = True
        elif filename is None:
            filename = sys.argv[i]
        elif domain is None:
            domain = sys.argv[i]
        else:
            usage()

    if domain is None:
        usage()

    # create logger
    logfname = "h5tohs.log"
    loglevel = logging.DEBUG
    logging.basicConfig(filename=logfname, format='%(levelname)s %(asctime)s %(message)s', level=loglevel)
    log = logging.getLogger()

    # check that the input file exists
    if not op.isfile(filename):
        sys.exit(f"Cannot find file: {filename}")

    log.info(f"h5tohs {filename}")

    db = Hdf5db(app_logger=log)
    db.writer = HSDSWriter(domain, no_data=no_data, app_logger=log)
    db.reader = H5pyReader(filename, app_logger=log)
    db.open()  # read HDF5 data into db

    db.close()  # close will trigger write to HSDS

if __name__ == "__main__":
    main()
