##############################################################################
# Copyright by The HDF Group.                                                #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HSDS (HDF5 REST Server) Service, Libraries and        #
# Utilities.  The full HDF5 REST Server copyright notice, including          #
# terms governing use, modification, and redistribution, is contained in     #
# the file COPYING, which can be found at the root of the source code        #
# distribution tree.  If you do not have access to this file, you may        #
# request a copy from help@hdfgroup.org.                                     #
##############################################################################

def getTrackTimes(obj_json):
    """ Return a boolean if trackTimes is set in the objects' creation Property list.
        Otherwise return None. """

    if "creationProperties" in obj_json:
        cpl = obj_json["creationProperties"]
    else:
        cpl = obj_json  # assume this is the cpl
    if "trackTimes" in cpl:
        track_times = bool(cpl["trackTimes"])
    else:
        track_times = None

    return track_times
