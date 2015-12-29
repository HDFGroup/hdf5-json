#!/usr/bin/env python
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

"""
    This is the main setup script for hdf5-json (https://github.com/HDFGroup/hdf5-json).
    
"""

import os 
from distutils.core import setup
 

VERSION = '1.0.0'

 

# --- Distutils setup and metadata --------------------------------------------

cls_txt = \
"""
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: Information Technology
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Database
Topic :: Software Development :: Libraries :: Python Modules
Operating System :: Unix
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows
"""

short_desc = "HDF5/JSON conversion library and tools"

long_desc = \
"""
The hdf5-json package provides a library that contains methods for converting
HDF5 objects as JSON and creating HDF5 objects based on a JSON specification.  
It also contains scripts to convert HDF5 files to JSON and vice-versa.

Supports HDF5 versions 1.8.10 and higher.   
"""

if os.name == 'nt':
    package_data = {'h5json': ['*.dll']}
else:
    package_data = {'h5json': []}

setup(
  name = 'hdf5json',
  version = VERSION,
  description = short_desc,
  long_description = long_desc,
  classifiers = [x for x in cls_txt.split("\n") if x],
  author = 'John Readey',
  author_email = 'jreadey at hdfgroup dot org',
  maintainer = 'John Readey',
  maintainer_email = 'jreadey at hdfgroup dot org',
  url = 'https://github.com/HDFGroup/hdf5-json',
  download_url = 'https://pypi.python.org/pypi/h5json',
  packages = ['h5json'],
  package_data = package_data,
  requires = ['numpy (>=1.6.1)', 'h5py (>=2.5)'] 
)
