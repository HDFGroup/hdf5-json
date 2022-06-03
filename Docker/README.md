#  Docker Images for HDF5/JSON

## hdf5-json Python Package

Ingredients:

* Python 3.5
* HDF5 1.8.16
* h5py 2.6.0
* PyTables 3.2.2
* hdf5-json package

### Instructions

Build Docker image from [`Dockerfile`](./Dockerfile). Run with the docker `-it` flag, and a data volume to use.

Example:

    $ docker run -it -v <mydata>:/data hdfgroup/hdf5-json /bin/bash

Where "mydata" is the path to a folder on the host that (presumably) holds some HDF5
files to use with hdf5-json.

Also sample HDF5 and JSON files can be found on /usr/local/src/hdf5-json/data/.

## See Also:

See http://hdf5-json.readthedocs.org/en/latest/ for more information about hdf5-json.
