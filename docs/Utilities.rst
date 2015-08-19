###################
Utilities
###################

The hdf5-json distribution includes the following utility scripts.  These are all
located in the ``util`` and ``util\codegen`` directories.

 

jsontoh5.py
-----------

Converts a JSON representation of an HDF5 file to an HDF5 file.

Usage:

``jsontoh5.py [-h] <json_file> <h5_file>``

<json_file> is the input .json file.
<h5_file> is the output file (will be created by the script)

Options:
 * ``-h``: prints help message
 
h5tojson.py
-----------

This script converts the given HDF5 file to a JSON representation of the file.

Usage:

``python h5tojson.py [-h] -[D|-d] <hdf5_file>``

Output is a file the hdf5 file base name and the extension ``.json``.

Options:
 * ``-h``: prints help message
 * ``-D``: suppress all data output
 * ``-d``: suppress data output for datasets (but not attributes)
 
 
 jsontoFortran.py
 ----------------
 
 This script generates code to create Fortran source code that when compiled and run,
 will create an HDF5 file that reflects the JSON input.
 
 Note: Dataset values are not initialized by the Fortran code.
 
 Usage:
 
``python jsontoFortran.py [-h] <json_file> <out_filename>``
 
positional arguments:
  in_filename   JSON file to be converted to h5py
  out_filename  name of HDF5 file to be created by generated code

optional arguments:
  -h, --help    show this help message and exit
  
jsontoh5py.py
 ----------------
 
This script generates code to create Python code (using the h5py package) that when run,
will create an HDF5 file that reflects the JSON input.
 
Note: Dataset values are not initialized by the Python code.
 
Usage:
 
``python jsontoh5py.py [-h] <json_file> <out_filename>``
 
positional arguments:
  in_filename   JSON file to be converted to h5py
  out_filename  name of HDF5 file to be created by generated code

optional arguments:
  -h, --help    show this help message and exit
  
    
  
  
 
 
 




    
