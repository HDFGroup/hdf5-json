####################
Installing hdf5-json
####################

Instructions for installing hdf5-json library and utilties.


Prerequisites
-------------

A computer running a 64-bit version of Windows, Mac OS X, or Linux.

You will also need the following Python packages:

* Python 2.7, 3.3, 3.4, or 3.5
* NumPy 1.9.2 or later
* h5py 2.5 or later

If you are not familiar with installing Python packages, the easiest route is to 
use a package manager such as Anaconda (as described below).

If you have a git client installed on your system, you can directly download the hdf5-json 
source from GitHub: ``git clone --recursive https://github.com/HDFGroup/hdf5-json.git``.  
Otherwise, you can download a zip file of the source from GitHub (as described below).


Installing on Windows
---------------------

Anaconda from Continuum Analytics can be used to easily manage the package dependencies 
needed for hdf5-json.  

In a browser go to: http://continuum.io/downloads and click the "Windows 64-bit 
Python 2.7 Graphical Installer" button.

Install Anaconda using the default options.

Once Anaconda is installed select "Anaconda Command Prompt" from the start menu.

In the command window that appears, create a new anaconda environment using the following command:
``conda create -n hdf5json python=2.7 h5py``

Answer 'y' to the prompt, and the packages will be fetched.

In the same command window, run: ``activate hdf5json``

In a browser go to: https://github.com/HDFGroup/hdf5-json and click the "Download ZIP"
button (right side of page).  Save the file as "hdf5json.zip" to your Downloads directory.

Alternatively, if you have git installed, you can run: 
``git clone --recursive https://github.com/HDFGroup/hdf5-json.git`` to download the hdf5-json source tree. 

If you downloaded the ZIP file, in Windows Explorer, right-click on the file and select 
"Extract All...".  You can choose any folder as the destination.

Next, in the command window, cd to the folder you extracted the source files to.

Run:
``python setup.py install``
to install the package.

Installing on Linux/Mac OS X
-----------------------------

Anaconda from Continuum Analytics can be used to easily manage the package dependencies 
needed for hdf5-json.  

In a browser go to: http://continuum.io/downloads and click the "Mac OS X 64-bit 
Python 2.7 Graphical Installer" button for Mac OS X or: "Linux 64-bit Python 2.7".

Install Anaconda using the default options.

Once Anaconda is installed, open a new shell and run the following on the command line:

``conda create -n hdf5json python=2.7 h5py``

Answer 'y' to the prompt, and the packages will be fetched.

In the same shell, run: ``source activate hdf5json``

Run: ``git clone --recursive https://github.com/HDFGroup/hdf5-json.git`` to download the source
tree.  Alternatively, in a browser go to: https://github.com/HDFGroup/hdf5-json and click 
the "Download ZIP" button (right side of page).  Download the zip file and extract to
the destination directory of your choice.  

Next, cd to the folder you extracted the source files to.

Run:
``python setup.py install``
to install the package.

 
Verification
-------------

To verify hdf5-json was installed correctly, you can convert a test HDF5 file to json and back.
 
Open a new shell (on Windows, run “Annaconda Command Prompt” from the start menu).

In this shell, run the following commands:

* source activate hdf5json (just: activate hdf5json on Windows)
* cd <hdf5-json installation directory>
* cd util
* python h5tojson.py ../data/hdf5/tall.h5 > tall.json
* python jsontoh5.py tall.json tall.h5

At this point the file tall.json should contain a JSON description of the original file and
the file tall.h5 should be an HDF5 equivalent to the original file.

Running:
``python testall.py``
will run the complete set of unit and integration tests.

 
