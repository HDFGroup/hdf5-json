from setuptools import setup
from codecs import open
from os import path
import versioneer

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="h5json",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="HDF5/JSON Tools",
    long_description=long_description,
    url="https://github.com/HDFGroup/hdf5-json",
    author="John Readey",
    author_email="jreadey@hdfgroup.org",
    license="BSD",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="json hdf5 numpy array data datacube",
    packages=[
        "h5json",
        "h5json.h5tojson",
        "h5json.jsontoh5",
        "h5json.validator",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.16.6",
        "h5py>=3.0",
        "importlib_resources;python_version<'3.9'",
        "jsonschema>=4.4.0",
    ],
    setup_requires=["pkgconfig"],
    zip_safe=False,
    extras_require={
        "dev": ["check-manifest"],
        "test": ["coverage"],
    },
    package_data={
        "h5json.schema": ["*.json"]
    },
    entry_points={
        "console_scripts": [
            "h5tojson = h5json.h5tojson.h5tojson:main",
            "jsontoh5 = h5json.jsontoh5.jsontoh5:main",
            "h5jvalidate = h5json.validator.validator:main",
        ]
    },
)
