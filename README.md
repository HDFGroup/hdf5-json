[![CI](https://github.com/HDFGroup/hdf5-json/actions/workflows/main.yml/badge.svg)](https://github.com/HDFGroup/hdf5-json/actions/workflows/main.yml)
[![Dependency Review](https://github.com/HDFGroup/hdf5-json/actions/workflows/dependency-review.yml/badge.svg)](https://github.com/HDFGroup/hdf5-json/actions/workflows/dependency-review.yml)
[![CodeQL](https://github.com/HDFGroup/hdf5-json/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/HDFGroup/hdf5-json/actions/workflows/codeql-analysis.yml)
[![Documentation Status](https://readthedocs.org/projects/hdf5-json/badge/?version=latest)](https://hdf5-json.readthedocs.io/en/latest/?badge=latest)

# h5json

Specification and tools for representing HDF5 in JSON.

## Introduction

This repository contains a specification (as BNF grammar and JSON Schema), and a
package for working with HDF5 content in JSON. The package CLI utilities can be
used to convert any HDF5 file to JSON or from a JSON file (using the
specification described here) to HDF5.

The package is also useful for any Python application that needs to translate between HDF5 objects and JSON
serializations. In addition to the utilities provided, the package is used by the [HDF
Server](https://www.hdfgroup.org/solutions/highly-scalable-data-service-hsds/) (a RESTful web service for HDF5), and [HDF Product Designer](https://wiki.earthdata.nasa.gov/display/HPD/HDF+Product+Designer) (an application for planning HDF5 file content).

## Websites

* Main website: http://www.hdfgroup.org
* Source code: https://github.com/HDFGroup/hdf5-json
* HDF Forum: https://forum.hdfgroup.org/c/hsds
* Documentation: https://hdf5-json.readthedocs.org

## Reporting bugs (and general feedback)

Create new issues at http://github.com/HDFGroup/hdf5-json/issues for any problems you find.

For general questions/feedback, please post on the [HDF Forum](https://forum.hdfgroup.org/c/hsds).
