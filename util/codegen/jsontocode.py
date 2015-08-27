import sys
import os.path as osp
import json
import argparse

# Modify system path to import needed modules
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), osp.pardir,
                                     osp.pardir, 'lib')))
from idl import IdlCode
from matlab import MCode
from python import PyCode

parser = argparse.ArgumentParser(
    description='Generate source code from HDF5/JSON that recreates HDF5 file',
    epilog='Copyright 2015 The HDF Group')
parser.add_argument('h5json_file',
                    help='HDF5/JSON file from which to generate source code')
parser.add_argument(
    'h5_file',
    help='Name of HDF5 file to be created by the generated code')
parser.add_argument('--lang', '-l', choices=['python', 'matlab', 'idl'],
                    default='python', help='Source code language')
args = parser.parse_args()

with open(args.h5json_file) as f:
    h5json = json.load(f)

if args.lang == 'python':
    codegen = PyCode(h5json, args.h5_file)
elif args.lang == 'matlab':
    codegen = MCode(h5json, args.h5_file)
elif args.lang == 'idl':
    codegen = IdlCode(h5json, args.h5_file)

print codegen.get_code()
