##############################################################################
# Copyright by The HDF Group.                                                #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of h5json. The full copyright notice, including          #
# terms governing use, modification, and redistribution, is contained in     #
# the file COPYING, which can be found at the root of the source code        #
# distribution tree. If you do not have access to this file, you may         #
# request a copy from help@hdfgroup.org.                                     #
##############################################################################
import sys
import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="HDF5/JSON validator",
        epilog="Copyright 2021 The HDF Group",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "jsonloc",
        nargs="+",
        help="JSON location (file(s) or folder(s)",
        metavar="JSON_LOC",
        type=Path,
    )
    parser.add_argument(
        "--schema",
        "-s",
        metavar="SCHEMA_DIR",
        help="Directory of HDF5/JSON schema files",
        type=Path,
    )
    parser.add_argument(
        "--docker-image",
        "-di",
        metavar="DOCKER_IMAGE",
        help="Docker image with the Ajv JSON Schema validator",
        default="hdf5json/ajv",
    )
    args = parser.parse_args()

    if not args.schema.is_dir():
        raise OSError(f"{args.schema} is not a directory or does not exist")
    elif not args.schema.joinpath("hdf5.schema.json").is_file():
        raise OSError(f"Main HDF5/JSON schema file not found in {args.schema}")
    full_schema_dir = str(args.schema.resolve())

    # Find all JSON files...
    json_table = dict()
    for p in args.jsonloc:
        if p.is_file():
            dir_ = str(p.resolve().parent)
            if dir_ not in json_table:
                json_table[dir_] = list()
            json_table[dir_].append(p.name)
        elif p.is_dir():
            dir_ = str(p.resolve())
            if dir_ not in json_table:
                json_table[dir_] = list()
            json_table[dir_].extend([f.name for f in p.glob("*.json")])
    if not json_table:
        print("No JSON files found.")
        return

    was_error = False
    for dir_ in json_table.keys():
        docker_cmd = [
            "docker",
            "run",
            "--rm",
            f"-v {full_schema_dir}:/schema",
            f"-v {dir_}:/data",
            args.docker_image,
        ]
        ajv_opts = [
            "--spec=draft2020",
            "-c ajv-formats",
            "--all-errors",
            "-s /schema/hdf5.schema.json",
            '-r "/schema/data*.schema.json"',
            "-r /schema/filters.schema.json",
            "-r /schema/attribute.schema.json",
            "-r /schema/group.schema.json",
        ]
        for f in json_table[dir_]:
            ajv_opts.append(f"-d /data/{f}")
        cmd = docker_cmd + ajv_opts
        ret = subprocess.run(" ".join(cmd), shell=True, check=False)
        # See https://docs.docker.com/engine/reference/run/#exit-status
        if 0 < ret.returncode < 125:
            was_error = True
    if was_error:
        sys.exit("ERROR: There were validation errors.")


if __name__ == "__main__":
    main()
