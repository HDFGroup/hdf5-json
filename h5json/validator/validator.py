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
import importlib.resources
import json
import jsonschema
from h5json import schema


def prepare_validator() -> jsonschema.Draft202012Validator:
    """Return a configured jsonschema.Draft202012Validator instance."""
    with importlib.resources.open_text(schema, "hdf5.schema.json") as f:
        h5schema = json.load(f)

    schema_store = dict()
    schema_components = [
        "attribute.schema.json",
        "filters.schema.json",
        "group.schema.json",
        "datatypes.schema.json",
        "dataspaces.schema.json",
        "dataset.schema.json",
    ]
    for sc in schema_components:
        with importlib.resources.open_text(schema, sc) as f:
            temp = json.load(f)
        schema_store[temp["$id"]] = temp
    resolver = jsonschema.RefResolver(h5schema["$id"], h5schema, store=schema_store)
    return jsonschema.Draft202012Validator(h5schema, resolver=resolver)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HDF5/JSON validator",
        epilog="Copyright 2021 The HDF Group",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "jsonloc",
        nargs="+",
        help="JSON location (files or folders)",
        metavar="JSON_LOC",
        type=Path,
    )
    parser.add_argument(
        "--stop",
        "-s",
        action="store_true",
        help="Stop after first HDF5/JSON file failed validation",
    )
    args = parser.parse_args()

    # Find all JSON files for validation...
    json_files = list()
    for p in args.jsonloc:
        if p.is_file():
            json_files.append(p)
        elif p.is_dir():
            json_files.extend([f for f in p.glob("*.json")])
    if not json_files:
        sys.exit("No JSON files for validation found.")

    validator = prepare_validator()

    # Validate HDF5/JSON files...
    valid_errors = False
    for h5j in json_files:
        print(f"Validating {str(h5j)} ... ", end="")
        try:
            with h5j.open() as f:
                inst = json.load(f)
            validator.validate(inst)
            print("pass")
        except jsonschema.exceptions.ValidationError:
            print("FAIL")
            valid_errors = True
            inst_name = str(h5j)
            print(f"HDF5/JSON validation failed for {inst_name}", file=sys.stderr)
            for err in validator.iter_errors(inst):
                print(f"{inst_name} ---> {err}", file=sys.stderr)
            if args.stop:
                sys.exit("HDF5/JSON validation failed.")
    if valid_errors:
        sys.exit("HDF5/JSON validation failed.")


if __name__ == "__main__":
    main()
