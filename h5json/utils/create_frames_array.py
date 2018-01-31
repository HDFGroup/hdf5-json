#!/usr/bin/env python3

import sys
import json


def main():

    frames = []

    for arg in sys.argv[1:]:
        with open(arg, encoding='utf-8') as f:
            data = json.loads(f.read())
            frames.append(data)

    print(json.dumps(frames))

if __name__ == "__main__":
    main()
    