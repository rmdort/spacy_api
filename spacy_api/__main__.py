import sys
from spacy_api import launch


def main():
    print(sys.argv)
    if len(sys.argv) != 2:
        print("Need exactly 1 argument: the filename for the server config.")
        sys.exit()

    launch.from_cfg(sys.argv[1])
