import sys
from spacy_api.launch import from_config


def main():
    print(sys.argv)
    if len(sys.argv) != 2:
        print("Need exactly 1 argument: the filename for the server config.")
        sys.exit()

    from_config(sys.argv[1])
