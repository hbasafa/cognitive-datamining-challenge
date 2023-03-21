import os

DIRS = 1
FILES = 2
NFILES = 3


def get_files(path):
    files = list(os.fwalk(path))[0][FILES]
    return files


def create_directory(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
