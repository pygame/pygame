"""
A python helper script to install built (cached) mac deps into /usr/local
"""

import shutil
import sys
from pathlib import Path


def rmpath(path: Path, verbose: bool = False):
    """
    Tries to remove a path of any kind
    """
    if path.is_symlink():
        if verbose:
            print(f"- Removing existing symlink at '{path}'")

        path.unlink()

    elif path.is_file():
        if verbose:
            print(f"- Removing existing file at '{path}'")

        path.unlink()

    elif path.is_dir():
        if verbose:
            print(f"- Removing existing directory at '{path}'")

        shutil.rmtree(path)


def symtree(srcdir: Path, destdir: Path, verbose: bool = False):
    """
    This function creates symlinks pointing to srcdir, from destdir, such that
    existing folders and files in the tree of destdir are retained
    """
    if not destdir.is_dir():
        # dest dir does not exist at all, create dir symlink
        rmpath(destdir, verbose)
        if verbose:
            print(
                f"- Creating directory symlink from '{destdir}' pointing to '{srcdir}'"
            )

        destdir.symlink_to(srcdir)
        return

    for path in srcdir.glob("*"):
        destpath = destdir / path.name
        if path.is_dir():
            symtree(path, destpath, verbose)
        else:
            rmpath(destpath, verbose)
            if verbose:
                print(f"- Creating file symlink from '{destpath}' pointing to '{path}'")

            destpath.symlink_to(path)


symtree(Path(sys.argv[1]), Path("/"), verbose=True)
