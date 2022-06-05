#! /usr/bin/env python

"""Tar-zip the pygame documents and examples.

Run this script from the pygame source root directory.
"""

import os
import tarfile
import re


def add_files(bundle, root, alias, file_names):
    """Add files to the bundle."""
    for file_name in file_names:
        file_alias = os.path.join(alias, file_name)
        print(f"  {file_name} --> {file_alias}")
        bundle.add(os.path.join(root, file_name), file_alias)


def add_directory(bundle, root, alias):
    """Recursively add a directory, subdirectories, and files to the bundle."""
    reject_dirs = re.compile(r'(.svn)$')

    # Since it is the file extension that is of interest the reversed
    # file name is checked.
    reject_files_reversed = re.compile(r'((~.*)|(cyp\..*))')

    for sub_root, directories, files in os.walk(root):
        directories[:] = [
            d for d in directories if reject_dirs.match(d) is None]
        files[:] = [
            f for f in files if reject_files_reversed.match(f[-1::-1]) is None]

        sub_alias = os.path.join(alias, sub_root[len(root)+1:])
        add_files(bundle, sub_root, sub_alias, files)


def main():
    """Create a tar-zip file containing the pygame documents and examples."""
    with open('setup.py') as setup:
        match = re.search(r'"version":[ \t]+"([0-9]+\.[0-9]+)\.[^"]+"',
                          setup.read())

    if match is None:
        print("*** Unable to find the pygame version data in setup.py")
        version = ''
    else:
        version = f'-{match.group(1)}'

    bundle_name = f'pygame{version}-docs-and-examples.tar.gz'
    print(f"Creating bundle {bundle_name}")

    with tarfile.open(bundle_name, 'w:gz') as bundle:
        root = os.path.abspath('.')
        alias = 'pygame'

        add_files(bundle, root, alias, ['README.rst'])
        add_directory(bundle, os.path.join(root, 'docs'),
                      os.path.join(alias, 'docs'))
        add_directory(bundle, os.path.join(root, 'examples'),
                      os.path.join(alias, 'examples'))

    print(f"\nFinished {bundle_name}")


if __name__ == '__main__':
    main()
