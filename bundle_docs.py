#! /usr/bin/env python

"""Tar-zip the Pygame documents and examples

Run this script from the Pygame source root directory.
"""

import os
import tarfile
import re

def add_files(bundle, root, alias, file_names):
    for file_name in file_names:
        file_alias = os.path.join(alias, file_name)
        print " ", file_name, "-->", file_alias
        bundle.add(os.path.join(root, file_name), file_alias)

def add_directory(bundle, root, alias):
    reject_dirs = re.compile(r'(.svn)$')
    # Since it is the file extension that is of interest the reversed
    # file name is checked.
    reject_files_reversed = re.compile(r'((~.*)|(cyp\..*))')
    for sub_root, directories, files in os.walk(root):
        directories[:] = [d for d in directories if reject_dirs.match(d) is None]
        files[:] = [f for f in files if reject_files_reversed.match(f[-1::-1]) is None]
        sub_alias = os.path.join(alias, sub_root[len(root)+1:])
        add_files(bundle, sub_root, sub_alias, files)
        
def main():
    bundle_name_elements = ['pygame', 'docs']
    setup = open('setup.py', 'r')
    try:
        match = re.search(r'"version":[ \t]+"([0-9]+\.[0-9]+)\.[^"]+"', setup.read())
    finally:
        setup.close()
    if match is None:
        print "*** Unable to find Pygame version in setup.py"
        version = ''
    else:
        version = '-%s' % match.group(1)
    bundle_name = 'pygame%s-docs-and-examples.tar.gz' % version
    print "Creating bundle", bundle_name
    bundle = tarfile.open(bundle_name, 'w:gz')
    try:
        root = os.path.abspath('.')
        alias = 'pygame'
        add_files(bundle, root, alias, ['LGPL', 'readme.html', 'install.html'])
        add_directory(bundle, os.path.join(root, 'docs'), os.path.join(alias, 'docs'))
        add_directory(bundle, os.path.join(root, 'examples'), os.path.join(alias, 'examples'))
        print "\nFinished", bundle_name
    finally:
        bundle.close()

if __name__ == '__main__':
    main()
