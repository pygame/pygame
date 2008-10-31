#! /usr/bin/env python
"""Tar-zip the Pygame documents and examples

Run this script from the Pygame source root directory.
"""

import os, sys
import tarfile
import re

sys.path.append (os.path.pardir)

# Exclude list that cannot be easily regexp'd.
excludes = [ "create_cref.py", "create_htmlref.py", "create_doc.py" ]

def add_files (bundle, root, alias, file_names):
    for file_name in file_names:
        file_alias = os.path.join (alias, file_name)
        print ("  %s --> %s" % (file_name, file_alias))
        bundle.add (os.path.join (root, file_name), file_alias)

def add_directory (bundle, root, alias):
    reject_dirs = re.compile (r'(src)|(.svn)$')

    # Since it is the file extension that is of interest the reversed
    # file name is checked.
    reject_files_reversed = re.compile(r'((~.*)|(cyp\..*)|(lmx\..*))')
    for sub_root, directories, files in os.walk (root):
        directories[:] = [d for d in directories if reject_dirs.match(d) is None]
        files[:] = [f for f in files \
                    if reject_files_reversed.match(f[-1::-1]) is None and \
                    f not in excludes]
        sub_alias = os.path.join (alias, sub_root[len (root)+1:])
        add_files (bundle, sub_root, sub_alias, files)

def main():
    bundle_name_elements = ['pygame', 'docs']
    fp = open ("setup.py", "r")
    try:
        match = re.search (r'VERSION = "([0-9]+\.[0-9]+\.[0-9])"', fp.read ())
    finally:
        fp.close ()
    if match is None:
        print ("*** Unable to find version in setup.py")
        version = ''
    else:
        version = '-%s' % match.group(1)

    bundle_name = "dist/pygame2%s-docs-and-examples.tar.gz" % version
    print ("Creating bundle %s" % bundle_name)

    if not os.path.exists ("dist"):
        os.mkdir ("dist")
        
    bundle = tarfile.open (bundle_name, "w:gz")
    try:
        root = os.path.abspath (".")
        alias = "pygame2"
        add_files (bundle, root, alias, ['README.txt', ])
        add_directory (bundle, os.path.join(root, 'doc'),
                       os.path.join(alias, 'doc'))
        print ("\nFinished: %s" % bundle_name)
    finally:
        bundle.close()

if __name__ == '__main__':
    main()
