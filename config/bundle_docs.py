#! /usr/bin/env python
"""Tar-zip the Pygame documents.

Run this script from the Pygame source root directory.
"""

import os, sys
import tarfile, zipfile
import re

sys.path.append (os.path.pardir)

# Exclude list that cannot be easily regexp'd.
excludes = [ "create_cref.py", "create_rstref.py", "create_doc.py",
             "conf.py", "Makefile" ]

def add_files (bundle, root, alias, file_names, iszip):
    for file_name in file_names:
        file_alias = os.path.join (alias, file_name)
        print ("%s --> %s" % (file_name, file_alias))
        if iszip:
            bundle.write (os.path.join (root, file_name), file_alias)
        else:
            bundle.add (os.path.join (root, file_name), file_alias)

def add_directory (bundle, root, alias, iszip):
    reject_dirs = re.compile (r'(sphinx)|(src)|(.svn)$')

    # Since it is the file extension that is of interest the reversed
    # file name is checked.
    reject_files_reversed = re.compile(r'((~.*)|(cyp\..*)|(lmx\..*)|(tsr\..*))')
    for sub_root, directories, files in os.walk (root):
        directories[:] = [d for d in directories if reject_dirs.match(d) is None]
        files[:] = [f for f in files \
                    if reject_files_reversed.match(f[-1::-1]) is None and \
                    f not in excludes]
        sub_alias = os.path.join (alias, sub_root[len (root)+1:])
        add_files (bundle, sub_root, sub_alias, files, iszip)

def main():
    fp = open ("setup.py", "r")
    try:
        match = re.search (r'VERSION = "([0-9]+\.[0-9]+\.[0-9]+(\-\w+){0,1})"',
                           fp.read ())
    finally:
        fp.close ()
    if match is None:
        print ("*** Unable to find version in setup.py")
        version = ''
    else:
        version = '-%s' % match.group(1)

    bundle_name = os.path.join ("dist", "pygame2%s-docs.tar.gz" % version)
    zip_name = os.path.join ("dist", "pygame2%s-docs.zip" % version)
    print ("Creating bundle %s" % bundle_name)

    if not os.path.exists ("dist"):
        os.mkdir ("dist")
    
    bundle = tarfile.open (bundle_name, "w:gz")
    zipbundle = zipfile.ZipFile (zip_name, "w", zipfile.ZIP_DEFLATED)
    
    try:
        root = os.path.abspath (".")
        alias = "pygame2"
        add_files (bundle, root, alias, ['README.txt', ], False)
        add_files (zipbundle, root, alias, ['README.txt', ], True)
        add_directory (bundle, os.path.join(root, 'doc'),
                       os.path.join(alias, 'doc'), False)
        add_directory (zipbundle, os.path.join(root, 'doc'),
                       os.path.join(alias, 'doc'), True)
        print ("\nFinished: %s\n          %s\n" % (bundle_name, zip_name))
    finally:
        bundle.close()
        zipbundle.close()

if __name__ == '__main__':
    main()
