#!/usr/bin/env python

'''Log differences between SDL headers in different versions.

Checks out every copy of SDL known (define in `versions`), and diffs
each incremental version, writing to ``diff_1.2.x.txt``, where x is the
newer version.

The format of the file is concatenated diff context dumps of the header
files, as well as a note of which files are new to that version.
Most of the file is garbage (changes to comments, license, keyword
expansion, etc), but it's easy enough to skim through for structure and
function changes.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import os
import subprocess
import sys

# 1.2.1 to 1.2.11
versions = ['1.2.%d' % (i+1) for i in range(11)]

# Checkout under support/
working_base = os.path.dirname(sys.argv[0])

# Write diffs to support/
diff_base = os.path.dirname(sys.argv[0])

def working(version):
    return os.path.join(working_base, 'SDL-%s/include' % version)

def checkout(version):
    url = 'svn://libsdl.org/tags/SDL/release-%s/include' % version
    if not os.path.exists(working(version)):
        print 'Checking out %s' % version
        subprocess.call('svn co %s %s' % (url, working(version)), shell=True)

def compare(v1, v2):
    diff_file = open(os.path.join(diff_base, 'diff_%s.txt' % v2), 'w')
    for file in os.listdir(working(v2)):
        if file[-2:] == '.h':
            print >> diff_file, '\n%s\n%s' % (file, '-' * 80)
            if os.path.exists(os.path.join(working(v1), file)):
                output = subprocess.Popen('diff -c %s %s' % \
                            (os.path.join(working(v1), file), 
                             os.path.join(working(v2), file)),
                            shell=True, stdout=subprocess.PIPE).stdout
                diff_file.write(output.read())
                output.close()
            else:
                print >> diff_file, '\nNew file %s in version %s\n' % (file, v2)

if __name__ == '__main__':
    for version in versions:
        checkout(version)
    for i in range(len(versions) - 1):
        v1 = versions[i]
        v2 = versions[i+1]
        print 'Comparing %s and %s' % (v1, v2)
        compare(v1, v2)
