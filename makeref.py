#!/usr/bin/env python

import sys
import os
import subprocess


rst_dir = 'docs'
rst_source_dir = os.path.join(rst_dir, 'reST')
rst_build_dir = rst_dir
rst_doctree_dir = os.path.join(rst_build_dir, 'doctrees')
html_dir = 'docs'
c_header_dir = os.path.join('src', 'doc')

def Run():
    return subprocess.run(['sphinx-build',
                        '-b', 'html',
                        '-d', rst_doctree_dir,
                        '-D', 'headers_dest=%s' % (c_header_dir,),
                        '-D', 'headers_mkdirs=0',
                        rst_source_dir,
                        html_dir,]).returncode

if __name__ == '__main__':
    sys.exit(Run())
