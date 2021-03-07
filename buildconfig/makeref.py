#!/usr/bin/env python

import sys
import os
import subprocess


rst_dir = 'docs'
rst_source_dir = os.path.join(rst_dir, 'reST')
rst_build_dir = rst_dir
rst_doctree_dir = os.path.join(rst_build_dir, 'doctrees')
html_dir = 'docs'
c_header_dir = os.path.join('src_c', 'doc')

def Run():
    try:
        subprocess_args = [sys.executable, '-m', 'sphinx',
                            '-b', 'html',
                            '-d', rst_doctree_dir,
                            '-D', 'headers_dest=%s' % (c_header_dir,),
                            '-D', 'headers_mkdirs=0',
                            rst_source_dir,
                            html_dir,]
        print("executing sphinx in subprocess with args:", subprocess_args)
        return subprocess.run(subprocess_args).returncode
    except:
        print('---')
        print('Have you installed sphinx?')
        print('---')
        raise
if __name__ == '__main__':
    sys.exit(Run())
