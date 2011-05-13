#!/usr/bin/env python

import makerst

import sphinx

import sys
import os


rst_dir = 'reST'
rst_source_dir = os.path.join(rst_dir, 'source')
rst_build_dir = os.path.join(rst_dir, 'build')
rst_source_doc_dir = os.path.join(rst_source_dir, 'ref')
rst_doctree_dir = os.path.join(rst_build_dir, 'doctrees')
html_dir = 'docs'
c_header_dir = os.path.join('src', 'doc')

def Run():
    makerst.Run(rst_source_doc_dir)
    return sphinx.main([sys.argv[0],
                        '-b', 'html',
                        '-d', rst_doctree_dir,
                        '-D', 'headers_dest=%s' % (c_header_dir,),
                        '-D', 'headers_mkdir=0',
                        rst_source_dir,
                        html_dir,])

if __name__ == '__main__':
    sys.exit(Run())
