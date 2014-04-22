# module mingwcfg.py

"""Manage the MinGW configuration file for setup.py"""

import os
import sys

if sys.version_info >= (3,):
    import functools
    open_ = functools.partial(open, encoding='utf-8')
else:
    open_ = open

directory = os.path.abspath(os.path.split(__file__)[0])
path = os.path.join(directory, 'mingw.cfg')

def write(mingw_root):
    cnf = open_(path, 'w')
    try:
        cnf.write(os.path.abspath(mingw_root))
        cnf.write('\n')
    finally:
        cnf.close()

def read():
    cnf = open_(path, 'r')
    try:
        for ln in cnf:
            return ln.strip()
    finally:
        cnf.close()
