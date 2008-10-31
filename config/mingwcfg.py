# module mingwcfg.py

"""Manage the MinGW configuration file for setup.py"""

import os

directory = os.path.abspath(os.path.split(__file__)[0])
path = os.path.join(directory, 'mingw.cfg')

def write(mingw_root):
    cnf = open(path, 'w')
    try:
        cnf.write(os.path.abspath(mingw_root))
        cnf.write('\n')
    finally:
        cnf.close

def read():
    cnf = open(path, 'r')
    try:
        for ln in cnf:
            return ln.strip()
    finally:
        cnf.close()
