"""Python 2.x/3.x compatibility tools"""

import sys

__all__ = ['geterror']

def geterror ():
    return sys.exc_info()[1]

try:
    long_ = long
except NameError:
    def long_(n):
        int(n)
