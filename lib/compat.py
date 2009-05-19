"""Python 2.x/3.x compatibility tools"""

import sys

__all__ = ['geterror', 'long_', 'xrange_', 'ord_', 'unichr_',
           'unicode_', 'raw_input_']

def geterror ():
    return sys.exc_info()[1]

try:
    long_ = long
except NameError:
    long_ = int

try:
    xrange_ = xrange
except NameError:
    xrange_ = range

def get_BytesIO():
    try:
        from cStringIO import StringIO as BytesIO
    except ImportError:
        from io import BytesIO
    return BytesIO

def ord_(o):
    try:
        return ord(o)
    except TypeError:
        return o

try:
    unichr_ = unichr
except NameError:
    unichr_ = chr

try:
    unicode_ = unicode
except NameError:
    unicode_ = str

try:
    raw_input_ = raw_input
except NameError:
    raw_input_ = input
