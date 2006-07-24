#!/usr/bin/env python

'''Internal functions for dealing with Numeric, numpy and numarray.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

from ctypes import *
import sys

class _Numeric_PyArrayObject(Structure):
    _fields_ = [('ob_refcnt', c_int),
                ('ob_type', c_void_p),
                ('data', POINTER(c_char)),
                ('nd', c_int),
                ('dimensions', POINTER(c_int)),
                ('strides', POINTER(c_int)),
                ('base', c_void_p),
                ('descr', c_void_p),
                ('flags', c_uint),
                ('weakreflist', c_void_p)]

# Numeric flags constants
_CONTIGUOUS = 1
_OWN_DIMENSIONS = 2
_OWN_STRIDES = 4
_OWN_DATA = 8
_SAVESPACE = 16

# numarray constants
_MAXDIM = 40

class _numarray_PyArrayObject(Structure):
    _fields_ = [('ob_refcnt', c_int),
                ('ob_type', c_void_p),
                ('data', POINTER(c_char)),
                ('nd', c_int),
                ('dimensions', POINTER(c_int)),
                ('strides', POINTER(c_int)),
                ('base', c_void_p),
                ('descr', c_void_p),
                ('flags', c_uint),
                ('_dimensions', c_int * _MAXDIM),
                ('_strides', c_int * _MAXDIM),
                ('_data', c_void_p),
                ('_shadows', c_void_p),
                ('nstrides', c_int),
                ('byteoffset', c_long),
                ('bytestride', c_long),
                ('itemsize', c_long),
                ('byteorder', c_char)]

# Provide support for numpy and numarray in addition to Numeric.  To
# be compatible with Pygame, by default the module will be unavailable
# if Numeric is not available.  You can activate it to use any available
# array module by calling set_array_module().
try:
    import Numeric
    _array = Numeric
except ImportError:
    _array = None

def set_array_module(module=None):
    '''Set the array module to use; numpy, numarray or Numeric.

    If no arguments are given, every array module is tried and the
    first one that can be imported will be used.  The order of
    preference is numpy, numarray, Numeric.  You can determine which
    module was set with `get_array_module`.

    :Parameters:
        `module` : module or None
            Module to use.

    '''
    global _array
    if not module:
        for name in ('numpy', 'numarray', 'Numeric'):
            try:
                set_array_module(__import__(name, locals(), globals(), []))
            except ImportError:
                pass
    else:
        _array = module

def get_array_module():
    '''Get the currently set array module.

    If None is returned, no array module is set and the surfarray
    functions will not be useable.

    :rtype: module
    '''
    return _array

def _check_array():
    if not _array:
        raise ImportError, \
              'No array module set; use set_array_module if you want to ' + \
              'use numpy or numarray instead of Numeric.'

def _get_array_local_module(array):
    # Given an array, determine what array module it is from.  Note that
    # we don't require it to be the same module as _array, which is
    # only for returned arrays.

    # "strides" attribute is different in each module, so is hacky way
    # to check.
    if hasattr(array, 'strides'):
        import numpy
        return numpy
    elif hasattr(array, '_strides'):
        import numarray
        return numarray
    else:
        import Numeric
        return Numeric

def _array_from_string(string, bpp, shape, signed=False):
    if signed:
        typecode = (_array.Int8, _array.Int16, None, _array.Int32)[bpp-1]
    else:
        typecode = (_array.UInt8, _array.UInt16, None, _array.UInt32)[bpp-1]

    if _array.__name__ == 'numpy':
        return _array.fromstring(string, typecode).reshape(shape)
    elif _array.__name__ == 'numarray':
        return _array.fromstring(string, typecode, shape)
    elif _array.__name__ == 'Numeric':
        return _array.reshape(_array.fromstring(string, typecode), shape)

def _array_from_buffer(buffer, bpp, shape, signed=False):
    if signed:
        typecode = (_array.Int8, _array.Int16, None, _array.Int32)[bpp-1]
    else:
        typecode = (_array.UInt8, _array.UInt16, None, _array.UInt32)[bpp-1]

    if _array.__name__ == 'numpy':
        return _array.frombuffer(buffer, typecode).reshape(shape)

    elif _array.__name__ == 'Numeric':
        # Free old data and point to new data, updating dimension size and
        # clearing OWN_DATA flag so it doesn't get free'd.
        array = _array.array([0], typecode)
        array_obj = _Numeric_PyArrayObject.from_address(id(array))
        assert array_obj.flags & _OWN_DATA != 0
        try:
            libc = _get_libc()
            libc.free(array_obj.data)
        except OSError:
            pass # Couldn't find libc; accept a small memory leak
        array_obj.data = cast(buffer, POINTER(c_char))
        array_obj.dimensions.contents.value = reduce(lambda a,b:a*b, shape)
        array_obj.flags &= ~_OWN_DATA
        return _array.reshape(array, shape)

    elif _array.__name__ == 'numarray':
        # numarray PyArrayObject is source-compatible with Numeric,
        # but deallocation is managed via a Python buffer object.
        # XXX this fails under uncertain circumstances: reading the array
        # never works, writing works for some arrays and not others.
        array = _array.array([0], typecode)
        array_obj = _numarray_PyArrayObject.from_address(id(array))
        array_obj.dimensions.contents.value = reduce(lambda a,b:a*b, shape)
        array._data = buffer
        return _array.reshape(array, shape)

    else:
        assert False

def _get_libc():
    if sys.platform == 'windows':
        return cdll.msvcrt
    else:
        return cdll.load_version('c', 6)
