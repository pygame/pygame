#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from ctypes import *

# numpy
try:
    import numpy
    _numpy_typemap = {
        c_ubyte: numpy.UInt8,
        c_ushort: numpy.UInt16,
        c_uint: numpy.UInt32
    }
    _have_numpy = True
except ImportError:
    _have_numpy = False

class SDL_array:
    def __init__(self, ptr, count, ctype):
        '''Construct an array at memory location `ptr` with `count` elements
        of type `ctype`.

        :Parameters:
            `ptr` : ctypes.Array, POINTER(ctype) or POINTER(ctypes.Array)
                Starting point of the array space.  Don't use c_void_p; this
                will not cast correctly.  If `ptr` is None, the array
                will be created (filled with random data).
            `count` : int
                Number of elements in the array.
            `ctype` : type
                ctypes type if each element, e.g., c_ubyte, c_int, etc.

        '''
        count = int(count)
        if not ptr:
            ptr = (ctype * count)()
        self.ptr = ptr
        self.count = count
        self.ctype = ctype
        self._ctypes_array = None

    def as_bytes(self):
        '''Access the array as raw bytes, regardless of the underlying
        data type.

        This can be useful, for example, in accessing a 32-bit colour
        buffer by individual components rather than the encoded pixel.

        :rtype: SDL_array
        '''
        return SDL_array(self.ptr, (self.count * sizeof(self.ctype)), c_ubyte)

    def as_ctypes(self):
        '''Access the array as a ctypes array.

        :rtype: ctypes.Array
        '''
        if not self._ctypes_array:
            self._ctypes_array = \
                cast(self.ptr, POINTER(self.ctype * self.count)).contents
        return self._ctypes_array

    def as_numpy(self):
        '''Access the array as a numpy array.  
        
        The numpy array shares the same underlying memory buffer, so
        changes are immediate, and you can use the numpy array as you would
        normally.  To set the entire contents of the array at once, use a
        ``[:]`` slice.

        If numpy is not installed, an ImportError will be raised.

        :rtype: numpy.ndarray
        '''
        if not _have_numpy:
            raise ImportError, 'numpy could not be imported'
        if self.ctype not in _numpy_typemap:
            raise TypeError, '%s has no numpy compatible type' % self.ctype
        return numpy.frombuffer(self.as_ctypes(), _numpy_typemap[self.ctype])

    def to_string(self):
        '''Return a string with the contents of this array.

        :rtype: string
        '''
        return ''.join([chr(c) for c in self.as_bytes().as_ctypes()])

    def __repr__(self):
        return 'SDL_array(ctype=%s, count=%r)' % (self.ctype, self.count)

    def __len__(self):
        return self.count

    def __getitem__(self, key):
        if type(key) is slice:
            if key.step:
                raise TypeError, 'slice step not supported'
            return self.as_ctypes()[key.start:key.stop]
        else:
            return self.as_ctypes()[key]

    def __setitem__(self, key, value):
        if type(key) is slice:
            if key.step:
                raise TypeError, 'slice step not supported'
            self.as_ctypes()[key.start:key.stop] = value
        else:
            self.as_ctypes()[key] = value

def to_ctypes(values, count, ctype):
    '''Create a ctypes array of the given count and type, with the contents
    of sequence `values`.

    :Parameters:
     - `values`: sequence of length `count`, or SDL_array instance, or
       ctypes.Array, or POINTER(ctypes.Array)
     - `count`: int
     - `ctype`: type

    '''

    # Convert SDL_array instances to ctypes array
    if isinstance(values, SDL_array):
        values = values.as_ctypes()

    # Cast ctypes array to correct type if necessary
    if isinstance(values, Array):
        if values._type_ is ctype:
            return values
        else:
            return cast(values, POINTER(ctype * count)).contents

    # Otherwise assume sequence
    return (ctype * count)(*values)
