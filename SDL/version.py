#!/usr/bin/env python

'''Functions related to the SDL shared library version.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from ctypes import *

import SDL.dll

class SDL_version(Structure):
    '''Version structure.

    :Ivariables:
        `major` : int
            Major version number
        `minor` : int
            Minor version number
        `patch` : int
            Patch revision number

    '''
    _fields_ = [('major', c_ubyte),
                ('minor', c_ubyte),
                ('patch', c_ubyte)]

    def __repr__(self):
        return '%d.%d.%d' % \
            (self.major, self.minor, self.patch)

    def is_since(self, required):
        if hasattr(required, 'major'):
            return self.major >= required.major and \
                   self.minor >= required.minor and \
                   self.patch >= required.patch
        else:
            return self.major >= required[0] and \
                   self.minor >= required[1] and \
                   self.patch >= required[2]

def SDL_VERSIONNUM(major, minor, patch):
    '''Turn the version numbers into a numeric value.

    For example::

        >>> SDL_VERSIONNUM(1, 2, 3)
        1203

    :Parameters:
     - `major`: int
     - `minor`: int
     - `patch`: int

    :rtype: int
    '''
    return x * 1000 + y * 100 + z

SDL_Linked_Version = SDL.dll.function('SDL_Linked_Version',
    '''Get the version of the dynamically linked SDL library.

    :rtype: `SDL_version`
    ''',
    args=[],
    arg_types=[],
    return_type=POINTER(SDL_version),
    dereference_return=True,
    require_return=True)

def SDL_VERSION_ATLEAST(major, minor, patch):
    '''Determine if the SDL library is at least the given version.
    
    :Parameters:
     - `major`: int
     - `minor`: int
     - `patch`: int

    :rtype: bool
    '''
    v = SDL_Linked_Version()
    return SDL_VERSIONNUM(v.major, v.minor, v.patch) >= \
           SDL_VERSIONNUM(major, minor, patch)

# SDL_VERSION and SDL_COMPILEDVERSION not implemented as there is no
# sensible mapping to compiled version numbers.
