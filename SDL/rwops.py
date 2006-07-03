#!/usr/bin/env python

'''General interface for SDL to read and write data sources.

For files, use `SDL_RWFromFile`.  Other Python file-type objects can be
used with `SDL_RWFromObject`.  If another library provides a constant void
pointer to a contiguous region of memory, `SDL_RWFromMem` and
`SDL_RWFromConstMem` can be used.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from ctypes import *

import SDL.dll
import SDL.constants

_rwops_p = POINTER('SDL_RWops')
_seek_fn = CFUNCTYPE(c_int, _rwops_p, c_int, c_int)
_read_fn = CFUNCTYPE(c_int, _rwops_p, c_void_p, c_int, c_int)
_write_fn = CFUNCTYPE(c_int, _rwops_p, c_void_p, c_int, c_int)
_close_fn = CFUNCTYPE(c_int, _rwops_p)

class _hidden_mem_t(Structure):
    _fields_ = [('base', c_void_p),
                ('here', c_void_p),
                ('stop', c_void_p)]

class SDL_RWops(Structure):
    '''Read/write operations structure.

    :Ivariables:
        `seek` : function
            seek(context: `SDL_RWops`, offset: int, whence: int) -> int
        `read` : function
            read(context: `SDL_RWops`, ptr: c_void_p, size: int, maxnum: int) 
            -> int
        `write` : function
            write(context: `SDL_RWops`, ptr: c_void_p, size: int, num: int) ->
            int
        `close` : function
            close(context: `SDL_RWops`) -> int
        `type` : int
            Undocumented

    '''
    _fields_ = [('seek', _seek_fn),
                ('read', _read_fn),
                ('write', _write_fn),
                ('close', _close_fn),
                ('type', c_uint),
                ('_hidden_mem', _hidden_mem_t)]
SetPointerType(_rwops_p, SDL_RWops)

SDL_RWFromFile = SDL.dll.function('SDL_RWFromFile',
    '''Create an SDL_RWops structure from a file on disk.

    :Parameters:
        `file` : string
            Filename
        `mode` : string
            Mode to open the file with; as with the built-in function ``open``.

    :rtype: `SDL_RWops`
    ''',
    args=['file', 'mode'],
    arg_types=[c_char_p, c_char_p],
    return_type=POINTER(SDL_RWops),
    dereference_return=True,
    require_return=True)

SDL_RWFromMem = SDL.dll.function('SDL_RWFromMem',
    '''Create an SDL_RWops structure from a contiguous region of memory.

    :Parameters:
     - `mem`: ``c_void_p``
     - `size`: int

    :rtype: `SDL_RWops`
    ''',
    args=['mem', 'size'],
    arg_types=[c_void_p, c_int],
    return_type=POINTER(SDL_RWops),
    dereference_return=True,
    require_return=True)

SDL_RWFromConstMem = SDL.dll.function('SDL_RWFromConstMem',
    '''Create an SDL_RWops structure from a contiguous region of memory.

    :Parameters:
     - `mem`: ``c_void_p``
     - `size`: int

    :rtype: `SDL_RWops`
    :since: 1.2.7
    ''',
    args=['mem', 'size'],
    arg_types=[c_void_p, c_int],
    return_type=POINTER(SDL_RWops),
    dereference_return=True,
    require_return=True,
    since=(1,2,7))

""" These functions shouldn't be useful to Pythoners.
SDL_AllocRW = SDL.dll.function('SDL_AllocRW',
    '''Allocate a blank SDL_Rwops structure.

    :rtype: `SDL_RWops`
    '''
    args=[],
    arg_types=[],
    return_type=POINTER(SDL_RWops),
    dereference_return=True,
    require_return=True)

SDL_FreeRW = SDL.dll.function('SDL_FreeRW',
    '''Free a SDL_RWops structure.

    :param area: `SDL_RWops`
    '''
    args=['area'],
    arg_types=[POINTER(SDL_RWops)],
    return_type=None)
"""

# XXX Tested read from open() only so far
def SDL_RWFromObject(obj):
    '''Construct an SDL_RWops structure from a Python file-like object.

    The object must support the following methods in the same fashion as
    the builtin file object: 

        - ``read(len) -> data``
        - ``write(data)``
        - ``seek(offset, whence)``
        - ``close()``

    :Parameters:
     - `obj`: Python file-like object to wrap

    :rtype: `SDL_RWops`
    '''

    ctx = SDL_RWops()

    def _seek(context, offset, whence):
        obj.seek(offset, whence)
        return obj.tell()
    ctx.seek = _seek_fn(_seek)

    def _read(context, ptr, size, maximum):
        try:
            r = obj.read(maximum * size)
            memmove(ptr, r, len(r))
            return len(r) / size
        except:
            return -1
    ctx.read = _read_fn(_read)

    def _write(context, ptr, size, num):
        try:
            obj.write(string_at(ptr, size*num))
            return num
        except:
            return -1
    ctx.write = _write_fn(_write)

    def _close(context):
        obj.close()
    ctx.close = _close_fn(_close)
    return ctx

"""
# XXX Usefulness of the following using raw pointers?

def SDL_RWseek(ctx, offset, whence):
    return ctx.seek(ctx, offset, whence)

def SDL_RWtell(ctx):
    return ctx.seek(ctx, 0, SDL.constants.RW_SEEK_CUR)

def SDL_RWread(ctx, ptr, size, n):
    return ctx.read(ctx, ptr, size, n)

def SDL_RWwrite(ctx, ptr, size, n):
    return ctx.write(ctx, ptr, size, n)

def SDL_RWclose(ctx):
    return ctx.close(ctx)
"""

# XXX not implemented: SDL_Read{BL}E* and SDL_Write{BL}E*
