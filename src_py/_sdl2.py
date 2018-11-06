"""
temporary bindings for the SDL2 renderer
"""

import ctypes
import ctypes.util
from ctypes import c_char_p, c_char, c_void_p, c_int, c_uint, c_uint32, c_uint8, POINTER, Structure, c_size_t
from ctypes import py_object, cast, POINTER, byref, c_byte


SDL2_dll = ctypes.CDLL(ctypes.util.find_library('SDL2'))

# WINDOW

SDL_WINDOWPOS_UNDEFINED = 0x1FFF0000
SDL_WINDOWPOS_CENTERED = 0x2FFF0000

SDL_CreateWindow = SDL2_dll.SDL_CreateWindow
SDL_CreateWindow.restype = ctypes.c_void_p
SDL_CreateWindow.argtypes = (c_char_p, c_int, c_int, c_int, c_int, c_uint32)
SDL_DestroyWindow = SDL2_dll.SDL_DestroyWindow
SDL_DestroyWindow.argtypes = (c_void_p,)
SDL_DestroyWindow.restype = None

# RENDERER

SDL_RENDERER_SOFTWARE = 0x00000001
SDL_RENDERER_ACCELERATED = 0x00000002
SDL_RENDERER_PRESENTVSYNC = 0x00000004
SDL_RENDERER_TARGETTEXTURE = 0x00000008

SDL_CreateRenderer = SDL2_dll.SDL_CreateRenderer
SDL_CreateRenderer.argtypes = (c_void_p, c_int, c_uint32)
SDL_CreateRenderer.restype = c_void_p
SDL_DestroyRenderer = SDL2_dll.SDL_DestroyRenderer
SDL_DestroyRenderer.argtypes = c_void_p,
SDL_DestroyRenderer.restype = None

SDL_RenderCopy = SDL2_dll.SDL_RenderCopy
SDL_RenderCopy.argtypes = (c_void_p, c_void_p, c_void_p, c_void_p)
SDL_RenderCopy.restype = c_int

SDL_RenderClear = SDL2_dll.SDL_RenderClear
SDL_RenderClear.argtypes = c_void_p,
SDL_RenderClear.restype = c_int

SDL_RenderPresent = SDL2_dll.SDL_RenderPresent
SDL_RenderPresent.argtypes = c_void_p,
SDL_RenderPresent.restype = None

SDL_SetRenderDrawColor = SDL2_dll.SDL_SetRenderDrawColor
SDL_SetRenderDrawColor.argtypes = (c_void_p, c_uint8, c_uint8, c_uint8, c_uint8)
SDL_SetRenderDrawColor.restype = c_int

SDL_GetRenderDrawColor = SDL2_dll.SDL_GetRenderDrawColor
SDL_GetRenderDrawColor.argtypes = (c_void_p, POINTER(c_uint8), POINTER(c_uint8), POINTER(c_uint8), POINTER(c_uint8))
SDL_GetRenderDrawColor.restype = c_int

# TEXTURE

SDL_CreateTexture = SDL2_dll.SDL_CreateTexture
SDL_CreateTexture.argtypes = (c_void_p, c_uint32, c_int, c_int, c_int)
SDL_CreateTexture.restype = c_void_p

SDL_CreateTextureFromSurface = SDL2_dll.SDL_CreateTextureFromSurface
SDL_CreateTextureFromSurface.argtypes = (c_void_p, c_void_p)
SDL_CreateTextureFromSurface.restype = c_void_p

SDL_DestroyTexture = SDL2_dll.SDL_DestroyTexture
SDL_DestroyTexture.argtypes = c_void_p,
SDL_DestroyTexture.restype = None

SDL_UpdateTexture = SDL2_dll.SDL_UpdateTexture
SDL_UpdateTexture.argtypes = (c_void_p, c_void_p, c_void_p, c_int)
SDL_UpdateTexture.restype = c_int

class SDL_Rect(Structure):
    _fields_ = [
        ('x', c_int),
        ('y', c_int),
        ('w', c_int),
        ('h', c_int),
    ]

# SURFACE

class _pgSurfaceObject(Structure):
    _fields_ = [
        ('HEAD', c_byte * object.__basicsize__),
        ('surf', c_void_p), # SDL_Surface *
        #owner
        #subsurface
        #weakreflist
        #locklist
        #dependency
    ]

def get_surface_ptr(surface):
    return _pgSurfaceObject.from_address(id(surface)).surf
