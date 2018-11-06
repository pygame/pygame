"""
temporary bindings for the SDL2 renderer
"""

import ctypes
import ctypes.util
from ctypes import c_char_p, c_char, c_void_p, c_int, c_uint, c_uint32, POINTER


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

# TEXTURE

SDL_CreateTexture = SDL2_dll.SDL_CreateTexture
SDL_CreateTexture.argtypes = (c_void_p, c_uint32, c_int, c_int, c_int)
SDL_CreateTexture.restype = c_void_p

SDL_DestroyTexture = SDL2_dll.SDL_DestroyTexture
SDL_DestroyTexture.argtypes = c_void_p,
SDL_DestroyTexture.restype = None

SDL_UpdateTexture = SDL2_dll.SDL_UpdateTexture
SDL_UpdateTexture.argtypes = (c_void_p, c_void_p, c_void_p, c_int)
SDL_UpdateTexture.restype = c_int
