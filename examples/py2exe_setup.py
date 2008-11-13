from distutils.core import setup
import os, py2exe
import pygame2

# This will include the necessary pygame2 DLL dependencies, that are available.
# By default they will be taken from the pygame2.dll module directory.
#
# If you have your own SDL.dll, SDL_mixer.dll, ... to ship, remove the line
# below.
os.environ['PATH'] += ";%s" % pygame2.DLLPATH


setup (windows = ['hello_world.py'], 
       options = { "py2exe": { "bundle_files" : 1 }},
       data_files = [ ('.', ["logo.gif"]), ('.', ["logo.bmp"]) ]
)
