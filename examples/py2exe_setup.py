from distutils.core import setup
import os, py2exe
import pygame2

def run ():
    # This will include the necessary pygame2 DLL dependencies, that are
    # available. By default they will be taken from the pygame2.dll
    # module directory.
    #
    # If you have your own SDL.dll, SDL_mixer.dll, ... to ship, remove
    # the line below.
    os.environ['PATH'] += ";%s" % pygame2.DLLPATH

    # ALWAYS include pygame2.sdl.rwops if you are using some pygame2.sdl
    # or related module that can load or save files or file-like
    # objects. Otherwise it might happen that pygame2.sdl.rwops is not
    # included in the final exe/Library.zip, which in turn causes the
    # application to fail badly.

    setup (windows = [os.path.join ('sdl', 'hello_world.py')], 
           options = { "py2exe" : { "bundle_files" : 1,
                                    "compressed" : 1,
                                    "includes" : [ "pygame2.sdl.rwops" ] }
                       },
           data_files = [ ('.', ["sdl/logo.gif"]), ('.', ["sdl/logo.bmp"]) ]
           )

if __name__ == "__main__":
    run ()
