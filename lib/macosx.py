import os, sys

try:
    import MacOS
except:
    MacOS = None

from pygame.pkgdata import getResource

from pygame import sdlmain_osx

__all__ = ['Video_AutoInit']

def Video_AutoInit():
    """This is a function that's called from the c extension code
       just before the display module is initialized"""
    if MacOS and not MacOS.WMAvailable():
        if not sdlmain_osx.WMEnable():
            raise ImportError("Can not access the window manager.  Use py2app or execute with the pythonw script.")
    if not sdlmain_osx.RunningFromBundleWithNSApplication():
        try:
            default_icon_data = getResource('pygame_icon.tiff').read()
        except IOError:
            default_icon_data = None
        except NotImplementedError:
            default_icon_data = None

        sdlmain_osx.InstallNSApplication(default_icon_data)
    if (os.getcwd() == '/') and len(sys.argv) > 1:
        os.chdir(os.path.dirname(sys.argv[0]))
    return True
