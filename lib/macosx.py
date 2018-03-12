import platform
import os
import sys
from pygame.pkgdata import getResource
from pygame import sdlmain_osx

__all__ = ['Video_AutoInit']

def Video_AutoInit():
    """Called from the base.c just before display module is initialized."""
    if 'Darwin' in platform.platform():
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
