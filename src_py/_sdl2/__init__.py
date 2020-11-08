import sys

from importlib.machinery import ExtensionFileLoader

def unbulk_dyn_load_package_name(module_name, package_name, extension_name):
    foo = ExtensionFileLoader(package_name, extension_name).load_module()
    sys.modules[module_name] = foo
    return foo    

sdl2 = unbulk_dyn_load_package_name("_sdl2.sdl2", "_sdl2.sdl2", "_sdl2.pyd")
video = unbulk_dyn_load_package_name("_sdl2.video", "_sdl2.video", "_video.pyd")


from _sdl2.sdl2 import * # pylint: disable=wildcard-import; lgtm[py/polluting-import]
#from .audio import * # pylint: disable=wildcard-import; lgtm[py/polluting-import]
from _sdl2.video import * # pylint: disable=wildcard-import; lgtm[py/polluting-import]
