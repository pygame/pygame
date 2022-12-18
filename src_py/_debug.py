"""Debug functionality that allows for more useful issue reporting
"""

import sys
import traceback
import importlib
from typing import Tuple, Optional, TypeAlias

ImportResult: TypeAlias = Tuple[str, bool, Optional[function]]


def str_from_tuple(version_tuple):
    if version_tuple is None:
        return "None"

    strs = [str(i) for i in version_tuple]
    return ".".join(strs)


def attempt_import(module, function_name, output_str=""):
    try:
        mod = importlib.import_module(module)
        i = getattr(mod, function_name)
        success = True
    except (ImportError, AttributeError):
        i = None
        output_str += f"There was a problem with {module} import\n"
        output_str += "A dummy value will be returned for the version\n"
        output_str += traceback.format_exc() + "\n" + "=" * 20 + "\n"
        success = False

    return (output_str, success, i)


def print_debug_info(filename=None):
    debug_str = ""

    # keyword for compat with getters
    def dummy_return(linked=True):
        # lint complains about unused keyword
        if linked:
            return (-1, -1, -1)
        return (-1, -1, -1)

    from pygame.base import get_sdl_version

    debug_str, *mixer = attempt_import(
        "pygame.mixer", "get_sdl_mixer_version", debug_str
    )
    if not mixer[0]:
        get_sdl_mixer_version = dummy_return
    else:
        get_sdl_mixer_version = mixer[1]

    debug_str, *font = attempt_import("pygame.font", "get_sdl_ttf_version", debug_str)
    if not font[0]:
        get_sdl_ttf_version = dummy_return
    else:
        get_sdl_ttf_version = font[1]

    debug_str, *image = attempt_import(
        "pygame.image", "get_sdl_image_version", debug_str
    )
    if not image[0]:
        get_sdl_image_version = dummy_return
    else:
        get_sdl_image_version = image[1]

    debug_str, *freetype = attempt_import("pygame.freetype", "get_version", debug_str)
    if not freetype[0]:
        ft_version = dummy_return
    else:
        ft_version = freetype[1]

    from pygame.version import ver

    import platform

    debug_str += f"Platform:\t\t{platform.platform()}\n"

    debug_str += f"System:\t\t\t{platform.system()}\n"

    debug_str += f"System Version:\t\t{platform.version()}\n"

    debug_str += f"Processor:\t\t{platform.processor()}\n"

    debug_str += (
        f"Architecture:\t\tBits: {platform.architecture()[0]}\t"
        f"Linkage: {platform.architecture()[1]}\n\n"
    )

    debug_str += f"Python:\t\t\t{platform.python_implementation()}\n"

    debug_str += f"pygame version:\t\t{ver}\n"

    debug_str += f"python version:\t\t{str_from_tuple(sys.version_info[0:3])}\n\n"

    debug_str += (
        f"SDL versions:\t\tLinked: {str_from_tuple(get_sdl_version())}\t"
        f"Compiled: {str_from_tuple(get_sdl_version(linked = False))}\n"
    )

    debug_str += (
        f"SDL Mixer versions:\tLinked: {str_from_tuple(get_sdl_mixer_version())}\t"
        f"Compiled: {str_from_tuple(get_sdl_mixer_version(linked = False))}\n"
    )

    debug_str += (
        f"SDL Font versions:\tLinked: {str_from_tuple(get_sdl_ttf_version())}\t"
        f"Compiled: {str_from_tuple(get_sdl_ttf_version(linked = False))}\n"
    )

    debug_str += (
        f"SDL Image versions:\tLinked: {str_from_tuple(get_sdl_image_version())}\t"
        f"Compiled: {str_from_tuple(get_sdl_image_version(linked = False))}\n"
    )

    debug_str += (
        f"Freetype versions:\tLinked: {str_from_tuple(ft_version())}\t"
        f"Compiled: {str_from_tuple(ft_version(linked = False))}"
    )

    if filename is None:
        print(debug_str)

    else:
        with open(filename, "w", encoding="utf8") as debugfile:
            debugfile.write(debug_str)
