"""Debug functionality that allows for more useful issue reporting"""

import importlib
import platform
import sys
import traceback
from os import environ

from pygame.system import get_cpu_instruction_sets
from pygame.version import ver


def str_from_tuple(version_tuple):
    """Converts a tuple like (2, 0, 20) into a string joined by periods

    Args:
        version_tuple: tuple(version_major, version_minor, version_patch)

    Returns:
       str: "major.minor.patch"
    """
    if version_tuple is None:
        return "None"

    strs = map(str, version_tuple)
    return ".".join(strs)


def attempt_import(module, function_name, output_str=""):
    """Attempts to import function_name from module

    Args:
        module: string representing module name
        function_name: string representing function name to be imported
        output_str: optional string to prepend error messages to if one occurs

    Returns:
        tuple(str, bool, Any):
            [0]: output_str + error_message
            [1]: True if successful, False if failed
            [2]: if successful, the thing that was imported, else None
    """
    try:
        mod = importlib.import_module(module)
        i = getattr(mod, function_name)
        success = True
    except (ImportError, AttributeError):
        i = None
        output_str += f"There was a problem with {module} import\n"
        output_str += "A default value will be returned for the version\n"
        output_str += traceback.format_exc() + "\n" + "=" * 20 + "\n"
        success = False

    return (output_str, success, i)


def _get_platform_info():
    """
    Internal helper to get platform information
    """
    cpu_inst_dict = get_cpu_instruction_sets()
    sse2 = "Yes" if cpu_inst_dict["SSE2"] else "No"
    avx2 = "Yes" if cpu_inst_dict["AVX2"] else "No"
    neon = "Yes" if cpu_inst_dict["NEON"] else "No"
    ret = f"Platform:\t\t{platform.platform()}\n"
    ret += f"System:\t\t\t{platform.system()}\n"
    ret += f"System Version:\t\t{platform.version()}\n"
    ret += f"Processor:\t\t{platform.processor()}\tSSE2: {sse2}\tAVX2: {avx2}\tNEON: {neon}\n"
    ret += (
        f"Architecture:\t\tBits: {platform.architecture()[0]}\t"
        f"Linkage: {platform.architecture()[1]}\n\n"
    )

    ret += f"Python:\t\t\t{platform.python_implementation()} {sys.version}\n"
    ret += (
        f"GIL Enabled:\t\t{sys._is_gil_enabled()}\n"
        if hasattr(sys, "_is_gil_enabled")
        else "GIL Enabled:\t\tTrue\n"
    )
    ret += f"pygame version:\t\t{ver}\n"
    return ret


def print_debug_info(filename=None):
    """Gets debug information for reporting bugs. Prints to console
    if filename is not specified, otherwise writes to that file
    (note: if filename is not an empty file, it will overwrite whatever is
    in there)

    Args:
        filename: string name of the file to save
    """
    debug_str = ""

    # keyword for compat with getters
    def default_return(linked=True):
        # pylint: disable=unused-argument
        return (-1, -1, -1)

    from pygame.base import get_sdl_version
    from pygame.display import (
        get_driver as get_display_driver,
        get_init as display_init,
    )

    debug_str, *mixer = attempt_import("pygame.mixer", "get_driver", debug_str)
    get_mixer_driver = mixer[1] if mixer[0] else lambda: None

    debug_str, *mixer = attempt_import("pygame.mixer", "get_init", debug_str)
    mixer_init = mixer[1] if mixer[0] else lambda: False

    debug_str, *mixer = attempt_import(
        "pygame.mixer", "get_sdl_mixer_version", debug_str
    )
    get_sdl_mixer_version = mixer[1] if mixer[0] else default_return

    debug_str, *font = attempt_import("pygame.font", "get_sdl_ttf_version", debug_str)
    if not font[0]:
        get_sdl_ttf_version = default_return
    else:
        get_sdl_ttf_version = font[1]

    debug_str, *image = attempt_import(
        "pygame.image", "get_sdl_image_version", debug_str
    )
    if not image[0]:
        get_sdl_image_version = default_return
    else:
        get_sdl_image_version = image[1]

    debug_str, *freetype = attempt_import("pygame.freetype", "get_version", debug_str)
    if not freetype[0]:
        ft_version = default_return
    else:
        ft_version = freetype[1]

    debug_str += _get_platform_info()

    debug_str += (
        f"SDL versions:\t\tLinked: {str_from_tuple(get_sdl_version())}\t"
        f"Compiled: {str_from_tuple(get_sdl_version(linked=False))}\n"
    )

    debug_str += (
        f"SDL Mixer versions:\tLinked: {str_from_tuple(get_sdl_mixer_version())}\t"
        f"Compiled: {str_from_tuple(get_sdl_mixer_version(linked=False))}\n"
    )

    debug_str += (
        f"SDL Font versions:\tLinked: {str_from_tuple(get_sdl_ttf_version())}\t"
        f"Compiled: {str_from_tuple(get_sdl_ttf_version(linked=False))}\n"
    )

    debug_str += (
        f"SDL Image versions:\tLinked: {str_from_tuple(get_sdl_image_version())}\t"
        f"Compiled: {str_from_tuple(get_sdl_image_version(linked=False))}\n"
    )

    debug_str += (
        f"Freetype versions:\tLinked: {str_from_tuple(ft_version())}\t"
        f"Compiled: {str_from_tuple(ft_version(linked=False))}\n\n"
    )

    if display_init():
        driver = get_display_driver()
        if driver.upper() != "X11":
            debug_str += f"Display Driver:\t\t{driver}\n"
        else:
            is_xwayland = (environ.get("XDG_SESSION_TYPE") == "wayland") or (
                "WAYLAND_DISPLAY" in environ
            )
            debug_str += f"Display Driver:\t\t{driver} ( xwayland == {is_xwayland} )\n"
    else:
        debug_str += "Display Driver:\t\tDisplay Not Initialized\n"

    if mixer_init():
        debug_str += f"Mixer Driver:\t\t{get_mixer_driver()}"
    else:
        debug_str += "Mixer Driver:\t\tMixer Not Initialized"

    if filename is None:
        print(debug_str)

    else:
        with open(filename, "w", encoding="utf8") as debugfile:
            debugfile.write(debug_str)
