import platform
from os import getcwd, chdir, path
import sys

__all__ = ["Video_AutoInit"]


def Video_AutoInit():
    """Called from the base.c just before display module is initialized."""
    if "Darwin" in platform.platform():
        if (getcwd() == "/") and len(sys.argv) > 1:
            chdir(path.dirname(sys.argv[0]))
    return True
