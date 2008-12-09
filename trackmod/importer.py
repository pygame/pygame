# module trackmod.importer

"""A sys.meta_path importer for tracking module usage."""

import sys

from trackmod import loader
from trackmod import module


def find_module(fullname, path=None):
    if fullname in sys.modules:
        return None
    try:
        m = module.Module(fullname)
        sys.modules[fullname] = m
        try:
            reload(m)
        finally:
            del sys.modules[fullname]
    except ImportError:
        return None
    m.__class__ = module.TrackerModule
    loader.add(fullname, m)
    return loader


