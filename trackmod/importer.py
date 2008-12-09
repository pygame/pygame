# module trackmod.importer

"""A sys.meta_path importer for tracking module usage."""

import sys

from trackmod import loader
from trackmod import module


collect_data = True

def find_module(fullname, path=None):
    if collect_data and fullname not in sys.modules:
        m = module.Module(fullname)
        sys.modules[fullname] = m
        try:
            try:
                reload(m)
            except ImportError:
                return None;
        finally:
            del sys.modules[fullname]
        m.__class__ = module.TrackerModule
        loader.add(fullname, m)
        return loader
    else:
        return None

def end():
    global collect_data
    collect_data = False
