# module trackmod.loader

"""A sys.meta_path loader for module usage tracking."""

import sys
import threading

from trackmod import reporter

module_table = {}
module_table_lock = threading.Lock()

def add(name, module):
    module_table_lock.acquire()
    try:
        module_table[name] = module
    finally:
        module_table_lock.release()

def pop(name):
    module_table_lock.acquire()
    try:
        return module_table.pop(name)
    finally:
        module_table_lock.release()

def load_module(fullname):
    m = pop(fullname)
    sys.modules[fullname] = m
    reporter.add_loaded(fullname)
    return m

