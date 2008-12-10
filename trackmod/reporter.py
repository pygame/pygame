# module trackmod.reporter

# Keep this first.
def listmods():
    return [n for n, m in sys.modules.iteritems() if m is not None]

import sys
previous_imports = listmods()  #  Keep this after sys but before other imports.
import threading

# This module is does not need explicit thread protection since all calls
# to the data entry methods are made while the import lock is acquired.
collect_data = True
my_imports = None
accesses = None
failed_imports = None


def process_accessed():
    acc_names = dict(accessed)
    for name, attr in accessed:
        parts = name.split('.')
        for i in range(1, len(parts)):
            subname = '.'.join(parts[0:i])
            if subname not in acc_names:
                acc_names[subname] = parts[i]
    return set(acc_names.iteritems())

def begin():
    global previous_imports, my_imports, accesses, failed_imports
    my_imports = list(set(listmods()) - set(previous_imports))
    accesses = {}
    failed_imports = set()

def end():
    global collect_data
    collect_data = False

def add_import(name):
    """Add a module to the import list

    Expects to be called in the order in which modules are created:
    package, submodule, etc.

    """
    if collect_data:
        accesses[name] = set()
 
def remove_import(name):
    del accesses[name]
    failed_imports.add(name)

def add_access(name, attr):
    if collect_data:
        accesses[name].add(attr)

def get_previous_imports():
    """Return a new sorted name list of previously imported modules"""
    return sorted(previous_imports)

def get_my_imports():
    """Return a new sorted name list of module imported by this package"""
    return sorted(my_imports)

def get_imports():
    """Return a new sorted name list of imported modules"""
    return sorted(accesses.iterkeys())

def get_unaccessed_modules():
    """Return a new sorted name list of unaccessed imported modules"""
    return sorted(n for n, a in accesses.iteritems() if not a)
    
def get_accessed_modules():
    """Return a new sorted name list of accessed modules"""
    return sorted(n for n, a in accesses.iteritems() if a)

def get_accesses():
    """Return a new dictionary of sorted lists of attributes by module name"""
    return dict((n, sorted(a)) for n, a in accesses.iteritems() if a)




