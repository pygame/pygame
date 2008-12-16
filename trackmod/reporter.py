# module trackmod.reporter

# Keep this first.
def listmods():
    return [n for n, m in sys.modules.iteritems() if m is not None]

import sys
previous_imports = listmods()  #  Keep this after sys but before other imports.
import threading

import module


# This module is does not need explicit thread protection since all calls
# to the data entry methods are made while the import lock is acquired.
collect_data = True
my_imports = None
accesses = None
failed_imports = None

try:
    next
except NameError:
    def next(iterator):
        return iterator.next()

class Largest(object):
    """This object is always greater than any other non Largest object"""
    def __lt__(self, other):
        return False
    def __le__(self, other):
        return self == other
    def __eq__(self, other):
        return isinstance(other, Largest)
    def __ne__(self, other):
        not self == other
    def __gt__(self, other):
        return True
    def __ge__(self, other):
        return True

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
    tracked_types = (module.Module, module.TrackerModule)
    return sorted(n for n, m in list(sys.modules.iteritems())
                    if isinstance(m, tracked_types))

def get_unaccessed_modules():
    """Return a new sorted name list of unaccessed imported modules"""
    unaccessed = []
    iaccessed = iter(get_accessed_modules())
    accessed_name = ''
    for imports_name in get_imports():
        while accessed_name < imports_name:
            try:
                accessed_name = next(iaccessed)
            except StopIteration:
                accessed_name = Largest()
        if imports_name < accessed_name:
            unaccessed.append(imports_name)
    return unaccessed

def get_accessed_modules():
    """Return a new sorted name list of accessed modules"""
    accessed = []
    previous_name = ''
    for name, ignored in module.get_accesses():
        if name != previous_name:
            accessed.append(name)
            previous_name = name
    return accessed

def get_accesses():
    """Return a new dictionary of sorted lists of attributes by module name"""
    accesses = {}
    previous_name = ''
    for name, attribute in module.get_accesses():
        if name != previous_name:
            attributes = []
            accesses[name] = attributes
            previous_name = name
        attributes.append(attribute)
    return accesses





