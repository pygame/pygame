# module trackmod.importer

"""A sys.meta_path importer for tracking module usage."""

import sys
from trackmod import module, namereg

try:
    from imp import reload
except ImportError:
    pass

try:
    collect_data
except NameError:
    pass
else:
    # reload: reload imported modules.
    reload(module)  # implicit reload of reporter
    reload(namereg)

no_modules = []  # Contains nothing.
modules_of_interest = no_modules
add_submodule_accesses = True


class Loader(object):
    def __init__(self, fullname, module):
        self.fullname = fullname
        self.module = module

    def load_module(self, fullname):
        assert fullname == self.fullname, (
            "loader called with wrong module %s: expecting %s" %
              (fullname, self.fullname))
        sys.modules[fullname] = self.module
        return self.module

def find_module(fullname, path=None):
    if fullname in modules_of_interest and fullname not in sys.modules:
        # reload doesn't "get" any tracked TrackerModule attributes.
        m = module.TrackerModule(fullname)

        # Add m to modules so reload works and to prevent infinite recursion.
        sys.modules[fullname] = m
        try:
            try:
                reload(m)
            except ImportError:
                return None
        finally:
            del sys.modules[fullname]

        # Add parent package access.
        if add_submodule_accesses:
            parts = fullname.rsplit('.', 1)
            if len(parts) == 2:
                try:
                    pkg = sys.modules[parts[0]]
                except KeyError:
                    pass
                else:
                    try:
                        getattr(pkg, parts[1])
                    except AttributeError:
                        pass

        return Loader(fullname, m)
    else:
        return None

def end():
    global modules_of_interest
    modules_of_interest = no_modules

def begin(pattern=None, submodule_accesses=True):
    global modules_of_interest, collect_data, add_submodule_accesses
    if pattern is None:
        pattern = ['*']
    modules_of_interest = namereg.NameRegistry(pattern)
    add_submodule_accesses = submodule_accesses
