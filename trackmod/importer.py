# module trackmod.importer

"""A sys.meta_path importer for tracking module usage."""

import sys

from trackmod import module, reporter

try:
    collect_data
except NameError:
    pass
else:
    # reload: reload imported modules.
    reload(module)  # implicit reload of reporter

collect_data = True

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
    if collect_data and fullname not in sys.modules:
        # Put this first so the order of inserts follows the order of calls to
        # find_module: package, subpackage, etc.
        reporter.add_import(fullname)

        # reload doesn't get any tracked TrackerModule attributes.
        m = module.TrackerModule(fullname)

        # Add m to modules so reload works and to prevent infinite recursion.
        sys.modules[fullname] = m
        try:
            try:
                reload(m)
            except ImportError, e:
                reporter.remove_import(fullname)
                return None;
        finally:
            del sys.modules[fullname]

        # Add parent package access.
        parts = fullname.rsplit('.', 1)
        if len(parts) == 2:
            try:
                pkg = sys.modules[parts[0]]
                try:
                    getattr(pkg, parts[1])
                except AttributeError:
                    pass
            except KeyError:
                pass

        return Loader(fullname, m)
    else:
        return None

def end():
    global collect_data
    collect_data = False
