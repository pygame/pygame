# module trackmod.module

"""Implements a module usage tracker module type"""

from trackmod import keylock
from trackmod import reporter

try:
    ModuleType
except NameError:
    pass
else:
    # reload; reload imported modules
    reload(keylock)
    reload(reporter)

ModuleType = type(reporter)


class Module(ModuleType):
    # A heap subtype of the module type.
    #
    # Allows __class__ to be changed. Otherwise it is just the same.
    # To preserve the module docs this description is a comment.
    
    pass


class TrackerModule(ModuleType):
    def __getattribute__(self, attr):
        if attr in ['__name__', '__path__']:
            # The name attribute is the one attribute guaranteed to not trigger
            # an import. __path__ is just noise in the reporting.
            return ModuleType.__getattribute__(self, attr)
        report(self, attr)
        # At this point self's type has changed so getattr will not
        # recursively call this method.
        return getattr(self, attr)


def report(module, attr):
    name = module.__name__  # Safe: no recursive call on __name__.
    lock = keylock.Lock(name)
    try:
        reporter.add_access(name, attr)
        ModuleType.__setattr__(module, '__class__', Module)
    finally:
        lock.free()
