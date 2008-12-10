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
getattribute = ModuleType.__getattribute__


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
            return getattribute(self, attr)
        report(self, attr)
        return getattribute(self, attr)


def report_oneshot(module, attr):
    name = module.__name__  # Safe: no recursive call on __name__.
    lock = keylock.Lock(name)
    try:
        reporter.add_access(name, attr)
        ModuleType.__setattr__(module, '__class__', Module)
    finally:
        lock.free()

def report_continuous(module, attr):
    name = module.__name__  # Safe: no recursive call on __name__.
    lock = keylock.Lock(name)
    try:
        reporter.add_access(name, attr)
    finally:
        lock.free()

def report_quit(module, attr):
    name = module.__name__  # Safe: no recursive call on __name__.
    lock = keylock.Lock(name)
    try:
        ModuleType.__setattr__(module, '__class__', Module)
    finally:
        lock.free()
    
def set_report_mode(mode=None):
    """Set whether access checking is oneshot or continuous

    if mode (default 'oneshot') is 'oneshot' or None then a TrackerModule
    module will stop recording attribute accesses after the first non-trivial
    access. If 'continuous' then all attribute accesses are recorded. If
    'quit' then access recording stops and further calls to this function
    have no effect.

    """
    global report
    
    if report is report_quit:
        return
    if mode is None:
        mode = 'oneshot'
    if mode == 'oneshot':
        report = report_oneshot
    elif mode == 'continuous':
        report = report_continuous
    elif mode == 'quit':
        report = report_quit
    else:
        raise ValueError("Unknown mode %s" % mode)

report = report_oneshot
