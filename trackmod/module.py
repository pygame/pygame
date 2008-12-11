# module trackmod.module

"""Implements a module usage tracker module type"""

import threading


ModuleType = type(threading)
getattribute = ModuleType.__getattribute__
accesses = set()
accesses_lock = threading.RLock()


class Module(ModuleType):
    # A heap subtype of the module type.
    #
    # Allows __class__ to be changed. Otherwise it is just the same.
    # To preserve the module docs this description is a comment.
    
    pass


class TrackerModule(ModuleType):
    # A heap subtype of the module type that tracks attribute gets.
    #
    # Allows __class__ to be changed. Otherwise it is just the same.
    # To preserve the module docs this description is a comment.

    # Attributes to ignore in reporting. The module name is the one
    # attribute guarenteed to not be recorded. The class is used by
    # the reporter. The path is just noise.
    ignored_attributes = set(['__name__', '__class__', '__path__'])
    
    def __getattribute__(self, attr):
        if attr in TrackerModule.ignored_attributes:
            return getattribute(self, attr)
        report(self, attr)
        return getattribute(self, attr)


def report_continuous(module, attr):
    accesses_lock.acquire()
    try:
        # Safe: no recursive call on __name__ attribute.
        accesses.add((module.__name__, attr))
    finally:
        accesses_lock.release()

def report_quit(module, attr):
    module.__class__ = Module

def report_oneshot(module, attr):
    report_continuous(module, attr)
    report_quit(module, attr)

report = report_oneshot


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


def get_accesses():
    accesses_lock.acquire()
    try:
        return sorted(accesses)
    finally:
        accesses_lock.release()

