def listmods():
    return [n for n, m in sys.modules.iteritems() if m is not None]

import sys
already_loaded = listmods()
import threading


def print_(*args, **kwds):
    stream = kwds.get('file', sys.stdout)
    sep = kwds.get('sep', ' ')
    end = kwds.get('end', '\n')

    if args:
        stream.write(sep.join([str(arg) for arg in args]))
    if end:
        stream.write(end)

def process_accessed():
    acc_names = dict(accessed)
    for name, attr in accessed:
        parts = name.split('.')
        for i in range(1, len(parts)):
            subname = '.'.join(parts[0:i])
            if subname not in acc_names:
                acc_names[subname] = parts[i]
    return set(acc_names.iteritems())

def write_report(repfile):
    def rep(*args, **kwds):
        print_(file=repfile, *args, **kwds)

    accessed = process_accessed()
    rep("=== module usage report ===")
    rep("\n-- modules already imported (ignored) --")
    already_loaded.sort()
    for name in already_loaded:
        rep(name)
    rep("\n-- modules added by trackmod (ignored) --")
    added_by_trackmod.sort()
    for name in added_by_trackmod:
        rep(name)
    rep("\n-- modules imported but not accessed --")
    acc = set([n for n, ignored in accessed])
    unaccessed = list(loaded - acc)
    unaccessed.sort()
    for name in unaccessed:
        rep(name)
    rep("\n-- modules accessed --")
    acc = list(accessed)
    acc.sort()
    for name, attr in acc:
        rep(name, "(%s)" % attr)
    rep("\n=== end of report ===")

def init():
    global already_loaded, loaded, accessed, data_lock, added_by_trackmod
    added_by_trackmod = list(set(listmods()) - set(already_loaded))
    loaded = set()
    accessed = set()
    data_lock = threading.Lock()

def add_loaded(name):
    data_lock.acquire()
    try:
        loaded.add(name)
    finally:
        data_lock.release()

def add_accessed(name, attr):
    data_lock.acquire()
    try:
        accessed.add((name, attr))
    finally:
        data_lock.release()


