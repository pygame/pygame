import os
from config import msys

def exec_config(libconfig, flags):
    if msys.is_msys ():
        pipe = os.popen ("sh %s %s " % (libconfig, flags), "r")
    else:
        pipe = os.popen ("%s %s " % (libconfig, flags), "r")
    data = pipe.readline ().strip ()
    pipe.close ()
    return data.split ()

def get_incdirs(libconfig):
    flags = exec_config (libconfig, "--cflags")
    newflags = []
    for f in flags:
        if f.startswith ('-I'):
            newflags.append (f[2:])
    return newflags

def get_cflags(libconfig):
    flags = exec_config (libconfig, "--cflags")
    newflags = []
    for f in flags:
        if f.startswith ('-I'):
            continue
        newflags.append (f)
    return newflags

def get_libdirs(libconfig):
    flags = exec_config (libconfig, "--libs")
    newflags = []
    for f in flags:
        if f.startswith ('-L'):
            newflags.append (f[2:])
    return newflags

def get_libs(libconfig):
    flags = exec_config (libconfig, "--libs")
    newflags = []
    for f in flags:
        if f.startswith ('-l'):
            newflags.append (f[2:])
    return newflags

def get_lflags(libconfig):
    flags = exec_config (libconfig, "--libs")
    newflags = []
    for f in flags:
        if f.startswith ('-l') or f.startswith ('-L'):
            continue
        newflags.append (f)
    return newflags

def get_version(libconfig):
    return exec_config(libconfig, "--version")

def has_libconfig(libconfig):
    if msys.is_msys ():
        return os.system ("sh %s --version" % libconfig) == 0
    else:
        return os.system ("%s --version > /dev/null 2>&1" % libconfig) == 0
