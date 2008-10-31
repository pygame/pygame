import os
from config import msys

def exec_sdlconfig (flags):
    if msys.is_msys ():
        pipe = os.popen ("sh sdl-config %s " % flags, "r")
    else:
        pipe = os.popen ("sdl-config %s " % flags, "r")
    data = pipe.readline ().strip ()
    pipe.close ()
    return data.split ()

def get_incdirs ():
    flags = exec_sdlconfig ("--cflags")
    newflags = []
    for f in flags:
        if f.startswith ('-I'):
            newflags.append (f[2:])
    return newflags

def get_cflags ():
    flags = exec_sdlconfig ("--cflags")
    newflags = []
    for f in flags:
        if f.startswith ('-I'):
            continue
        newflags.append (f)
    return newflags

def get_libdirs ():
    flags = exec_sdlconfig ("--libs")
    newflags = []
    for f in flags:
        if f.startswith ('-L'):
            newflags.append (f[2:])
    return newflags

def get_libs ():
    flags = exec_sdlconfig ("--libs")
    newflags = []
    for f in flags:
        if f.startswith ('-l'):
            newflags.append (f[2:])
    return newflags

def get_lflags ():
    flags = exec_sdlconfig ("--libs")
    newflags = []
    for f in flags:
        if f.startswith ('-l') or f.startswith ('-L'):
            continue
        newflags.append (f)
    return newflags

def get_version ():
    return exec_sdlconfig ("--version")

def has_sdlconfig ():
    if msys.is_msys ():
        return os.system ("sh sdl-config --version") == 0
    else:
        return os.system ("sdl-config --version > /dev/null 2>&1") == 0
