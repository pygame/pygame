import os
from config import msys

def exec_pkgconfig (package, flags, repl=None):
    if msys.is_msys ():
        pipe = os.popen ("sh pkg-config %s %s" % (flags, package), "r")
    else:
        pipe = os.popen ("pkg-config %s %s" % (flags, package), "r")
    data = pipe.readline ().strip ()
    pipe.close ()
    if data and repl:
        return data.replace (repl, "").split ()
    return data.split ()

def get_incdirs (package):
    return exec_pkgconfig (package, "--cflags-only-I", "-I")

def get_cflags (package):
    return exec_pkgconfig (package, "--cflags-only-other")

def get_libdirs (package):
    return exec_pkgconfig (package, "--libs-only-L", "-L")

def get_libs (package):
    return exec_pkgconfig (package, "--libs-only-l", "-l")

def get_lflags (package):
    return exec_pkgconfig (package, "--libs-only-other")

def get_version (package):
    return exec_pkgconfig (package, "--modversion")

def exists (package):
    pipe = os.popen ("pkg-config --exists %s" % package, "r")
    ret = pipe.close ()
    return ret in (None, 0)

def has_pkgconfig ():
    if msys.is_msys():
        return os.system ("sh pkg-config --version") == 0
    else:
        return os.system ("pkg-config --version > /dev/null 2>&1") == 0
