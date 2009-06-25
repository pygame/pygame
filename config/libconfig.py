import subprocess
from config import msys, helpers
try:
    msys_obj = msys.Msys (require_mingw=False)
except:
    msys_obj = None

def run_command (cmd):
    try:
        retcode, output = 0, None
        if msys.is_msys():
            retcode, output = msys_obj.run_shell_command (cmd)
        else:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            output = p.communicate()[0]
            retcode = p.returncode
        if helpers.getversion()[0] >= 3:
            output = str (output, "utf-8")
        return retcode, output
    except OSError:
        return -1, None

def exec_config(libconfig, flags):
    cmd = [libconfig, flags]
    return run_command (cmd)[1].split ()

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
    cmd = [libconfig, "--version"]
    return run_command (cmd)[0] == 0
