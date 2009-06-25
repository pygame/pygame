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

def exec_pkgconfig (package, flags, repl=None):
    cmd = ["pkg-config", flags, package]
    data = run_command (cmd)[1]
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
    cmd = ["pkg-config", "--exists", package]
    return run_command (cmd)[0] in (None, 0)

def has_pkgconfig ():
    cmd = ["pkg-config", "--version"]
    return run_command (cmd)[0] == 0
