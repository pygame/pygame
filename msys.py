# module msys.py
# Requires Python 2.4 or better and win32api.

"""MSYS specifics for Msys terminal IO and for running shell scripts

exports msys_raw_input, MsysException, Msys
"""

from msysio import raw_input_ as msys_raw_input, print_ as msys_print
from msysio import is_msys
import os
import time
import subprocess
import re
import glob
try:
    import _winreg
except ImportError:
    import winreg as _winreg

# For Python 2.x/3.x compatibility
def geterror():
    return sys.exc_info()[1]

FSTAB_REGEX = (r'^[ \t]*(?P<path>'
               r'([a-zA-Z]:){0,1}([\\/][^\s*^?:%\\/]+)+)'
               r'[ \t]+/mingw(\s|$)'
               )

def has_drive(path):
    """Return true if the MSYS path strats with a drive letter"""
    
    return re.match('/[A-Z]/', path, re.I) is not None

class MsysException(Exception):
    """Path retrieval problem"""
    pass

def find_msys_version_subdir(msys_dir):
    """Return the full MSYS root directory path

    If msys_dir path lacks the version subdirectory, e.g. 1.0, then the
    path is searched for one. The user will be prompted to choose if more
    than one version is found.
    """

    regex = r'[\\/][1-9][.][0-9]$'
    if re.search(regex, msys_dir) is not None:
        return msys_dir
    
    roots = glob.glob(os.path.join(msys_dir, '[1-9].[0-9]'))
    roots.sort()
    roots.reverse()
    if not roots:
        raise MsysException("No msys versions found.\n")
    else:
        if len(roots) == 1:
            root = roots[0]
        else:
            msys_print("Select an Msys version:")
            for i, path in enumerate(roots):
                msys_print("  %d = %s" % (i+1, os.path.split(path)[1]))
            choice = msys_raw_input("Select 1-%d (1 = default):")
            if not choice:
                root = roots[0]
            else:
                root = roots[int(choice)-1]
        return root
        
def input_msys_dir():
    """Return user entered MSYS directory path

    May raise MsysException."""

    while 1:
        dir_path = msys_raw_input("Enter the MSYS directory path,\n"
                              "(or press [Enter] to quit):")
        dir_path = dir_path.strip()
        if not dir_path:
            raise MsysException("Input aborted by user")
        dir_path = os.path.abspath(dir_path)
        try:
            return find_msys_version_subdir(dir_path)
        except MsysException:
            msys_print(geterror())
            
def find_msys_registry():
    """Return the MSYS 1.0 directory path stored in the Windows registry

    The return value is an encoded ascii str. The registry entry for the
    uninstaller is used. Raise a LookupError if not found.
    """
    
    subkey = (
        'Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\MSYS-1.0_is1')
    try:
        key = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, subkey)
        try:
            return _winreg.QueryValueEx(key, 'Inno Setup: App Path')[0].encode()
        finally:
            key.Close()
    except WindowsError:
        raise LookupError("MSYS not found in the registry")

def as_shell(msys_root):
    """Append MSYS shell program to MSYS root directory path"""

    return os.path.join(msys_root, 'bin', 'sh.exe')

def check_for_shell(msys_directory=None):
    """Check various locations for MSYS shell or root directory.

    May raise MsysException.
    """

    if msys_directory is not None:
        try:
            dir_path = find_msys_version_subdir(msys_directory)
        except MsysException:
            pass
        else:
            return as_shell(dir_path)

    try:
        shell = os.environ['SHELL']
    except KeyError:
        pass
    else:
        if is_msys():
            return shell + '.exe'
        return shell

    try:
        dir_path = find_msys_registry()
    except LookupError:
        pass
    else:
        return as_shell(dir_path)

    return as_shell(input_msys_dir())

def find_msys_shell(msys_directory=None):
    """Retrun the MSYS shell program path

    MsysException is raised if the shell program is not found. The user
    is prompt is prompted as a last resort if no directory is found or
    there are multiple choices.
    """

    shell = check_for_shell(msys_directory)

    while 1:
        shell = os.path.abspath(shell.replace('/', os.sep))
        if os.path.isfile(shell):
            break
        msys_print("Directory %s has no MSYS shell." % shell)
        shell = as_shell(input_msys_dir())
    return shell

def find_mingw_root(msys_directory):
    """Return the Windows equivalent of /mingw"""

    # Look it up in the fstabs file.
    fstab_path = os.path.join(msys_directory, 'etc', 'fstab')
    try:
        fstab = open(fstab_path, 'r')
    except IOError:
        raise MsysException("Unable to open MSYS fstab file %s" % fstab_path)
    else:
        match = re.search(FSTAB_REGEX, fstab.read(), re.MULTILINE)
        if match is None:
            raise MsysException(
                "The required MinGW path is not in the MSYS fstab file")

        dir_path = os.path.abspath(match.groupdict()['path'])
        if not os.path.isdir(dir_path):
            raise MsysException("%s is not a directory" % dir_path)
    return dir_path


class Msys(object):
    """Return a new Msys environment;  May raise MsysException

    Msys([msys_directory, [require_mingw]])

    msys_directory: A string giving the path of the MSYS directory.

    Either or both keyword arguments can be omitted. If msys_directory
    is not provided then the environment variable SHELL and the Windows
    registry are checked. Finally the user is prompted for the directory
    path. If require_mingw is True, the default, the mingw directory path
    is retrieved from the MSYS fstab file. An MsysException is raised if
    the required paths are not found.
    """

    _is_msys = is_msys()

    def __init__(self, msys_directory=None, require_mingw=None):
        """New environment

        May raise MsysException"""

        if require_mingw is None:
            require_mingw = True
        self._environ = os.environ.copy()
        self._shell = find_msys_shell(msys_directory)
        self._msys_root = os.path.split(os.path.split(self.shell)[0])[0].lower()
        try:
            self._mingw_root = find_mingw_root(self.msys_root)
        except MsysException:
            if require_mingw:
                raise
            self._mingw_root = None
        else:
            self.environ['MINGW_ROOT_DIRECTORY'] = self._mingw_root

    environ = property(lambda self: self._environ,
                       doc="Environment variables")
    shell = property(lambda self: self._shell,
                     doc="MSYS shell program path")
    msys_root = property(lambda self: self._msys_root,
                         doc="MSYS root directory path")
    mingw_root = property(lambda self: self._mingw_root,
                          doc="MinGW root directory path")
    is_msys = property(lambda self: self._is_msys,
                       doc="True if the execution environment is MSYS")

    def windows_to_msys(self, path):
        """Return an MSYS translation of an absolute Windows path

        """
        
        msys_root = self.msys_root
        mingw_root = self.mingw_root
        path_lower = path.lower()
        if path_lower.startswith(msys_root.lower()):
            return '/usr' + path[len(msys_root):].replace(os.sep, '/')
        if mingw_root is not None and path_lower.startswith(mingw_root.lower()):
            return '/mingw' + path[len(mingw_root):].replace(os.sep, '/')
        drive, tail = os.path.splitdrive(path)
        tail = tail.replace(os.sep, '/')
        return '/%s%s' % (drive[0], tail)

    def msys_to_windows(self, path):
        """Return a Windows translation of an MSYS path
        
        The Unix path separator is used as it survives the distutils setup
        file read process. Raises a ValueError if the path cannot be
        translated.
        """

        msys_root = self.msys_root
        mingw_root = self.mingw_root
        if path.startswith('/usr'):
            path =  msys_root + path[4:]
        elif path.startswith('/mingw'):
            if mingw_root is None:
                raise ValueError('Unable to map the MinGW directory')
            path =  mingw_root + path[6:]
        elif has_drive(path):
            path =  path[1] + ":" + path[2:]
        elif path == '/':
            path = msys_root
        elif path.startswith('/'):
            path =  msys_root + path
        return path.replace(os.sep, '/')


    def run_shell_script(self, script):
        """Run the MSYS shell script and return the shell return code

        script is a string representing the contents of the script.
        """
        
        cmd = [self.shell]
        if not self._is_msys:
            cmd.append('--login')
        previous_cwd = os.getcwd()
        try:
            process = subprocess.Popen(cmd,
                                       stdin=subprocess.PIPE,
                                       env=self.environ)
            process.communicate(script)
            return process.returncode
        finally:
            time.sleep(2)  # Allow shell subprocesses to terminate.
            os.chdir(previous_cwd)

    def run_shell_command(self, command):
        """Run the MSYS shell command and return stdout output as a string

        command is a list of strings giving the command and its arguments.
        The first list entry  must be the MSYS path name of a bash shell
        script file.
        """
        
        args = [self.shell]
        if not self._is_msys:
            args.append('--login')
        args.extend(command)
        previous_cwd = os.getcwd()
        try:
            return subprocess.Popen(args,
                                    stdout=subprocess.PIPE,
                                    env=self.environ).communicate()[0]
        finally:
            time.sleep(3)  # Allow shell subprocesses to terminate.
            os.chdir(previous_cwd)

__all__ = ['Msys', 'msys_raw_input', 'msys_print', 'MsysException']
