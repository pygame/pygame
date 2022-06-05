#!/usr/bin/env python

"""Quick tool to help setup the needed paths and flags
in your Setup file. This will call the appropriate sub-config
scripts automatically.

each platform config file only needs a "main" routine
that returns a list of instances. the instances must
contain the following variables.
name: name of the dependency, as references in Setup (SDL, FONT, etc)
inc_dir: path to include
lib_dir: library directory
lib: name of library to be linked to
found: true if the dep is available
cflags: extra compile flags
"""

try:
    import msysio
except ImportError:
    import buildconfig.msysio as msysio
import sys, os, shutil, logging
import sysconfig
import re

BASE_PATH = '.'


def print_(*args, **kwds):
    """Similar to the Python 3.0 print function"""
    # This not only supports MSYS but is also a head start on the move to
    # Python 3.0. Also, this function can be overridden for testing.
    msysio.print_(*args, **kwds)


def is_msys2():
    """Return true if this in an MSYS2 build"""
    return ('MSYSTEM' in os.environ and
            re.match(r'MSYS|MINGW.*|CLANG.*|UCRT.*', os.environ['MSYSTEM']))


def is_msys_mingw():
    """Return true if this in an MinGW/MSYS build

    The user may prompted for confirmation so only call this function
    once.
    """
    return False
    # if msysio.is_msys():
    #     return 1
    # if ('MINGW_ROOT_DIRECTORY' in os.environ or
    #     os.path.isfile(mingwcfg.path)):
    #     return confirm("Is this an mingw/msys build")
    # return 0

def prepdep(dep, basepath):
    """add some vars to a dep"""
    if dep.libs:
        dep.line = dep.name + ' ='
        for lib in dep.libs:
            dep.line += ' -l' + lib
    else:
        dep.line = dep.name + ' = -I.'

    dep.varname = '$('+dep.name+')'

    if not dep.found:
        if dep.name == 'SDL': #fudge if this is unfound SDL
            dep.line = 'SDL = -I/NEED_INC_PATH_FIX -L/NEED_LIB_PATH_FIX -lSDL'
            dep.varname = '$('+dep.name+')'
            dep.found = 1
        return

    incs = []
    lids = []
    IPREFIX = ' -I$(BASE)' if basepath else ' -I'
    LPREFIX = ' -L$(BASE)' if basepath else ' -L'
    startind = len(basepath) if basepath else 0
    if dep.inc_dir:
        if isinstance(dep.inc_dir, str):
            incs.append(IPREFIX+dep.inc_dir[startind:])
        else:
            for dir in dep.inc_dir:
                incs.append(IPREFIX+dir[startind:])
    if dep.lib_dir:
        if isinstance(dep.lib_dir, str):
            lids.append(LPREFIX+dep.lib_dir[startind:])
        else:
            for dir in dep.lib_dir:
                lids.append(LPREFIX+dir[startind:])
    libs = ''
    for lib in dep.libs:
        libs += ' -l' + lib

    if dep.name.startswith('COPYLIB_'):
        dep.line = dep.name + libs + ''.join(lids)
    else:
        dep.line = dep.name+' =' + ''.join(incs) + ''.join(lids) + ' ' + dep.cflags + libs

def writesetupfile(deps, basepath, additional_lines):
    """create a modified copy of Setup.SDLx.in"""
    sdl_setup_filename = os.path.join(BASE_PATH, 'buildconfig',
                                          'Setup.SDL2.in')

    with open(sdl_setup_filename) as origsetup, \
            open(os.path.join(BASE_PATH, 'Setup'), 'w') as newsetup:
        line = ''
        while line.find('#--StartConfig') == -1:
            newsetup.write(line)
            line = origsetup.readline()
        while line.find('#--EndConfig') == -1:
            line = origsetup.readline()

        if basepath:
            newsetup.write('BASE = ' + basepath + '\n')
        for d in deps:
            newsetup.write(d.line + '\n')

        lines = origsetup.readlines()

        # overwrite lines which already exist with new ones.
        new_lines = []
        for l in lines:
            addit = 1
            parts = l.split()
            for al in additional_lines:
                aparts = al.split()
                if aparts and parts:
                    if aparts[0] == parts[0]:
                        #print('the same!' + repr(aparts) + repr(parts))
                        #the same, we should not add the old one.
                        #It will be overwritten by the new one.
                        addit = 0
            if addit:
                new_lines.append(l)

        new_lines.extend(additional_lines)
        lines = new_lines
        legalVars = {d.varname for d in deps}
        legalVars.add('$(DEBUG)')

        for line in lines:
            useit = 1
            if not line.startswith('COPYLIB') and not (line and line[0]=='#'):
                lineDeps = set(re.findall(r'\$\([a-z0-9\w]+\)', line, re.I))
                if lineDeps.difference(legalVars):
                    newsetup.write('#'+line)
                    useit = 0
                if useit:
                    for d in deps:
                        if d.varname in lineDeps and not d.found:
                            useit = 0
                            newsetup.write('#'+line)
                            break
                if useit:
                    legalVars.add(f"$({line.split('=')[0].strip()})")
            if useit:
                newsetup.write(line)

def main(auto=False):
    additional_platform_setup = []
    conan = "-conan" in sys.argv

    if '-sdl2' in sys.argv:
        sys.argv.remove('-sdl2')
    if '-sdl1' in sys.argv:
        raise SystemExit("""Building PyGame with SDL1.2 is no longer supported.
Only SDL2 is supported now.""")

    kwds = {}
    if conan:
        print_('Using CONAN configuration...\n')
        try:
            import config_conan as CFG
        except ImportError:
            import buildconfig.config_conan as CFG

    elif sys.platform == 'win32':
        if sys.version_info >= (3, 8) and is_msys2():
            print_('Using WINDOWS MSYS2 configuration...\n')
            try:
                import config_msys2 as CFG
            except ImportError:
                import buildconfig.config_msys2 as CFG
        else:
            print_('Using WINDOWS configuration...\n')
            try:
                import config_win as CFG
            except ImportError:
                import buildconfig.config_win as CFG

    elif sys.platform == 'darwin':
        print_('Using Darwin configuration...\n')
        try:
            import config_darwin as CFG
        except ImportError:
            import buildconfig.config_darwin as CFG
    elif sysconfig.get_config_var('MACHDEP') == 'emscripten':
        print_('Using Emscripten configuration...\n')
        try:
            import config_emsdk as CFG
        except ImportError:
            import buildconfig.config_emsdk as CFG
    else:
        print_('Using UNIX configuration...\n')
        try:
            import config_unix as CFG
        except ImportError:
            import buildconfig.config_unix as CFG


    if sys.platform == 'win32':
        additional_platform_setup = open(
            os.path.join(BASE_PATH, 'buildconfig', "Setup_Win_Camera.in")).readlines()
    elif sys.platform == 'darwin':
        additional_platform_setup = open(
            os.path.join(BASE_PATH, 'buildconfig', "Setup_Darwin.in")).readlines()
    elif sysconfig.get_config_var('MACHDEP') == 'emscripten':
        additional_platform_setup = open(
            os.path.join(BASE_PATH, 'buildconfig', "Setup.Emscripten.SDL2.in")).readlines()
    else:
        additional_platform_setup = open(
            os.path.join(BASE_PATH, 'buildconfig', "Setup_Unix.in")).readlines()


    if os.path.isfile('Setup'):
        if auto:
            logging.info('Backing up existing "Setup" file into Setup.bak')
            shutil.copyfile(os.path.join(BASE_PATH, 'Setup'), os.path.join(BASE_PATH, 'Setup.bak'))

    deps = CFG.main(**kwds, auto_config=auto)
    if '-conan' in sys.argv:
        sys.argv.remove('-conan')

    if deps:
        basepath = None
        for d in deps:
            prepdep(d, basepath)
        writesetupfile(deps, basepath, additional_platform_setup, **kwds)
        print_("""\nIf you get compiler errors during install, double-check
the compiler flags in the "Setup" file.\n""")
    else:
        print_("""\nThere was an error creating the Setup file, check for errors
or make a copy of "Setup.in" and edit by hand.""")

if __name__ == '__main__':
    main()
