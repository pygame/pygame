#!/usr/bin/env python
# For MinGW build requires Python 2.4 or better and win32api.

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

import msysio
import mingwcfg
import sys, os, shutil

def print_(*args, **kwds):
    """Simular to the Python 3.0 print function"""
    # This not only supports MSYS but is also a head start on the move to
    # Python 3.0. Also, this function can be overridden for testing.
    msysio.print_(*args, **kwds)

def confirm(message, default=None):
    "ask a yes/no question, return result"
    if not sys.stdout.isatty():
        if default is None:
            raise RuntimeError("Non interactive, tried to ask: %s" % message)
        return default
    reply = msysio.raw_input_("\n%s [Y/n]:" % message)
    if reply and reply[0].lower() == 'n':
        return False
    return True

def is_msys_mingw():
    """Return true if this in an MinGW/MSYS build

    The user may prompted for confirmation so only call this function
    once.
    """
    if msysio.is_msys():
        return 1
    if ('MINGW_ROOT_DIRECTORY' in os.environ or
        os.path.isfile(mingwcfg.path)):
        return confirm("Is this an mingw/msys build")
    return 0

def prepdep(dep, basepath):
    "add some vars to a dep"
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

    inc = lid = lib = ""
    if basepath:
        if dep.inc_dir: inc = ' -I$(BASE)'+dep.inc_dir[len(basepath):]
        if dep.lib_dir: lid = ' -L$(BASE)'+dep.lib_dir[len(basepath):]
    else:
        if dep.inc_dir: inc = ' -I' + dep.inc_dir
        if dep.lib_dir: lid = ' -L' + dep.lib_dir
    libs = ''
    for lib in dep.libs: 
        libs += ' -l' + lib

    if dep.name.startswith('COPYLIB_'):
        dep.line = dep.name + libs + lid
    else:
        dep.line = dep.name+' =' + inc + lid + ' ' + dep.cflags + libs

def writesetupfile(deps, basepath, additional_lines, SDL2=False):
    "create a modified copy of Setup.SDLx.in"
    if SDL2:
        origsetup = open('Setup.SDL2.in', 'r')
    else:
        origsetup = open('Setup.SDL1.in', 'r')
    newsetup = open('Setup', 'w')
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
                    #print ('the same!' + repr(aparts) + repr(parts))
                    #the same, we should not add the old one.  
                    #It will be overwritten by the new one.
                    addit = 0
        if addit:
            new_lines.append(l)

    new_lines.extend(additional_lines)
    lines = new_lines

    for line in lines:
        useit = 1
        if not line.startswith('COPYLIB'):
            for d in deps:
                if line.find(d.varname)!=-1 and not d.found:
                    useit = 0
                    newsetup.write('#'+line)
                    break
        if useit:
            newsetup.write(line)        

def main():
    additional_platform_setup = []
    SDL2 = "-SDL2" in sys.argv
    kwds = {}
    if SDL2:
        kwds['SDL2'] = True
    if (sys.platform == 'win32' and
        # Note that msys builds supported for 2.6 and greater. Use prebuilt.
        (sys.version_info >= (2, 6) or not is_msys_mingw())):
        print_('Using WINDOWS configuration...\n')
        import config_win as CFG
    elif sys.platform == 'win32':
        print_('Using WINDOWS mingw/msys configuration...\n')
        import config_msys as CFG
    elif sys.platform == 'darwin':
        print_('Using Darwin configuration...\n')
        import config_darwin as CFG
        additional_platform_setup = open("Setup_Darwin.in", "r").readlines()
    else:
        print_('Using UNIX configuration...\n')
        import config_unix as CFG
        additional_platform_setup = open("Setup_Unix.in", "r").readlines()
    
    if os.path.isfile('Setup'):
        if "-auto" in sys.argv or confirm('Backup existing "Setup" file', False):
            shutil.copyfile('Setup', 'Setup.bak')
    if not "-auto" in sys.argv and os.path.isdir('build'):
        if confirm('Remove old build directory (force recompile)', False):
            shutil.rmtree('build', 0)

    deps = CFG.main(**kwds)
    if SDL2:
        e = deps[0]
        assert(e.name == 'SDL')
        e.cflags += ' -DSDL2'
    if deps:
        basepath = None
        for d in deps:
            prepdep(d, basepath)
        writesetupfile(deps, basepath, additional_platform_setup, **kwds)
        print_("""\nIf you get compiler errors during install, doublecheck
the compiler flags in the "Setup" file.\n""")
    else:
        print_("""\nThere was an error creating the Setup file, check for errors
or make a copy of "Setup.in" and edit by hand.""")

if __name__ == '__main__': main()
