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

import sys, os, shutil, string



if sys.platform == 'win32':
    print 'Using WINDOWS configuration...\n'
    import config_win
    CFG = config_win
elif sys.platform == 'darwin':
    print 'Using Darwin configuration...\n'
    import config_darwin
    CFG = config_darwin
else:
    print 'Using UNIX configuration...\n'
    import config_unix
    CFG = config_unix




def confirm(message):
    "ask a yes/no question, return result"
    reply = raw_input('\n' + message + ' [y/N]:')
    if reply and string.lower(reply[0]) == 'y':
        return 1
    return 0



def prepdep(dep, basepath):
    "add some vars to a dep"
    if dep.lib:
        dep.line = dep.name + ' = -l' + dep.lib
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
    if dep.lib: lib = ' -l'+dep.lib
    dep.line = dep.name+' =' + inc + lid + ' ' + dep.cflags + lib



def writesetupfile(deps, basepath):
    "create a modified copy of Setup.in"
    origsetup = open('Setup.in', 'r')
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

    while line:
        line = origsetup.readline()
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
    if os.path.isfile('Setup'):
        if confirm('Backup existing "Setup" file'):
            shutil.copyfile('Setup', 'Setup.bak')
    if os.path.isdir('build'):
        if confirm('Remove old build directory (force recompile)'):
            shutil.rmtree('build', 0)

    deps = CFG.main()
    if deps:
        basepath = None
        for d in deps:
            prepdep(d, basepath)
        writesetupfile(deps, basepath)
        print """\nIf you get compiler errors during install, doublecheck
the compiler flags in the "Setup" file.\n"""
    else:
        print """\nThere was an error creating the Setup file, check for errors
or make a copy of "Setup.in" and edit by hand."""



if __name__ == '__main__': main()
