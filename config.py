#!/usr/bin/env python

"""Quick tool to help setup the needed paths and flags
in your Setup file. This will call the appropriate sub-config
scripts automatically.
"""

import sys, os, shutil

if sys.platform == 'win32':
    print 'Using WINDOWS configuration...\n'
    import config_win as CFG
else:
    print 'Using UNIX configuration...\n'
    import config_unix as CFG



def confirm(message):
    reply = raw_input('\n' + message + ' [y/N]:')
    if reply and reply[0].lower() == 'y':
        return 1
    return 0


def main():
    if os.path.isfile('Setup'):
        if confirm('Backup existing "Setup" file'):
            shutil.copyfile('Setup', 'Setup.bak')
    if os.path.isdir('build'):
        if confirm('Remove old build directory (force recompile)'):
            shutil.rmtree('build', 0)

    CFG.main()

    print """Doublecheck that the new "Setup" file looks correct, then
run "python setup.py install" to build and install pygame."""
    

if __name__ == '__main__': main()



