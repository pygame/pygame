"""Config on Unix"""
#would be nice if it auto-discovered which other modules where available

import os, sys, shutil
from glob import glob

configcommand = 'sdl-config --version --cflags --libs'

    

def writesetupfile(flags):
    origsetup = open('Setup.in', 'r')
    newsetup = open('Setup', 'w')
    while 1:
        line = origsetup.readline()
        if not line: break
        if line.startswith('SDL = '):
            line = 'SDL = ' + flags
        newsetup.write(line)
    

def main():
    configinfo = "-I/usr/include/SDL -D_REENTRANT -lSDL"
    print 'calling "sdl-config"'
    try:
        configinfo = os.popen(configcommand).readlines()
        print 'Found SDL version:', configinfo[0]
        configinfo = ' '.join(configinfo[1:])
        configinfo = configinfo.split()
        configinfo = ' '.join(configinfo)
        print 'Flags:', configinfo
    except:
        raise SystemExit, """Cannot locate command, "sdl-config". Default SDL compile
flags have been used, which will likely require a little editing."""

    print '\n--Creating new "Setup" file...'
    writesetupfile(configinfo)
    print '--Finished'


    
if __name__ == '__main__':
    print """This is the configuration subscript for Unix.
Please run "config.py" for full configuration."""

