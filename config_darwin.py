"""Config on Darwin w/ frameworks"""

import os, sys, string
from glob import glob
from distutils.sysconfig import get_python_inc

class Dependency:
    libext = '.dylib'
    def __init__(self, name, checkhead, checklib, lib):
        self.name = name
        self.inc_dir = None
        self.lib_dir = None
        self.lib = lib
        self.found = 0
        self.checklib = checklib+self.libext
        self.checkhead = checkhead
        self.cflags = ''

    def configure(self, incdirs, libdirs):
        incname = self.checkhead
        libnames = self.checklib, string.lower(self.name)
        for dir in incdirs:
            path = os.path.join(dir, incname)
            if os.path.isfile(path):
                self.inc_dir = dir
                break
        for dir in libdirs:
            for name in libnames:
                path = os.path.join(dir, name)
                if os.path.isfile(path):
                    self.lib_dir = dir
                    break
        if self.lib_dir and self.inc_dir:
            print self.name + '        '[len(self.name):] + ': found'
            self.found = 1
        else:
            print self.name + '        '[len(self.name):] + ': not found'

class FrameworkDependency(Dependency):
    def configure(self, incdirs, libdirs):
      for n in '/Library/Frameworks/','$HOME/Library/Frameworks/','/System/Library/Frameworks/':
        n = os.path.expandvars(n)
        if os.path.isfile(n+self.lib+'.framework/Versions/Current/'+self.lib):
          print 'Framework '+self.lib+' found'
          self.found = 1
          self.inc_dir = n+self.lib+'.framework/Versions/Current/Headers'
          self.cflags = '-Xlinker "-framework" -Xlinker "'+self.lib+'"'
          self.cflags += ' -Xlinker "-F'+n+'"'
          self.origlib = self.lib
          self.lib = ''
          return
      print 'Framework '+self.lib+' not found'


class DependencyPython:
    def __init__(self, name, module, header):
        self.name = name
        self.lib_dir = ''
        self.inc_dir = ''
        self.lib = ''
        self.cflags = ''
        self.found = 0
        self.ver = '0'
        self.module = module
        self.header = header

    def configure(self, incdirs, libdirs):
        self.found = 1
        if self.module:
            try:
                self.ver = __import__(self.module).__version__
            except ImportError:
                self.found = 0
        if self.found and self.header:
            fullpath = os.path.join(get_python_inc(0), self.header)
            if not os.path.isfile(fullpath):
                found = 0
            else:
                self.inc_dir = os.path.split(fullpath)[0]
        if self.found:
            print self.name + '        '[len(self.name):] + ': found', self.ver
        else:
            print self.name + '        '[len(self.name):] + ': not found'

DEPS = [
    FrameworkDependency('SDL', 'SDL.h', 'libSDL', 'SDL'),
    FrameworkDependency('FONT', 'SDL_ttf.h', 'libSDL_ttf', 'SDL_ttf'),
    FrameworkDependency('IMAGE', 'SDL_image.h', 'libSDL_image', 'SDL_image'),
    FrameworkDependency('MIXER', 'SDL_mixer.h', 'libSDL_mixer', 'SDL_mixer'),
    FrameworkDependency('SMPEG', 'smpeg.h', 'libsmpeg', 'smpeg'),
    DependencyPython('NUMERIC', 'Numeric', 'Numeric/arrayobject.h')
]


from distutils.util import split_quoted
def main():
    global DEPS

    print 'Hunting dependencies...'
    incdirs = []
    libdirs = []
    newconfig = []
    for d in DEPS:
      d.configure(incdirs, libdirs)
    DEPS[0].cflags = '-Ddarwin '+ DEPS[0].cflags
    return DEPS


if __name__ == '__main__':
    print """This is the configuration subscript for OSX Darwin.
             Please run "config.py" for full configuration."""
