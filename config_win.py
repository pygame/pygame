"""Config on Windows"""

import os, sys, shutil
from glob import glob

huntpaths = ['..', '..'+os.sep+'..', '..'+os.sep+'*', \
             '..'+os.sep+'..'+os.sep+'*']


class Dependency:
    inc_hunt = ['include']
    lib_hunt = [os.path.join('VisualC', 'SDL', 'Release'),
                os.path.join('VisualC', 'Release'),
                'Release', 'lib']
    def __init__(self, name, wildcard, lib, required = 0):
        self.name = name
        self.wildcard = wildcard
        self.required = required
        self.paths = []
        self.path = None
        self.inc_dir = None
        self.lib_dir = None
        self.lib = lib
        self.varname = '$('+name+')'
        self.line = ""
                 
    def hunt(self):
        parent = os.path.abspath('..')
        for p in huntpaths:
            found = glob(os.path.join(p, self.wildcard))
            found.sort() or found.reverse()  #reverse sort
            for f in found:
                if f[:5] == '..'+os.sep+'..' and os.path.abspath(f)[:len(parent)] == parent:
                    continue
                if os.path.isdir(f):
                    self.paths.append(f)

    def choosepath(self):
        if not self.paths:
            print 'Path for ', self.name, 'not found.'
            if self.required: print 'Too bad that is a requirement! Hand-fix the "Setup"'
        elif len(self.paths) == 1:
            self.path = self.paths[0]
            print 'Path for '+self.name+':', self.path
        else:
            print 'Select path for '+self.name+':'
            for i in range(len(self.paths)):
                print '  ', i+1, '=', self.paths[i]
            print '  ', 0, '= <Nothing>'
            choice = raw_input('Select 0-'+`len(self.paths)`+' (1=default):')
            if not choice: choice = 1
            else: choice = int(choice)
            if(choice):
                self.path = self.paths[choice-1]

    def findhunt(self, base, paths):
        for h in paths:
            hh = os.path.join(base, h)
            if os.path.isdir(hh):
                return hh.replace('\\', '/')
        return base.replace('\\', '/')

    def configure(self):
        self.hunt()
        self.choosepath()
        if self.path:
            self.inc_dir = self.findhunt(self.path, Dependency.inc_hunt)
            self.lib_dir = self.findhunt(self.path, Dependency.lib_hunt)

    def buildline(self, basepath):
        inc = lid = lib = " "
        if basepath:
            if self.inc_dir: inc = ' -I$(BASE)'+self.inc_dir[len(basepath):]
            if self.lib_dir: lid = ' -L$(BASE)'+self.lib_dir[len(basepath):]
        else:
            if self.inc_dir: inc = ' -I' + self.inc_dir
            if self.lib_dir: lid = ' -L' + self.lib_dir
        if self.lib: lib = ' -l'+self.lib
        self.line = self.name+' =' + inc + lid + lib
            
        

DEPS = (
    Dependency('SDL', 'SDL-[0-9].*', 'SDL', 1),
    Dependency('FONT', 'SDL_ttf-[0-9].*', 'SDL_ttf'),
    Dependency('IMAGE', 'SDL_image-[0-9].*', 'SDL_image'),
    Dependency('MIXER', 'SDL_mixer-[0-9].*', 'SDL_mixer'),
#copy only dependencies
    Dependency('SMPEG', 'smpeg-[0-9].*', 'smpeg')
)




    

def writesetupfile(DEPS, basepath):
    origsetup = open('Setup.in', 'r')
    newsetup = open('Setup', 'w')
    line = ''
    while line.find('#--StartConfig') == -1:
        newsetup.write(line)
        line = origsetup.readline()
    while line.find('#--EndConfig') == -1:
        line = origsetup.readline()
    if basepath:
        newsetup.write('BASE=' + basepath + '\n')
    for d in DEPS:
        newsetup.write(d.line + '\n')
    while line:
        line = origsetup.readline()
        useit = 1
        for d in DEPS:
            if line.find(d.varname)!=-1 and not d.path:
                useit = 0
                newsetup.write('#'+line)
                break
        if useit:          
            newsetup.write(line)
    

def main():
    global DEPS

    allpaths = []
    for d in DEPS:
        d.configure()
        if d.path:
            allpaths.append(d.inc_dir)
            allpaths.append(d.lib_dir)

    basepath = os.path.commonprefix(allpaths)
    lastslash = basepath.rfind('/')
    if(lastslash < 3 or len(basepath) < 3):
        basepath = ""
    else:
        basepath = basepath[:lastslash]

    for d in DEPS:
        d.buildline(basepath)

    writesetupfile(DEPS, basepath)
    

if __name__ == '__main__':
    print """This is the configuration subscript for Windows.
Please run "config.py" for full configuration."""

