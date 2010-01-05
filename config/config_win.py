import os, glob, sys
from config import helpers, dll
from config import config_generic

def sdl_get_version ():
    # Gets the SDL version.
    # TODO: Is there some way to detect the correct version?
    return "Unknown"

def update_sys_deps (deps):
    deps["user32"] = Dependency ([], 'user32')
    deps["user32"].nocheck = True
    deps["gdi32"] = Dependency ([], 'gdi32')
    deps["gdi32"].nocheck = True

def add_sys_deps (module):
    if module.name.startswith("sdl"):
        module.libs += ["SDLmain"]
    if module.name == "sdlext.scrap":
        module.depends.append ("user32")
        module.depends.append ("gdi32")

def _hunt_libs (name, dirs):
    # Used by get_install_libs(). It resolves the dependency libraries
    # and returns them as dict.
    libs = {}
    x = dll.dependencies (name)
    for key in x.keys ():
        values = _get_libraries (key, dirs)
        libs.update (values)
    return libs

def _get_libraries (name, directories):
    # Gets the full qualified library path from directories.
    libs = {}
    dotest = dll.tester (name)
    for d in directories:
        try:
            files = os.listdir (d)
        except:
            pass
        else:
            for f in files:
                filename = os.path.join (d, f)
                if dotest (f) and os.path.isfile (filename):
                    # Found
                    libs[filename] = 1
    return libs

def get_install_libs (cfg):
    # Gets the libraries to install for the target platform.
    _libdirs = [ "", "VisualC\\SDL\\Release", "VisualC\\Release", "Release",
                 "lib"]
    _searchdirs = [ "prebuilt", "..", "..\\..", sys.prefix ]

    libraries = {}
    values = {}
    dirs = []
    
    for d in _searchdirs:
        for g in _libdirs:
            dirs.append (os.path.join (d, g))
    
    if cfg.build['SDL']:
        libraries.update (_hunt_libs ("SDL", dirs))
    if cfg.build['SDL_MIXER']:
        libraries.update (_hunt_libs ("SDL_mixer", dirs))
    if cfg.build['SDL_IMAGE']:
        libraries.update (_hunt_libs ("SDL_image", dirs))
    if cfg.build['SDL_TTF']:
        libraries.update (_hunt_libs ("SDL_ttf", dirs))
    if cfg.build['SDL_GFX']:
        libraries.update (_hunt_libs ("SDL_gfx", dirs))
    if cfg.build['PNG']:
        libraries.update (_hunt_libs ("png", dirs))
    if cfg.build['JPEG']:
        libraries.update (_hunt_libs ("jpeg", dirs))
    if cfg.build['FREETYPE']:
        libraries.update (_hunt_libs ("freetype", dirs))
    if cfg.build['PORTMIDI']:
        libraries.update (_hunt_libs ("portmidi", dirs))

    return libraries.keys ()

class Dependency (config_generic.Dependency):
    _searchdirs = [ "prebuilt", "..", "..\\..", sys.prefix ]
    _incdirs = [ "", "include" ]
    _libdirs = [ "", "VisualC\\SDL\\Release", "VisualC\\Release", "Release",
                 "lib"]
    _libprefix = ""

    def configure (self, cfg):
        super(Dependency, self).configure (cfg)
        
        # HACK: freetype.h is in freetype2\\freetype, but we need
        # freetype2\\.
        if self.library_id != "freetype":
            return
        incs = []
        for d in self.incdirs:
            if d.endswith ("freetype2\\freetype"):
                incs.append (d.replace ("freetype2\\freetype", "freetype2"))
            else:
                incs.append (d)
        self.incdirs = incs
