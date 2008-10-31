import os, sys
from config import sdlconfig, pkgconfig, helpers
from config import config_msys, config_unix, config_win

def get_install_libs (buildsystem, cfg):
    # Gets the libraries to install for the target platform.
    if buildsystem == "unix":
        return config_unix.get_install_libs (cfg)
    elif buildsystem == "msys":
        return config_msys.get_install_libs (cfg)
    elif buildsystem == "win":
        return config_win.get_install_libs (cfg)

def find_incdir (buildsystem, header):
    # Gets the include directory for the specified header file.
    if buildsystem == "msys":
        return config_msys.find_incdir (header)
    elif buildsystem == "unix":
        return config_unix.find_incdir (header)
    elif buildsystem == "win":
        return config_win.find_incdir (header)

def find_libdir (buildsystem, lib):
    # Gets the library directory for the specified library file.
    if buildsystem == "msys":
        return config_msys.find_libdir (lib)
    elif buildsystem == "unix":
        return config_unix.find_libdir (lib)
    elif buildsystem == "win":
        return config_win.find_libdir (lib)

def get_sys_libs (buildsystem, module):
    # Gets a list of system libraries to link the module against.
    if buildsystem == "msys":
        return config_msys.get_sys_libs (module) or []
    elif buildsystem == "unix":
        return config_unix.get_sys_libs (module) or []
    elif buildsystem == "win":
        return config_win.get_sys_libs (module) or []
    
def sdl_get_version (buildsystem):
    # Gets the SDL version.
    if buildsystem == "msys":
        return config_msys.sdl_get_version ()
    elif buildsystem == "unix":
        return config_unix.sdl_get_version ()
    elif buildsystem == "win":
        return config_win.sdl_get_version ()

def prepare_modules (buildsystem, modules, cfg):
    # Prepares all passed modules, setting up their compile and linkage flags,
    # the necessary includes and the inter-module dependencies.
    haspkgconfig = hassdlconfig = False
    if buildsystem in ("unix", "msys"):
        haspkgconfig = pkgconfig.has_pkgconfig ()
        hassdlconfig = sdlconfig.has_sdlconfig ()

    sdlincpath = os.path.join ("src", "sdl")
    sdlincdirs = []
    sdllibdirs = []
    sdllibs = None
    sdlcflags = None
    sdllflags = None
    hassdl = sdl_get_version (buildsystem) != None

    pngincdirs = []
    pnglibdirs = []
    pnglibs = None
    pngcflags = None
    pnglflags = None
    haspng = False

    jpgincdirs = []
    jpglibdirs = []
    jpglibs = None
    jpgcflags = None
    jpglflags = None
    hasjpg = False

    if haspkgconfig:
        sdlincdirs = pkgconfig.get_incdirs ("sdl") + [ sdlincpath ]
        sdllibdirs = pkgconfig.get_libdirs ("sdl")
        sdllibs = pkgconfig.get_libs ("sdl")
        sdlcflags = pkgconfig.get_cflags ("sdl")
        sdllflags = pkgconfig.get_lflags ("sdl")
        if cfg.WITH_PNG and pkgconfig.exists ("libpng"):
            pngincdirs = pkgconfig.get_incdirs ("libpng")
            pnglibdirs = pkgconfig.get_libdirs ("libpng")
            pngcflags = pkgconfig.get_cflags ("libpng") + [ "-DHAVE_PNG" ]
            pnglflags = pkgconfig.get_lflags ("libpng")
            pnglibs = pkgconfig.get_libs ("libpng")
            haspng = True
    elif hassdlconfig:
        sdlincdirs = sdlconfig.get_incdirs () + [ sdlincpath ]
        sdllibdirs = sdlconfig.get_libdirs ()
        sdllibs = sdlconfig.get_libs ()
        sdlcflags = sdlconfig.get_cflags ()
        sdllflags = sdlconfig.get_lflags ()
    else:
        # TODO: Try to find all necessary things manually.
        pass

    if cfg.WITH_PNG and not haspng:
        d = find_incdir (buildsystem, "png.h")
        if d != None:
            pngincdirs = [ d ]
            pnglibdirs = [ find_libdir (buildsystem, "libpng") ]
            pnglibs = [ "png" ]
            pngcflags = [ "-DHAVE_PNG" ]
            pnglflags = []
            haspng = True

    if cfg.WITH_JPEG:
        d = find_incdir (buildsystem, "jpeglib.h")
        if d != None:
            jpgincdirs = [ d ]
            jpglibdirs = [ find_libdir (buildsystem, "libjpeg") ]
            jpglibs = [ "jpeg" ]
            jpgcflags = [ "-DHAVE_JPEG" ]
            jpglflags = []
            hasjpg = True

    if buildsystem == "msys":
        # Msys has to treat the paths differently.
        sdlincdirs = [ config_msys.msys_to_windows (d) for d in sdlincdirs ]
        sdllibdirs = [ config_msys.msys_to_windows (d) for d in sdllibdirs ]
        jpgincdirs = [ config_msys.msys_to_windows (d) for d in jpgincdirs ]
        jpglibdirs = [ config_msys.msys_to_windows (d) for d in jpglibdirs ]
        pngincdirs = [ config_msys.msys_to_windows (d) for d in pngincdirs ]
        pnglibdirs = [ config_msys.msys_to_windows (d) for d in pnglibdirs ]
    
    for mod in modules:
        # Do not build the numericXXX modules on 3.x
        if "numeric" in mod.name and helpers.getversion() >= (3, 0, 0):
            mod.canbuild = False
            continue
        
        # Get module-specific system libraries.
        mod.libs = []
        mod.libs += get_sys_libs (buildsystem, mod.name)
        
        # SDL based module
        if mod.name.startswith ("sdl"):
            mod.canbuild = cfg.WITH_SDL
            mod.incdirs = list (sdlincdirs)
            mod.libdirs = list (sdllibdirs)
            mod.libs += list (sdllibs)
            mod.cflags = list (sdlcflags)
            mod.lflags = list (sdllflags)

            # SDL Mixer linkage
            if mod.name.startswith ("sdlmixer"):
                mod.canbuild = cfg.WITH_SDL and cfg.WITH_SDL_MIXER
                mod.libs += [ "SDL_mixer" ]

            # SDL TTF linkage
            if mod.name.startswith ("sdlttf"):
                mod.canbuild = cfg.WITH_SDL and cfg.WITH_SDL_TTF
                mod.libs += [ "SDL_ttf" ]

            # SDL Image linkage
            if mod.name.startswith ("sdlimage"):
                mod.canbuild = cfg.WITH_SDL and cfg.WITH_SDL_IMAGE
                mod.libs += [ "SDL_image" ]

            # SDL GFX linkage
            if mod.name.startswith ("sdlgfx"):
                mod.canbuild = cfg.WITH_SDL and cfg.WITH_SDL_GFX
                mod.libs += [ "SDL_gfx" ]

            # PNG and JPEG support for Surface.save
            if mod.name == "sdl.video":
                if haspng:
                    mod.cflags += list (pngcflags)
                    mod.lflags += list (pnglflags)
                    mod.incdirs += list (pngincdirs)
                    mod.libdirs += list (pnglibdirs)
                    mod.libs += list (pnglibs)
                if hasjpg:
                    mod.cflags += list (jpgcflags)
                    mod.lflags += list (jpglflags)
                    mod.incdirs += list (jpgincdirs)
                    mod.libdirs += list (jpglibdirs)
                    mod.libs += list (jpglibs)

        # SDL linkage for pygame.mask
        elif mod.name == "mask" and hassdl:
            mod.incdirs = list (sdlincdirs)
            mod.libdirs = list (sdllibdirs)
            mod.libs += list (sdllibs)
            mod.cflags = list (sdlcflags)
            mod.lflags = list (sdllflags)
