from distutils.extension import Extension
import sys, os
import config, cfg
from config import helpers

# For create_cref.py
sys.path.append ("doc")
import create_cref

class Module:
    def __init__ (self, name, sources=None, instheaders=[], docfile=None,
                  depends=None, optional_dep=None):
        """
            Initializes the Module object.

            name -  Name of this module
            sources - List of all the C sources which make the module
            instheaders - Additional C headers
            docfile - XML file containing the documentation for the module
            depends -   List of all the external libraries on which this 
                        module depends.
                        These libraries must be declared beforehand in 
                        config.config_modules.DEPENDENCIES
            optional_dep - List of optional libraries with which this module
                           can be built.
        """

        self.name = name
        self.sources = sources
        self.installheaders = instheaders
        self.incdirs = []
        self.libdirs = []
        self.libs = []
        self.lflags = []
        self.cflags = []
        self.docfile = docfile
        self.canbuild = True
        nn = name.upper ().replace (".", "_")
        self.globaldefines = [("HAVE_PYGAME_" + nn, None)]

        self.depends = list (depends or [])
        self.optional_dep = list(optional_dep or [])

    def __repr__ (self):
        return "<Module '%s'>" % self.name

modules = [

    Module ("base",
        sources = [
            "src/base/basemod.c",
            "src/base/bufferproxy.c",
            "src/base/color.c",
            "src/base/floatrect.c",
            "src/base/rect.c",
            "src/base/streamwrapper.c",
            "src/base/surface.c",
            "src/base/font.c"],

        instheaders = [
            "src/pgcompat.h",
            "src/base/pgbase.h",
            "src/base/pgdefines.h",
            "src/base/pgtypes.h" ],
        
        docfile = "base.xml"),

    Module ("mask",
        sources = [
            "src/mask/bitmask.c",
            "src/mask/mask.c",
            "src/mask/maskmod.c" ],

        instheaders = [
            "src/mask/bitmask.h",
            "src/mask/pgmask.h" ],
        
        docfile = "mask.xml",
        optional_dep = ['SDL']),

    Module ("sdl.base",
        sources = [ "src/sdl/sdlmod.c" ],
        instheaders = [ "src/sdl/pgsdl.h" ],
        docfile = "sdlbase.xml",
        depends = ['SDL']),

    Module ("sdl.audio",
        sources = [ "src/sdl/audiomod.c" ],
        docfile = "sdlaudio.xml",
        depends = ['SDL']),

    Module ("sdl.cdrom",
        sources = [
            "src/sdl/cdrommod.c",
            "src/sdl/cdrom.c",
            "src/sdl/cdtrack.c" ],

        docfile="sdlcdrom.xml",
        depends = ['SDL']),

    Module ("sdl.constants",
        sources = [ "src/sdl/constantsmod.c" ],
        depends = ['SDL']),

    Module ("sdl.event",
        sources = [
            "src/sdl/eventmod.c",
            "src/sdl/event.c" ],

        docfile = "sdlevent.xml",
        depends = ['SDL']),

    Module ("sdl.gl",
        sources = [ "src/sdl/glmod.c" ],
        docfile = "sdlgl.xml",
        depends = ['SDL']),

    Module ("sdl.image",
        sources = [ "src/sdl/imagemod.c" ],
        docfile = "sdlimage.xml",
        depends = ['SDL']),

    Module ("sdl.joystick",
        sources = [
            "src/sdl/joystickmod.c",
            "src/sdl/joystick.c" ],

        docfile="sdljoystick.xml",
        depends = ['SDL']),

    Module ("sdl.keyboard",
        sources = [ "src/sdl/keyboardmod.c" ],
        docfile = "sdlkeyboard.xml",
        depends = ['SDL']),

    Module ("sdl.mouse",
        sources = [
            "src/sdl/cursor.c",
            "src/sdl/mousemod.c" ],
        docfile = "sdlmouse.xml",
        depends = ['SDL']),

    Module ("sdl.rwops",
        sources = [ "src/sdl/rwopsmod.c" ],
        docfile = "sdlrwops.xml",
        depends = ['SDL']),

    Module ("sdl.time",
        sources = [ "src/sdl/timemod.c" ],
        docfile = "sdltime.xml",
        depends = ['SDL']),

    Module ("sdl.video",
        sources = [
            "src/sdl/pixelformat.c",
            "src/sdl/surface_blit.c",
            "src/sdl/surface_blit_rgb.c",
            "src/sdl/surface_blit_rgba.c",
            "src/sdl/surface_fill.c",
            "src/sdl/surface_save.c",
            "src/sdl/surface.c",
            "src/sdl/tga.c",
            "src/sdl/png.c",
            "src/sdl/jpg.c",
            "src/sdl/overlay.c",
            "src/sdl/videomod.c" ],
        
        docfile = "sdlvideo.xml",
        depends = ['SDL'],
        optional_dep = ['jpeg', 'png']),

    Module ("sdl.wm",
        sources = [ "src/sdl/wmmod.c" ],
        docfile = "sdlwm.xml",
        depends = ['SDL']),

    Module ("sdlext.base",
        sources = [
            "src/sdlext/pixelarray.c",
            "src/sdlext/sdlextmod.c" ],
        instheaders = [ "src/sdlext/pgsdlext.h" ],
        docfile = "sdlextbase.xml",
        depends = ['SDL']),

    Module ("sdlext.constants",
        sources = [ "src/sdlext/constantsmod.c" ],
        depends = ['SDL']),

    Module ("sdlext.draw",
        sources = [
            "src/sdlext/draw.c",
            "src/sdlext/drawmod.c" ],

        docfile = "sdlextdraw.xml",
        depends = ['SDL']),

    Module ("sdlext.fastevent",
        sources = [
            "src/sdlext/fasteventmod.c",
            "src/sdlext/fastevents.c" ],

        docfile = "sdlextfastevent.xml",
        depends = ['SDL']),

    Module ("sdlext.scrap",
        sources = [
            "src/sdlext/scrapmod.c",
            "src/sdlext/scrap.c",
            "src/sdlext/scrap_x11.c",
            "src/sdlext/scrap_win.c" ],

        docfile = "sdlextscrap.xml",
        depends = ['SDL']),

    Module ("sdlext.transform",
        sources = [
            "src/sdlext/transform.c",
            "src/sdlext/filters.c",
            "src/sdlext/transformmod.c" ],

        docfile = "sdlexttransform.xml",
        depends = ['SDL']),

    Module ("sdlmixer.base",
        sources = [
            "src/sdlmixer/mixermod.c",
            "src/sdlmixer/chunk.c",
            "src/sdlmixer/channel.c",
            "src/sdlmixer/music.c" ],

        instheaders = [ "src/sdlmixer/pgmixer.h" ],
        docfile = "sdlmixerbase.xml",
        depends = ['SDL', 'SDL_mixer']),

    Module ("sdlmixer.constants",
        sources = [ "src/sdlmixer/constantsmod.c" ],
        depends = ['SDL', 'SDL_mixer']),

    Module ("sdlmixer.channel",
        sources = [ "src/sdlmixer/channelmod.c" ],
        docfile = "sdlmixerchannel.xml",
        depends = ['SDL', 'SDL_mixer']),

    Module ("sdlmixer.music",
        sources = [ "src/sdlmixer/musicmod.c" ],
        docfile = "sdlmixermusic.xml",
        depends = ['SDL', 'SDL_mixer']),


    Module ("sdlttf.base",
        sources = [
            "src/sdlttf/ttfmod.c",
            "src/sdlttf/font.c" ],

        instheaders = [ "src/sdlttf/pgttf.h" ],
        docfile = "sdlttfbase.xml",
        depends = ['SDL', 'SDL_ttf']),

    Module ("sdlttf.constants",
        sources = [ "src/sdlttf/constantsmod.c" ],
        depends = ['SDL', 'SDL_ttf']),

    Module ("sdlimage.base",
        sources = [ "src/sdlimage/imagemod.c" ],
        docfile = "sdlimagebase.xml",
        depends = ['SDL', 'SDL_image']),

    Module ("sdlimage.constants",
        sources = [ "src/sdlimage/constantsmod.c" ],
        depends = ['SDL', 'SDL_image']),

    Module ("sdlgfx.base",
        sources = [
            "src/sdlgfx/fpsmanager.c",
            "src/sdlgfx/gfxmod.c" ],

        instheaders = [ "src/sdlgfx/pggfx.h" ],
        docfile = "sdlgfxbase.xml",
        depends = ['SDL', 'SDL_gfx']),

    Module ("sdlgfx.constants",
        sources = [ "src/sdlgfx/constantsmod.c" ],
        depends = ['SDL', 'SDL_gfx']),

    Module ("sdlgfx.primitives",
        sources = [ "src/sdlgfx/primitivesmod.c" ],
        docfile = "sdlgfxprimitives.xml",
        depends = ['SDL', 'SDL_gfx']),

    Module ("sdlgfx.rotozoom",
        sources = [ "src/sdlgfx/rotozoommod.c" ],
        docfile = "sdlgfxrotozoom.xml",
        depends = ['SDL', 'SDL_gfx']),

    Module ("freetype.base",
        sources = [
            "src/freetype/ft_mod.c",
            "src/freetype/ft_font.c",
            "src/freetype/ft_wrap.c",
            "src/freetype/ft_render.c",
            "src/freetype/ft_render_cb.c",
            "src/freetype/ft_metrics.c",
            "src/freetype/ft_text.c",
            "src/freetype/ft_cache.c",
        ],
        instheaders = ["src/freetype/pgfreetype.h"],
        docfile = "freetypebase.xml",
        depends = ['freetype'],
        optional_dep = ['SDL']),

    Module ("freetype.constants",
        sources = [ "src/freetype/ft_constants.c" ],
        depends = ['freetype'],
        optional_dep = ['SDL']),

    Module ("midi.pypm",
        sources = [ "src/midi/pypm.c" ],
        depends = ['portmidi']),

    Module ("math.base",
        sources = [ "src/math/mathmod.c",
                    "src/math/vector.c",
                    "src/math/vector2.c",
                    "src/math/vector3.c" ],
        instheaders = ["src/math/pgmath.h"],
        docfile = "mathbase.xml"),

    Module ("openal.constants",
        sources = [ "src/openal/constantsmod.c" ],
        depends = [ 'openal' ]),
    
    Module ("openal.base",
        sources = [ "src/openal/openalmod.c",
                    "src/openal/context.c",
                    "src/openal/device.c" ],
        depends = [ 'openal' ]),
    ]

if helpers.getversion() < (3, 0, 0):
    modules.append(
        Module ("sdlmixer.numericsndarray",
            sources = [ "src/sdlmixer/numericsndarraymod.c" ],
            docfile = "sdlmixernumericsndarray.xml",
            depends = ['SDL', 'SDL_mixer']))

    modules.append(
        Module ("sdlext.numericsurfarray",
            sources = [ "src/sdlext/numericsurfarraymod.c" ],
            docfile = "sdlextnumericsurfarray.xml",
            depends = ['SDL']))

def get_extensions (buildsystem):
    extensions = []

    compatpath = "src"
    docpath = os.path.join ("src", "doc")
    baseincpath = os.path.join ("src", "base")

    config.config_modules.prepare_modules (buildsystem, modules, cfg)
    
    alldefines = []
    for mod in modules:
        # Build the availability defines
        if mod.canbuild:
            for entry in mod.globaldefines:
                if entry not in alldefines:
                    alldefines.append (entry)
    
    # Create the extensions
    for mod in modules:
        if not mod.canbuild:
            print ("Skipping module '%s'" % mod.name)
            continue
        ext = Extension ("pygame2." + mod.name, sources=mod.sources)
        ext.define_macros.append (("PYGAME_INTERNAL", None))
        for entry in alldefines:
            ext.define_macros.append (entry)
        ext.extra_compile_args = mod.cflags
        ext.extra_link_args = mod.lflags
        ext.include_dirs = mod.incdirs + [ baseincpath, compatpath, docpath ]
        ext.library_dirs = mod.libdirs
        ext.libraries = mod.libs
        ext.basemodule = mod
        extensions.append (ext)

    return extensions

def create_docheader (module, docincpath):
    docfile = module.docfile
    if not docfile:
        return

    incfile = "%s.h" % docfile.replace (".xml", "_doc")
    docpath = os.path.join ("doc", "src")
    create_cref.create_c_header (os.path.join (docpath, docfile),
                                 os.path.join (docincpath, incfile))

def update_packages (modules, packages, package_dir, package_data):

    pkg_list = set([m.name.split('.')[1] for m in modules])

    for p in pkg_list:
        pkg_name = "pygame2." + p
        pkg_dir = os.path.join("lib", p)

        if os.path.isdir(pkg_dir):
            packages.append(pkg_name)
            package_dir[pkg_name] = pkg_dir
