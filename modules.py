from distutils.extension import Extension
import sys, os
import config, cfg

# For create_cref.py
sys.path.append ("doc")
import create_cref

class Module:
    def __init__ (self, name, sources=None, instheaders=[], docfile=None):
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
        self.cflags_avail = "-DHAVE_PYGAME_" + nn

modules = [
    Module ("base", [ "src/base/basemod.c",
                      "src/base/bufferproxy.c",
                      "src/base/color.c",
                      "src/base/floatrect.c",
                      "src/base/rect.c",
                      "src/base/surface.c" ],
            [ "src/pgcompat.h",
              "src/base/pgbase.h",
              "src/base/pgdefines.h",
              "src/base/pgtypes.h" ], "base.xml"),

    Module ("mask", [ "src/mask/bitmask.c",
                      "src/mask/mask.c",
                      "src/mask/maskmod.c" ],
            [ "src/mask/bitmask.h",
              "src/mask/pgmask.h" ], "mask.xml"),

    Module ("physics", [ "src/physics/aabbox.c",
                         "src/physics/body.c",
                         "src/physics/collision.c",
                         "src/physics/contact.c",
                         "src/physics/joint.c",
                         "src/physics/physicsmod.c",
                         "src/physics/rectshape.c",
                         "src/physics/shape.c",
                         "src/physics/vector.c",
                         "src/physics/world.c" ],
           [ "src/physics/pgphysics.h" ], "physics.xml"),

    Module ("sdl.base", [ "src/sdl/sdlmod.c" ], [ "src/sdl/pgsdl.h" ],
            "sdlbase.xml"),
    Module ("sdl.audio", [ "src/sdl/audiomod.c" ], docfile="sdlaudio.xml"),
    Module ("sdl.cdrom", [ "src/sdl/cdrommod.c",
                           "src/sdl/cdrom.c",
                           "src/sdl/cdtrack.c" ], docfile="sdlcdrom.xml"),
    Module ("sdl.constants", [ "src/sdl/constantsmod.c" ]),
    Module ("sdl.event", [ "src/sdl/eventmod.c",
                           "src/sdl/event.c" ], docfile="sdlevent.xml"),
    Module ("sdl.gl", [ "src/sdl/glmod.c" ], docfile="sdlgl.xml"),
    Module ("sdl.image", [ "src/sdl/imagemod.c" ], docfile="sdlimage.xml"),
    Module ("sdl.joystick", [ "src/sdl/joystickmod.c",
                              "src/sdl/joystick.c" ],
            docfile="sdljoystick.xml"),
    Module ("sdl.keyboard", [ "src/sdl/keyboardmod.c" ],
            docfile="sdlkeyboard.xml"),
    Module ("sdl.mouse", [ "src/sdl/cursor.c",
                           "src/sdl/mousemod.c" ], docfile="sdlmouse.xml"),
    Module ("sdl.rwops", [ "src/sdl/rwopsmod.c" ], docfile="sdlrwops.xml"),
    Module ("sdl.time", [ "src/sdl/timemod.c" ], docfile="sdltime.xml"),
    Module ("sdl.video", [ "src/sdl/pixelformat.c",
                           "src/sdl/surface_blit.c",
                           "src/sdl/surface_fill.c",
                           "src/sdl/surface_save.c",
                           "src/sdl/surface.c",
                           "src/sdl/tga.c",
                           "src/sdl/png.c",
                           "src/sdl/jpg.c",
                           "src/sdl/overlay.c",
                           "src/sdl/videomod.c" ], docfile="sdlvideo.xml"),
    Module ("sdl.wm", [ "src/sdl/wmmod.c" ], docfile="sdlwm.xml"),

    Module ("sdlext.base", [ "src/sdlext/pixelarray.c",
                             "src/sdlext/sdlextmod.c" ],
            [ "src/sdlext/pgsdlext.h" ], "sdlextbase.xml"),
    Module ("sdlext.constants", [ "src/sdlext/constantsmod.c" ]),
    Module ("sdlext.draw", [ "src/sdlext/draw.c",
                             "src/sdlext/drawmod.c" ],
            docfile="sdlextdraw.xml"),
    Module ("sdlext.fastevent", [ "src/sdlext/fasteventmod.c",
                                  "src/sdlext/fastevents.c" ],
            docfile="sdlextfastevent.xml"),
    Module ("sdlext.numericsurfarray", [ "src/sdlext/numericsurfarraymod.c" ],
            docfile="sdlextnumericsurfarray.xml"),
    Module ("sdlext.scrap", [ "src/sdlext/scrapmod.c",
                              "src/sdlext/scrap.c",
                              "src/sdlext/scrap_x11.c",
                              "src/sdlext/scrap_win.c" ],
            docfile="sdlextscrap.xml"),
    Module ("sdlext.transform", [ "src/sdlext/transform.c",
                                  "src/sdlext/filters.c",
                                  "src/sdlext/transformmod.c" ],
            docfile="sdlexttransform.xml"),

    Module ("sdlmixer.base", [ "src/sdlmixer/mixermod.c",
                               "src/sdlmixer/chunk.c",
                               "src/sdlmixer/channel.c",
                               "src/sdlmixer/music.c" ],
            [ "src/sdlmixer/pgmixer.h" ], "sdlmixerbase.xml"),
    Module ("sdlmixer.constants", [ "src/sdlmixer/constantsmod.c" ]),
    Module ("sdlmixer.channel", [ "src/sdlmixer/channelmod.c" ],
            docfile="sdlmixerchannel.xml"),
    Module ("sdlmixer.music", [ "src/sdlmixer/musicmod.c" ],
            docfile="sdlmixermusic.xml"),
    Module ("sdlmixer.numericsndarray",
            [ "src/sdlmixer/numericsndarraymod.c" ],
            docfile="sdlmixernumericsndarray.xml"),

    Module ("sdlttf.base", [ "src/sdlttf/ttfmod.c",
                             "src/sdlttf/font.c" ],
            [ "src/sdlttf/pgttf.h" ], "sdlttfbase.xml"),
    Module ("sdlttf.constants", [ "src/sdlttf/constantsmod.c" ]),

    Module ("sdlimage.base", [ "src/sdlimage/imagemod.c" ],
            docfile="sdlimagebase.xml"),

    Module ("sdlgfx.base", [ "src/sdlgfx/fpsmanager.c",
                             "src/sdlgfx/gfxmod.c" ],
            [ "src/sdlgfx/pggfx.h" ], "sdlgfxbase.xml"),
    Module ("sdlgfx.constants", [ "src/sdlgfx/constantsmod.c" ]),
    Module ("sdlgfx.primitives", [ "src/sdlgfx/primitivesmod.c" ],
            docfile="sdlgfxprimitives.xml"),
    Module ("sdlgfx.rotozoom", [ "src/sdlgfx/rotozoommod.c" ],
            docfile="sdlgfxrotozoom.xml"),
    ]

def get_extensions (buildsystem):
    extensions = []

    compatpath = "src"
    docpath = os.path.join ("src", "doc")
    baseincpath = os.path.join ("src", "base")

    config.config_modules.prepare_modules (buildsystem, modules, cfg)
    
    allmodcflags = []
    for mod in modules:
        # Build the availability cflags
        if mod.canbuild:
            allmodcflags += [ mod.cflags_avail ]

    # Create the extensions
    for mod in modules:
        if not mod.canbuild:
            print ("Skipping module '%s'" % mod.name)
            continue
        ext = Extension ("pygame2." + mod.name, sources=mod.sources)
        ext.extra_compile_args = [ "-DPYGAME_INTERNAL" ] + mod.cflags + \
                                 allmodcflags
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

def update_packages (cfg, packages, package_dir, package_data):
    if cfg.WITH_SDL:
        packages += [ "pygame2.sdl", "pygame2.sdlext" ]
        package_dir["pygame2.sdl"] = "lib/sdl"
        package_dir["pygame2.sdlext"] = "lib/sdlext"
        
        if cfg.WITH_SDL_MIXER:
            packages += [ "pygame2.sdlmixer" ]
            package_dir["pygame2.sdlmixer"] = "lib/sdlmixer"
        if cfg.WITH_SDL_TTF:
            packages += [ "pygame2.sdlttf" ]
            package_dir["pygame2.sdlttf"] = "lib/sdlttf"
        if cfg.WITH_SDL_IMAGE:
            packages += [ "pygame2.sdlimage" ]
            package_dir["pygame2.sdlimage"] = "lib/sdlimage"
        if cfg.WITH_SDL_GFX:
            packages += [ "pygame2.sdlgfx" ]
            package_dir["pygame2.sdlgfx"] = "lib/sdlgfx"
