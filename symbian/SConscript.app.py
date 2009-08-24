"""This is the build recipe for the pygame launcher application"""

from scons_symbian import *

# Import all from main SConstruct
Import("*")

python_includes = [ PYTHON_INCLUDE ]

pygame_rss = File("app/pygame.rss")
pygame_reg_rss = File( "app/pygame_reg.rss")

defines = [
    #"__LOGMAN_ENABLED__", 
]
sources = [
    "app/pygame_app.cpp", 
    "app/pygame_main.cpp",                                          
]

if HAVE_STATIC_MODULES:
    sources += [ "common/builtins.c" ]
    defines += [ "HAVE_STATIC_MODULES=1"]
    
SymbianProgram( "pygame", TARGETTYPE_EXE,
                resources = [pygame_rss, pygame_reg_rss],
                sources   = sources, 
                includes  = ["app", "common",
                            join( "deps", "SDL", "include"),                              
                            join( "deps", "SDL", "symbian", "inc"),
                            C_INCLUDE ] 
                         + python_includes,
                defines   = defines,
                libraries = C_LIBRARY + ["euser", "avkon", "apparc", 
                             "cone","eikcore", "libGLES_CM", 
                             "bafl", # arrays and command line                            
                             PYTHON_LIB_NAME,
                             SDL_DLL_NAME,                             
                             'pygame_libjpeg',
                             #"LogMan"
                             ] + PYGAME_STATIC_MODULES, # Add static pygame modules
                uid3         = UID_PYGAMEAPP,
                icons        = [ ("../lib/pygame_icon.svg", "pygame") ],
                package      = PACKAGE_NAME,
                capabilities = CAPABILITIES,
                # 100k, 4MB
                epocheapsize   = ( 0x19000, 0x400000 ),
                epocstacksize  = 0x14000,
                winscw_options = "-w noempty",
)

# Install pygame app resources
from glob import glob

def doPackage(**kwargs):
    kwargs["source"] = abspath( kwargs["source"] )
    #kwargs["target"] = abspath( kwargs["target"] )
    ToPackage( package = PACKAGE_NAME, **kwargs )


doPackage( source = "app/pygame_main.py", target = "data/pygame", 
           dopycompile=False )
doPackage( source = "app/launcher/pygame_launcher.py", target = "data/pygame/launcher", 
           dopycompile=False )
doPackage( source = "app/apps/liquid_s60.py", target = "data/pygame/apps", 
           dopycompile=False )
doPackage( source = "app/launcher/logo.jpg", target = "data/pygame/launcher" )


