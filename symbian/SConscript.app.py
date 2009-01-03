from scons_symbian import *

# Import all from main SConstruct
Import("*")

python_includes = [ join( PYTHON_CORE, x ) for x in r"Symbian Objects Parser Python include Modules".split() ]

pygame_rss = File("app/pygame.rss")
pygame_reg_rss = File( "app/pygame_reg.rss")

SymbianProgram( "pygame", TARGETTYPE_EXE,
                resources = [pygame_rss, pygame_reg_rss],
                sources = ["app/pygame_app.cpp", 
                           "app/pygame_main.cpp"                           
                           ],
                includes = ["app", "common",
                join( EPOC32_INCLUDE, "SDL"), 
                join( EPOC32_INCLUDE, "libc"),] + python_includes
                ,
                defines = [
                    #"__LOGMAN_ENABLED__", 
                ],
                libraries = ["euser", "estlib", "avkon", "apparc", 
                             "cone","eikcore", "libGLES_CM",                             
                             PYTHON_LIB_NAME,
                             SDL_DLL_NAME,
                             "pygame.lib",
                             #"LogMan"
                             ],
                uid3=UID_PYGAMEAPP,
                icons = [ ("../lib/pygame_icon.svg", "pygame") ],
                package=PACKAGE_NAME,
                capabilities = CAPABILITIES,
                epocheapsize = ( 0x5000, 0x200000 ),
                epocstacksize = 0x14000,
)

# Install pygame app resources
from glob import glob

def to_package(**kwargs):
    kwargs["source"] = abspath( kwargs["source"] )
    #kwargs["target"] = abspath( kwargs["target"] )
    ToPackage( package = PACKAGE_NAME, **kwargs )

#for x in glob("*.bmp"):
#    to_package( source = x, target = "data/pygame")

to_package( source = "app/pygame_main.py", target = "data/pygame" )
#to_package( source = "app/ambient.ogg", target = "data/pygame" )

