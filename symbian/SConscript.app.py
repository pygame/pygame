from scons_symbian import *

# Import all from main SConstruct
Import("*")

python_includes = [ PYTHON_INCLUDE ]

pygame_rss = File("app/pygame.rss")
pygame_reg_rss = File( "app/pygame_reg.rss")

SymbianProgram( "pygame", TARGETTYPE_EXE,
                resources = [pygame_rss, pygame_reg_rss],
                sources = ["app/pygame_app.cpp", 
                           "app/pygame_main.cpp"                           
                           ],
                includes = ["app", "common",
                            join( "deps", "SDL", "include"),                              
                            join( "deps", "SDL", "symbian", "inc"),
                            C_INCLUDE ] + python_includes
                            
                ,
                defines = [
                    #"__LOGMAN_ENABLED__", 
                ],
                libraries = C_LIBRARY + ["euser", "avkon", "apparc", 
                             "cone","eikcore", "libGLES_CM", 
                             "bafl", # arrays and command line                            
                             PYTHON_LIB_NAME,
                             SDL_DLL_NAME,
                             "pygame.lib",
                             'pygame_libjpeg',
                             #"LogMan"
                             ],
                uid3=UID_PYGAMEAPP,
                icons = [ ("../lib/pygame_icon.svg", "pygame") ],
                package=PACKAGE_NAME,
                capabilities = CAPABILITIES,
                epocheapsize = ( 0x5000, 0x200000 ),
                epocstacksize = 0x14000,
                winscw_options = "-w noempty",
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
to_package( source = "app/launcher/logo.jpg", target = "data/pygame/launcher" )
to_package( source = "app/launcher/pygame_launcher.py", target = "data/pygame/launcher" )
to_package( source = "app/apps/liquid_s60.py", target = "data/pygame/apps" )


