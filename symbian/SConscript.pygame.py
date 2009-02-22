Import("*")

from scons_symbian import *
import glob

python_includes = [ PYTHON_INCLUDE ]

#music.c
#mixer.c
#font.c
#imageext.c
ignored = r"""
pypm.c
camera.c
ffmovie.c
movie.c
movieext.c
pixelarray_methods.c
scale_mmx32.c
scale_mmx64.c
scrap.c
scrap_mac.c
scrap_qnx.c
scrap_win.c
scrap_x11.c
_numericsndarray.c
joystick.c
cdrom.c""".split()

pygame_sources = glob.glob( "../src/*.c" )
removed = []
for x in pygame_sources:
    for y in ignored:        
        if x.endswith(y):
            removed.append(x)
            break

for x in removed:    
    pygame_sources.remove(x)

pygame_sources.append("common/builtins.c")

if USE_OPENC:
    C_LIB_INCLUDE = "OPENC"
else:
    C_LIB_INCLUDE = ""
    
# Build pygame library
SymbianProgram( "pygame", TARGETTYPE_LIB,
                sources = pygame_sources,
                defines = [ 
                   C_LIB_INCLUDE                   
                ],
                includes = python_includes + [
                             "common",                             
                             join( "deps", "jpeg"),
                             join( "deps", "SDL_image"),
                             join( "deps", "SDL_ttf"),
                             join( "deps", "SDL_mixer"),
                             join( "deps", "SDL", "include"),                              
                             join( "deps", "SDL", "symbian", "inc"),
                             C_INCLUDE,                             
                             #join( "..", "..", "tools", "debug" )
                           ],
                package = PACKAGE_NAME,
                libraries = C_LIBRARY + [
                     PYTHON_LIB_NAME,                     
                     "euser", "avkon", "apparc", 
                     "cone","eikcore", "libGLES_CM",                     
                     SDL_DLL_NAME,
                     ],
                winscw_options = "-w noempty",
                )

# Install pygame python libraries
from glob import glob

def to_package(**kwargs):
    kwargs["source"] = abspath( kwargs["source"] )
    ToPackage( package = PACKAGE_NAME, **kwargs )

pygame_lib = join( PATH_PY_LIBS, "pygame" )

# Copy main pygame libs
for x in glob( "../lib/*.py"):
    to_package( source = x, target = pygame_lib )

# Copy Symbian specific libs
for x in glob( "lib/*.py"): 
    to_package( source = x, target = pygame_lib )
    
# Install default font
to_package( source = "../lib/freesansbold.ttf", target = pygame_lib )
