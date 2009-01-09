
from scons_symbian import *
import glob

Import("*")

python_includes = [ PYTHON_INCLUDE ]

#music.c
#mixer.c
#font.c
#imageext.c
ignored = r"""
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

# Build pygame library
SymbianProgram( "pygame", TARGETTYPE_LIB,
                sources = pygame_sources,
                defines = [                    
                ],
                includes = python_includes + [
                             "common",                             
                             join( "deps", "jpeg"),
                             join( "deps", "SDL_image"),
                             join( EPOC32_INCLUDE, "SDL"),
                             join( EPOC32_INCLUDE, "libc"),                             
                             #join( "..", "..", "tools", "debug" )
                           ],
                package = PACKAGE_NAME,
                libraries = [
                     PYTHON_LIB_NAME,
                     "euser", "estlib", "avkon", "apparc", 
                     "cone","eikcore", "libGLES_CM",                     
                     SDL_DLL_NAME,
                     ],
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
