Import("*")

from scons_symbian import *
import glob

python_includes = [ PYTHON_INCLUDE ]

#music.c
#mixer.c
#font.c
#imageext.c
IGNORED = r"""
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
pygame_sources += glob.glob( "../src/SDL_gfx/*.c" )
removed = []
for x in pygame_sources:
    for y in IGNORED:        
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
                             join( "..", "src", "SDL_gfx"),                         
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

pylibzip = "data/pygame/libs/pygame.zip"
def to_package(**kwargs):
    kwargs["source"] = abspath( kwargs["source"] )
    return ToPackage( package = PACKAGE_NAME, pylibzip = pylibzip, 
               dopycompile = ".pyc", **kwargs )

pygame_lib = join( PATH_PY_LIBS, "pygame" )

# Copy main pygame libs
IGNORED_FILES = ["camera.py"]
for x in glob( "../lib/*.py"):
    for i in IGNORED_FILES:
        if x.endswith( i ): break 
    else:
        to_package( source = x, target = pygame_lib )

for x in glob( "../lib/threads/*.py"):
    to_package( source = x, target = join( pygame_lib, "threads") )
    
# Copy Symbian specific libs
for x in glob( "lib/*.py"): 
    to_package( source = x, target = pygame_lib )
    
# Install default font
#to_package( source = "../lib/freesansbold.ttf", target = pygame_lib )



def packagePyS60Stdlib(**kwargs):
    kwargs["source"] = join( "deps/PythonForS60/module-repo/standard-modules", kwargs["source"] )
    return to_package( **kwargs )

packagePyS60Stdlib( source = "glob.py",    target = "data/pygame/libs" )
zippath = packagePyS60Stdlib( source = "fnmatch.py", target = "data/pygame/libs" )

File2Zip( zippath, "../lib/freesansbold.ttf", "pygame/freesansbold.ttf" )
