Import("*")

from glob import glob
from scons_symbian import *
import os

python_includes = [ PYTHON_INCLUDE ]
 
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

pygame_sources = glob( "../src/*.c" )
pygame_sources += glob( "../src/SDL_gfx/*.c" )
removed = []
for x in pygame_sources:
    for y in IGNORED:        
        if x.endswith(y):
            removed.append(x)
            break

for x in removed:    
    pygame_sources.remove(x)
    
if USE_OPENC:
    C_LIB_INCLUDE = "OPENC"
else:
    C_LIB_INCLUDE = ""

#: List of static modules for linking
PYGAME_STATIC_MODULES = []

def createPygameLibrary( modname, sources, ):
    modname = "pygame_" + modname
    uid     = 0
    targettype = TARGETTYPE_LIB
    if not HAVE_STATIC_MODULES:
        targettype = TARGETTYPE_PYD
        # Add the PyS60 prefix
        modname    = "kf_" + modname
        uid        = getUID()
    # Build pygame library
    SymbianProgram( modname, targettype,
                sources = ["../src/" + x for x in sources],
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
                           ],
                package = PACKAGE_NAME,
                libraries = C_LIBRARY + [
                     PYTHON_LIB_NAME,                     
                     "euser", "avkon", "apparc", 
                     "cone","eikcore", "libGLES_CM", "pygame_libjpeg",                    
                     SDL_DLL_NAME,
                     ],
                winscw_options = "-w noempty",
                uid3 = uid,
                )
    m = ".".join( [modname, targettype] )
    
    if HAVE_STATIC_MODULES:
        PYGAME_STATIC_MODULES.append( m )
     

def createPygameMods():
    """ Create pygame native modules """
    
    # Get the mods. Python wrappers can be conveniently used here.
    mods = [ os.path.basename( x ).replace(".py", "") for x in glob("lib/*.py") ]
    
    # Most of the modules have only 1 source file : <modname>.c
    # This dict can be used to map differing or multiple source files to module. 
    module_src_map = { 
        "surface" : (
            "surface.c",
            "scale2x.c",
            "surface_fill.c",
            "alphablit.c",            
        ),
        "gfxdraw" : ( 
            "gfxdraw.c", 
            "SDL_gfx/SDL_gfxPrimitives.c" 
        ),   
        "fastevent" : (
            "fastevents.c",
            "fastevent.c"
        ),
        "transform" : (
            "transform.c",
            "rotozoom.c",
            "scale2x.c"
        )
    }

    for x in mods:
        # Get source mapping
        src = module_src_map.get( x, [ x + ".c"])
        createPygameLibrary( x, src )
    
    # This one is special    
    createPygameLibrary( "mixer_music", ["music.c"])        
    
createPygameMods()

# Install pygame python libraries
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
    
def packagePyS60Stdlib(**kwargs):
    kwargs["source"] = join( "deps/PythonForS60/module-repo/standard-modules", kwargs["source"] )
    return to_package( **kwargs )

# Add files missing from standard PyS60 installation
packagePyS60Stdlib( source = "glob.py", target = "data/pygame/libs" )
zippath = packagePyS60Stdlib( source = "fnmatch.py", target = "data/pygame/libs" )

# Install default font into zip as well
File2Zip( zippath, "../lib/freesansbold.ttf", "pygame/freesansbold.ttf" )

# Export static library names to be used for building pygame.exe
Export( "PYGAME_STATIC_MODULES")
