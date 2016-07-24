Import("*")

# This file is generated with mmp2sconscript
from scons_symbian import *


target     = "libsft2"
targettype = "lib"
libraries  = C_LIBRARY + ['euser', 'gdi', 'fbscli']

# Static libs
libraries += []

sources = ['deps/sft2/src/libsft2.cpp',
 'deps/sft2/src/libsft2DllMain.cpp',
 'deps/sft2/src/base/ftapi.c',
 'deps/sft2/src/base/ftbase.c',
 'deps/sft2/src/base/ftbbox.c',
 'deps/sft2/src/base/ftbdf.c',
 'deps/sft2/src/base/ftbitmap.c',
 'deps/sft2/src/base/ftdebug.c',
 'deps/sft2/src/base/ftgasp.c',
 'deps/sft2/src/base/ftglyph.c',
 'deps/sft2/src/base/ftgxval.c',
 'deps/sft2/src/base/ftinit.c',
 'deps/sft2/src/base/ftlcdfil.c',
 'deps/sft2/src/base/ftmm.c',
 'deps/sft2/src/base/ftotval.c',
 'deps/sft2/src/base/ftpfr.c',
 'deps/sft2/src/base/ftstroke.c',
 'deps/sft2/src/base/ftsynth.c',
 'deps/sft2/src/base/ftsystem.c',
 'deps/sft2/src/base/fttype1.c',
 'deps/sft2/src/base/ftwinfnt.c',
 'deps/sft2/src/base/ftxf86.c',
 'deps/sft2/src/winfonts/winfnt.c',
 'deps/sft2/src/bdf/bdf.c',
 'deps/sft2/src/type42/type42.c',
 'deps/sft2/src/type1/type1.c',
 'deps/sft2/src/truetype/truetype.c',
 'deps/sft2/src/cid/type1cid.c',
 'deps/sft2/src/pcf/pcf.c',
 'deps/sft2/src/psaux/psaux.c',
 'deps/sft2/src/pfr/pfr.c',
 'deps/sft2/src/cff/cff.c',
 'deps/sft2/src/psnames/psnames.c',
 'deps/sft2/src/pshinter/pshinter.c',
 'deps/sft2/src/autofit/autofit.c',
 'deps/sft2/src/autofit/afwarp.c',
 'deps/sft2/src/gzip/ftgzip.c',
 'deps/sft2/src/smooth/smooth.c',
 'deps/sft2/src/raster/raster.c',
 'deps/sft2/src/sfnt/sfnt.c',
 'deps/sft2/src/lzw/ftlzw.c']


includes    = ['deps/sft2/inc']
sysincludes = [ C_INCLUDE, EPOC32_INCLUDE, 'deps/sft2/inc/sys']
defines     = ['FT2_BUILD_LIBRARY', 
    #'LOGN_ENABLE', 
    #'LOGP_ENABLE', 
    'cplusplus']

SymbianProgram( target, targettype,
    sources = sources,
    includes    = includes,
    sysincludes = sysincludes,
    libraries   = libraries,
    defines     = defines,        
)




