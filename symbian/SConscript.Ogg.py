Import("*")

# This file is generated with mmp2sconscript
from scons_symbian import *

target     = "ogg"
targettype = "lib"
libraries  = []
# Static libs
libraries += []

uid3 = 0
sources = ['deps/ogg/src/bitwise.c', 'deps/ogg/src/framing.c']

includes    = ['/SDLS60/symbian']
sysincludes = [ EPOC32_INCLUDE, C_INCLUDE, 'deps/ogg/include']
defines     = []

SymbianProgram( target, targettype,
    sources = sources,
    includes    = includes,
    sysincludes = sysincludes,
    libraries   = libraries,
    defines     = defines,
    epocstacksize = 8192,
    epocheapsize  = (0x400,0x100000),
    uid3 = uid3,
)

