""" Defines project for PNG library """

Import("*")

# This file is generated with mmp2sconscript
from scons_symbian import *
from glob import glob

Import("TARGET_NAME UID3 PACKAGE_NAME")

# Built as dll because needed by SDL and the application
target     = TARGET_NAME
targettype = TARGETTYPE_LIB
libraries  = C_LIBRARY + ['euser']

# Static libs
libraries += []

#uid3 = UID3

sources = glob( "deps/libpng/png*.c")
[ sources.remove(x) for x in glob("deps/libpng/pngtes*.c")]


includes    = ['deps/libpng/']
sysincludes = ['/epoc32/include', C_INCLUDE ]
defines     = []

SymbianProgram( target, targettype,
    sources = sources,
    includes    = includes,
    sysincludes = sysincludes,
    libraries   = libraries,
    defines     = defines,
    #uid3 = uid3, 
    package = PACKAGE_NAME,
)

