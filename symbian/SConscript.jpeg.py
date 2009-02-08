""" Defines project for JPEG library """
Import("*")

from scons_symbian import *

Import("TARGET_NAME UID3 PACKAGE_NAME CAPABILITIES")

# Built as dll because needed by SDL and the application
target     = TARGET_NAME
targettype = "dll"
libraries  = C_LIBRARY + ['euser']

# Static libs
libraries += []

uid3 = UID3

sources = [
 'deps/jpeg/jcomapi.c',
 'deps/jpeg/jutils.c',
 'deps/jpeg/jerror.c',
 'deps/jpeg/jmemmgr.c',
 'deps/jpeg/jmemnobs.c',
 'deps/jpeg/jcapimin.c',
 'deps/jpeg/jcapistd.c',
 'deps/jpeg/jctrans.c',
 'deps/jpeg/jcparam.c',
 'deps/jpeg/jdatadst.c',
 'deps/jpeg/jcinit.c',
 'deps/jpeg/jcmaster.c',
 'deps/jpeg/jcmarker.c',
 'deps/jpeg/jcmainct.c',
 'deps/jpeg/jcprepct.c',
 'deps/jpeg/jccoefct.c',
 'deps/jpeg/jccolor.c',
 'deps/jpeg/jcsample.c',
 'deps/jpeg/jchuff.c',
 'deps/jpeg/jcphuff.c',
 'deps/jpeg/jcdctmgr.c',
 'deps/jpeg/jfdctfst.c',
 'deps/jpeg/jfdctflt.c',
 'deps/jpeg/jfdctint.c',
 'deps/jpeg/jdapimin.c',
 'deps/jpeg/jdapistd.c',
 'deps/jpeg/jdtrans.c',
 'deps/jpeg/jdatasrc.c',
 'deps/jpeg/jdmaster.c',
 'deps/jpeg/jdinput.c',
 'deps/jpeg/jdmarker.c',
 'deps/jpeg/jdhuff.c',
 'deps/jpeg/jdphuff.c',
 'deps/jpeg/jdmainct.c',
 'deps/jpeg/jdcoefct.c',
 'deps/jpeg/jdpostct.c',
 'deps/jpeg/jddctmgr.c',
 'deps/jpeg/jidctfst.c',
 'deps/jpeg/jidctflt.c',
 'deps/jpeg/jidctint.c',
 'deps/jpeg/jidctred.c',
 'deps/jpeg/jdsample.c',
 'deps/jpeg/jdcolor.c',
 'deps/jpeg/jquant1.c',
 'deps/jpeg/jquant2.c',
 'deps/jpeg/jdmerge.c']


includes    = []
sysincludes = ['/epoc32/include', C_INCLUDE ]
defines     = ['JPEG_DLL']

SymbianProgram( target, targettype,
    sources = sources,
    includes    = includes,
    sysincludes = sysincludes,
    libraries   = libraries,
    defines     = defines,
    capabilities = CAPABILITIES,
    uid3 = uid3,
    package = PACKAGE_NAME,
)

