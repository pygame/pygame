""" Full build configuration for pygame.
Reads configuration and creates appropriate parameters for SCons build scripts.
For SVN support, install pysvn.
"""

import sys
import time
import os
import shutil

import build_config as config

from glob import glob
from scons_symbian.config.constants import CAPS_SELF_SIGNED
from scons_symbian.arguments import COMPILER, RELEASE


BASE_CMD = "scons dosis=true"

UID_PACKAGE   = 0xE0006020
__uid = UID_PACKAGE 
def getuid(): 
    global __uid
    __uid += 1
    return __uid  

UID_PYGAMEAPP = getuid()
UID_SDL       = getuid()
UID_JPEG      = getuid()
UID_BASE      = getuid()

#: Capability configuration
CAPABILITIES = CAPS_SELF_SIGNED[:]
CAPABILITIES.remove("UserEnvironment") # Missing from sdl.dll
CAPABILITIES = "+".join(CAPABILITIES)

def dobuild(args):    
    cmd = ""
    for x in args:
        cmd += "%s=%s " % ( x, str(args[x]) )
    
    cmd = " ".join( [BASE_CMD, cmd] )
    cmd = " ".join( [cmd] + sys.argv[1:] )

    print cmd
    if os.system( cmd ): 
        raise SystemExit( "Error: Build failed" )
    

def build():

    version = list(time.gmtime()[:3])    
    version = [ str(x).zfill(2) for x in version ]
    version = "".join( version )
        
    sisname = "python_for_pygame_%s.sis" % version
    
    args = { "applications" : "",
             "capabilities" : CAPABILITIES,
             "builtin"      : "sysinfo,socket",
             #"pyds"         : "gles",
             "basename"     : "pygame_python",            
             "uidbase"      : hex(UID_BASE).replace("L",""),             
             "sisappname"   : '"Python for Pygame"',
             # Convert to int or may be converted to octal due to zero at beginning
             'sisversion'   : '"(1,%d,%d%s)"' % ( int(version[2:4]), int( version[4:6]),version[6:]),
             'pythonsis'    : sisname,
             'libpath'      : "data/pygame/libs",
             'pycompiler'   : config.pycompiler
             }
    
    # Add certificate stuff
    if config.cert is not None:
        args['cert'] = config.cert        
        args['privkey'] = config.privkey
        args['passphrase'] = config.passphrase
        
    # Build PyS60 CE
    sisname   = ""  
    if config.build_python:
        curdir = os.getcwd()
        os.chdir(config.pys60_ce_src)   
        a = args.copy()
        a["gcce_options"] = '"-O2 -fno-unit-at-a-time"'             
        dobuild(a)    
        os.chdir(curdir)
    
        sisname = "python_for_pygame_%s_signed.sisx" % version
        pys60_sis = os.path.join( config.pys60_ce_src, sisname )
                
        # Copy the sis to current directory
        if os.path.exists(pys60_sis):
            import shutil
            shutil.copyfile(pys60_sis, sisname)
        
        args['pythondll'] =  args['basename']
         
    else:
        sisname = config.pys60_sis
        if config.pythondll is not None:
            args['pythondll'] = config.pythondll
     
    # Build pygame
    args["pythonsis"]  = sisname
    args["pythonsis"]  = sisname
    args["sisappname"] = '"pygame"' 
    args['sisversion'] = '1,%d,%d%s' % ( int(version[2:4]), int( version[4:6]),version[6:])
    args['sisuid'] = hex(UID_PACKAGE).replace("L","")
    args['appuid'] = hex(UID_PYGAMEAPP).replace("L","")
    args['sdluid'] = hex(UID_SDL).replace("L","")
    args['jpeguid']= hex(UID_JPEG).replace("L","")
    args['pythoninclude'] = config.pythoninclude
    args["staticmods"]    = config.staticmods
    args["modbaseuid"]    = hex(UID_BASE).replace("L","")
    args["build_libs"]    = config.build_libs
    #args['defines'] = "STDERR_TO_STDOUT"
    args['winscw_options'] = "-nostderr" # Don't show the stdioserver output
    dobuild(args)
    
    
if __name__ == "__main__":
    build() 
