import sys
import time
import os
import shutil

import build_config as config

from glob import glob
from scons_symbian.config.constants import CAPS_SELF_SIGNED

BASE_CMD = "scons dosis=true"

UID_PACKAGE   = 0xE0006020
UID_PYGAMEAPP = UID_PACKAGE + 1
UID_SDL       = UID_PYGAMEAPP + 1

#: Base uid for PyS60 CE scripts
UID_BASE = UID_SDL + 1

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
             "basename"     : "pygame_python",            
             "uidbase"      : hex(UID_BASE).replace("L",""),             
             "sisappname"   : '"Python for Pygame"',
             # Convert to int or may be converted to octal due to zero at beginning
             'sisversion'   : '"(1,%d,%d%s)"' % ( int(version[2:4]), int( version[4:6]),version[6:]),
             'pythonsis'    : sisname,
             'libpath'      : "data/pygame/libs",
             }
    
    # Build PyS60 
    curdir = os.getcwd()
    os.chdir(config.pys60_ce_src)
    
    if config.build_python:
        dobuild(args)
    
    os.chdir(curdir)
    
    sisname = "python_for_pygame_%s_signed.sisx" % version
    sissource = os.path.join( config.pys60_ce_src, sisname )
    
    # Copy the sis to current directory
    import shutil
    shutil.copyfile(sissource, sisname)
     
    # Build pygame
    args["pythonsis"]  = sisname
    args["pythonsis"]  = sisname
    args["sisappname"] = '"pygame for S60"'
    args["package"]    = "pygame_%s.sis" % version
    args['sisversion'] = '1,%d,%d%s' % ( int(version[2:4]), int( version[4:6]),version[6:])
    args['sisuid'] = hex(UID_PACKAGE).replace("L","")
    args['appuid'] = hex(UID_PYGAMEAPP).replace("L","")
    args['sdluid'] = hex(UID_SDL).replace("L","")
    
    dobuild(args)
    
    
if __name__ == "__main__":
    build() 
