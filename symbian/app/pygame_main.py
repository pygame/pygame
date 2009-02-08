"""
pygame main script for S60
"""
import os
import sys

f = None
if sys.platform == "symbian_s60":
    f=open('/data/pygame/stdout.txt','w')
    sys.stdout = f
    sys.stderr = f


__file__ = sys.argv[0]
THISDIR = os.path.dirname( __file__ )
sys.path.append( os.path.join( THISDIR, "libs") )

path_to_app = os.path.join( THISDIR, "launcher", "pygame_launcher.py" )
execfile(path_to_app, {'__builtins__': __builtins__,
                   '__name__': '__main__',
                   '__file__': path_to_app } )

if f is not None:
    sys.stdout.flush()
    sys.stdout.close()
