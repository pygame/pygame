"""
pygame main script for S60. This initializes some system defaults and 
calls the pygame script given in sys.argv[1]
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
if len(sys.argv) < 2:
    path_to_app = os.path.join( THISDIR, "launcher", "pygame_launcher.py" )
    if sys.platform == "symbian_s60":
        import e32
        datapath = os.path.join( THISDIR, "launcher", "startapp.txt" )
        if os.path.exists(datapath):
            # Read the application script path from a file stored by launcher.
            df = open(datapath)
            data = df.read()
            df.close()
            path_to_app = data.strip()
            
            # TODO: Make sure previous pygame.exe has closed first
            import time
            time.sleep(1)
                         
            e32.start_exe( "pygame.exe", path_to_app, 1)
            
            e32.start_exe( "pygame.exe", "")
            
        
else:

    path_to_app = sys.argv[1]
    if sys.platform == "symbian_s60" and "apps" in path_to_app:
        # Use separate file so it won't be overwritten when the launcher restarts.
        fold = f
        f=open('/data/pygame/appout.txt','w')
        sys.stdout = f
        sys.stderr = f
        fold.close()

try:
    execfile(path_to_app, {'__builtins__': __builtins__,
                   '__name__': '__main__',
                   '__file__': path_to_app } )
except:
    import traceback
    traceback.print_exc()
    
if f is not None:
    sys.stdout.flush()
    sys.stdout.close()
