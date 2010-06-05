import os, sys

try: 
    import StringIO
except ImportError:
    import io as StringIO

"""Utility functions for the tests."""

def redirect_output ():
    """Redirects stderr and stdout into own streams."""
    yield sys.stderr, sys.stdout
    sys.stderr, sys.stdout = StringIO.StringIO(), StringIO.StringIO()
    yield sys.stderr, sys.stdout
    
def restore_output (err, out):
    """Restores stderr and stdout using the passed streams."""
    sys.stderr, sys.stdout = err, out

class StreamOutput (object):
    def __init__ (self, stream):
        self.stream = stream
        self.startoffset = self.stream.tell ()
        self.curoffset = 0

    def writeline (self, data=None):
        if data:
            self.stream.write (data)
        self.stream.write (os.linesep)
        if data:
            self.curoffset = len (data)
        else:
            self.curoffset = 0
        self.stream.flush ()

    def write (self, data):
        self.stream.write (data)
        self.curoffset = len (data)
        self.stream.flush ()

    def writesame (self, data):
        overhang = self.curoffset - len (data)
        if overhang > 0:
            self.stream.write (data + " " * overhang + "\r")
        else:
            self.stream.write (data + "\r")
        self.curoffset = len (data)
        self.stream.flush ()
