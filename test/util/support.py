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
