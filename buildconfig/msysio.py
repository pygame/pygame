# module msysio.py
# Requires Python 2.2 or better.

"""Provide helpful routines for interactive IO on the MSYS console"""

# Output needs to be flushed to be seen. It is especially important
# when prompting for user input.

import sys
import os

__all__ = ['print_', 'is_msys']


def print_(*args, **kwds):
    """Print arguments in an MSYS console friendly way

    Keyword arguments:
        file, sep, end
    """

    stream = kwds.get('file', sys.stdout)
    sep = kwds.get('sep', ' ')
    end = kwds.get('end', '\n')

    if args:
        stream.write(sep.join([str(arg) for arg in args]))
    if end:
        stream.write(end)
    try:
        stream.flush()
    except AttributeError:
        pass

def is_msys():
    """Return true if the execution environment is MSYS"""

    try:
        # Unfortunately there is no longer an MSYS specific identifier.
        return os.environ['TERM'] == 'cygwin'
    except KeyError:
        return False
