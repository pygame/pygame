# module msysio.py
# Requires Python 2.2 or better.

"""Provide helpful routines for interactive IO on the MSYS console"""

# Output needs to be flushed to be seen. It is especially important
# when prompting for user input.

import sys
import os

__all__ = ['raw_input_', 'print_', 'is_msys']

# 2.x/3.x compatibility stuff
try:
    raw_input
except NameError:
    raw_input = input

# Exported functions
__all__ = ['raw_input_', 'print_', 'is_msys']

# 2.x/3.x compatibility stuff
try:
    raw_input
except NameError:
    raw_input = input

# Exported functions
def raw_input_(prompt=None):
    """Prompt for user input in an MSYS console friendly way"""
    if prompt is None:
        prompt = ''
    print_(prompt, end='')
    return raw_input()

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
        return os.environ['OSTYPE'] == 'msys'
    except KeyError:
        return False
