"""Load and run a Pygame example program

python -c "import pygame.examples.go" <example name> [<example args>]

Commend line option --list displays a list or example programs.

python -c "import pygame.examples.go" --list

"""

import sys
import os


examples_dir = os.path.dirname(__file__)
if __name__ != '__main__' and (len(sys.argv) == 1 or sys.argv[0] != '-c'):
    print ("usage:\n%s" % __doc__)
elif sys.argv[1] == '--list':
    import glob
    for filename in glob.glob(os.path.join(examples_dir, '*.py')):
        name = os.path.splitext(os.path.basename(filename))[0]
        if name not in ('go', '__init__'):
            print (name)
else:
    program_name = sys.argv[1]
    if not program_name.lower().endswith('.py'):
        program_name += '.py'
    del sys.argv[1]
    _go__name__ = __name__
    __name__ = '__main__'
    try:
        exec (open(os.path.join(examples_dir, program_name)).read())
    finally:
        __name__ = _go__name__


