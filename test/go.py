"""Load and run the Pygame test suite

pygame -c "import pygame.tests.go" [<test options>]

or

pygame test.go [<test options>]

Command line option --help displays a command line usage message.

run_tests.py in the main installation directory is an alternative to test.go

"""

if __name__ == '__main__':
    import sys
    import os
    pkg_dir = os.path.split(os.path.abspath(__file__))[0]
    parent_dir, pkg_name = os.path.split(pkg_dir)
    is_pygame_pkg = (pkg_name == 'tests' and
                     os.path.split(parent_dir)[1] == 'pygame')
    if not is_pygame_pkg:
        sys.path.insert(0, parent_dir)
else:
    is_pygame_pkg = __name__.startswith('pygame.tests.')

if is_pygame_pkg:
    from pygame.tests import run
else:
    from test import run

if __name__ == '__main__':
    run('go.py')
else:
    run('python -c "%s"' % __name__)

