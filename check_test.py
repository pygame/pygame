# Program check_test.py
# Requires Python 2.4

"""A program for listing the modules accessed by a Pygame unit test module

Usage:

python check_test.py <test module>

e.g.

python check_test.py surface_test.py

The returned list will show which Pygame modules were imported and accessed.
Each module name is followed by a list of attributes accessed.

"""

import sys
import os
import trackmod
trackmod.begin(pattern=['pygame', 'pygame.*'],
               continuous=True,
               submodule_accesses=False)
skip = set(['pygame.locals', 'pygame.constants',
            'pygame.base', 'pygame.threads'])

os.chdir('test')
test_file = sys.argv[1]
del sys.argv[1]
try:
    execfile(test_file)
finally:
    trackmod.end()
    print "=== Pygame package submodule accesses ==="
    print
    accesses = [(n, a) for n, a in trackmod.get_accesses().iteritems()
                       if n not in skip]
    accesses.sort(key=lambda t: t[0])
    for name, attributes in accesses:
        print "%s (%s)" % (name, ', '.join(attributes))
