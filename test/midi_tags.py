__tags__ = []

import sys
# pymp not ported to Python 3.x.
if sys.version_info >= (3, 0, 0):
    __tags__.extend(['ignore', 'subprocess_ignore'])
