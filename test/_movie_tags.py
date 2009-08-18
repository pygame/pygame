__tags__ = []
import sys

# only supported on linux so far.
if (sys.platform == 'linux2'):
    exclude = False
else:
    exclude = True

if exclude:
    __tags__.extend(['ignore', 'subprocess_ignore'])

