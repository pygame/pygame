__tags__ = []
import sys

# only supported on linux so far.
if (sys.platform == 'linux2'):
    exclude = False
else:
    exclude = True

# not portable to Python 3 yet.
exclude = exclude or sys.version_info > (3,)

if exclude:
    __tags__.extend(['ignore', 'subprocess_ignore'])

