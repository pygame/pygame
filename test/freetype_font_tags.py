__tags__ = []
import sys

exclude = sys.version_info > (3,) # Remove when builds for Python 3.1

if exclude:
    __tags__.extend(['ignore', 'subprocess_ignore'])

