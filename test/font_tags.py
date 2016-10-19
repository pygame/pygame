__tags__ = []

import sys
# Font support not fully implemented for Python 3.x.
if sys.version_info >= (3, 0, 0) and sys.platform.startswith('win'):
    __tags__.extend(['ignore', 'subprocess_ignore'])

