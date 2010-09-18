__tags__ = []
__tags__ = ['ignore', 'subprocess_ignore']  #temporary to find Mac build error

import sys
# Font support not fully implemented for Python 3.x.
if sys.version_info >= (3, 0, 0) and sys.platform.startswith('Win'):
    __tags__.extend(['ignore', 'subprocess_ignore'])

