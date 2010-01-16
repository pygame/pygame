"""Examples package for Pygame2.

This package contains code snippets demonstrating the aspects of the
various Pygame2 modules. Each example is an module of its own and can be
easily executed using

    python -m pygame2.examples.<module>.<example>

To run the drawing pygame2.sdlext example, you would type

    python -m pygame2.examples.sdlext.draw

"""

import os
from pygame2.resources import Resources

_filepath = os.path.dirname (os.path.abspath (__file__))
RESOURCES = Resources (os.path.join (_filepath, "resources"), ".*\.svn\.*")
