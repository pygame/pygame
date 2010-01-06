"""Examples package for Pygame2.

This package contains code snippets demonstrating the aspects of the
various Pygame2 modules. Each example is an module of its own and can be
easily executed using

    python -m pygame2.examples.<module>.<example>

To run the drawing pygame2.sdlext example, you would type

    python -m pygame2.examples.sdlext.draw

"""

import os

_filepath = os.path.abspath (__file__)
RESOURCEDIR = os.path.join (os.path.dirname (_filepath), "resources")
FONTDIR = os.path.join (os.path.dirname (_filepath), "resources")
IMAGEDIR = os.path.join (os.path.dirname (_filepath), "resources")
