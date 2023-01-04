import sys

if sys.platform not in ("wasi", "emscripten"):
    from .sdl2 import *  # pylint: disable=wildcard-import; lgtm[py/polluting-import]
    from .audio import *  # pylint: disable=wildcard-import; lgtm[py/polluting-import]
    from .video import *  # pylint: disable=wildcard-import; lgtm[py/polluting-import]
