if __import__("sys").platform not in ("wasi", "emscripten"):
    from .audio import *  # pylint: disable=wildcard-import; lgtm[py/polluting-import]
    from .sdl2 import *  # pylint: disable=wildcard-import; lgtm[py/polluting-import]
    from .video import *  # pylint: disable=wildcard-import; lgtm[py/polluting-import]
