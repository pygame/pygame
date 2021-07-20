from .sdl2 import * # pylint: disable=wildcard-import; lgtm[py/polluting-import]
from .audio import * # pylint: disable=wildcard-import; lgtm[py/polluting-import]
from .video import * # pylint: disable=wildcard-import; lgtm[py/polluting-import]

from .cyvideo import * # pylint: disable=wildcard-import; lgtm[py/polluting-import]

for attribute in dir(cyvideo):
    if not attribute.startswith("_"):
        setattr(video, attribute, getattr(cyvideo, attribute))

del cyvideo #remove from namespace
    
