import os
##
## You can disable the components you do not want to support by setting them
## to False. If you e.g. do not need or want SDL_mixer support, simply set
##
##    WITH_SDL_MIXER = False
##
## In case certain parts do not build, try to disable them.
## 
##

import sys

if sys.version_info[0] >= 3:
    unicode = str

# Check function for the environment flags.
def istrue (val):
    if val is None:
        return False
    if type (val) in (str, unicode):
        return val.lower () in ("yes", "true", "1")
    return val in (1, True)

# SDL support.
# This MUST be enabled for the other SDL related modules.
WITH_SDL = istrue (os.getenv ("WITH_SDL", True))

# SDL_mixer support
WITH_SDL_MIXER = istrue (os.getenv ("WITH_SDL_MIXER", True)) and WITH_SDL

# SDL_image support
WITH_SDL_IMAGE = istrue (os.getenv ("WITH_SDL_IMAGE", True)) and WITH_SDL

# SDL_ttf support
WITH_SDL_TTF = istrue (os.getenv ("WITH_SDL_TTF", True)) and WITH_SDL

# SDL_gfx support
WITH_SDL_GFX = istrue (os.getenv ("WITH_SDL_GFX", True)) and WITH_SDL

# libpng support
# This is used by Surface.save() to enable PNG saving.
WITH_PNG = istrue (os.getenv ("WITH_PNG", True))

# libjpeg support
# This is used by Surface.save() to enable JPEG saving.
WITH_JPEG = istrue (os.getenv ("WITH_JPEG", True))
