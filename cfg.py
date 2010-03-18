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

build = {}

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
build['SDL'] = istrue (os.getenv ("WITH_SDL", True))

# SDL_mixer support
build['SDL_MIXER'] = istrue (os.getenv ("WITH_SDL_MIXER", True)) and build['SDL']

# SDL_image support
build['SDL_IMAGE'] = istrue (os.getenv ("WITH_SDL_IMAGE", True)) and build['SDL']

# SDL_ttf support
build['SDL_TTF'] = istrue (os.getenv ("WITH_SDL_TTF", True)) and build['SDL']

# SDL_gfx support
build['SDL_GFX'] = istrue (os.getenv ("WITH_SDL_GFX", True)) and build['SDL']

# libpng support
# This is used by Surface.save() to enable PNG saving.
build['PNG'] = istrue (os.getenv ("WITH_PNG", True))

# libjpeg support
# This is used by Surface.save() to enable JPEG saving.
build['JPEG'] = istrue (os.getenv ("WITH_JPEG", True))

# freetype (module) support
build['FREETYPE'] = istrue (os.getenv ("WITH_FREETYPE", True))

# midi (module) support
build['PORTMIDI'] = istrue (os.getenv ("WITH_PORTMIDI", True))

# OpenAL (module) support
build['OPENAL'] = istrue (os.getenv ("WITH_OPENAL", True))

# Open Multiprocessing support
build['OPENMP'] = istrue (os.getenv ("WITH_OPENMP", False))

# Experimental modules support
build['EXPERIMENTAL'] = istrue (os.getenv ("WITH_EXPERIMENTAL", False))
