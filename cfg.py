##
## You can disable the components you do not want to support by setting them
## to False. If you e.g. do not need or want SDL_mixer support, simply set
##
##    WITH_SDL_MIXER = False
##
## In case certain parts do not build, try to disable them.
## 
##

# SDL support.
# This MUST be enabled for the other SDL related modules.
WITH_SDL = True

# SDL_mixer support
WITH_SDL_MIXER = True

# SDL_image support
WITH_SDL_IMAGE = True

# SDL_ttf support
WITH_SDL_TTF = True

# SDL_gfx support
WITH_SDL_GFX = True

# libpng support
# This is used by Surface.save() to enable PNG saving.
WITH_PNG = True

# libjpeg support
# This is used by Surface.save() to enable JPEG saving.
WITH_JPEG = True
