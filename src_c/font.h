#ifndef PGFONT_INTERNAL_H
#define PGFONT_INTERNAL_H

#ifdef PG_SDL3
#include <SDL3_ttf/SDL_ttf.h>

// SDL3_ttf uses SDL3 error reporting API
#define TTF_GetError SDL_GetError

#else
#include <SDL_ttf.h>
#endif
/* test font initialization */
#define FONT_INIT_CHECK()           \
    if (!(*(int *)PyFONT_C_API[2])) \
    return RAISE(pgExc_SDLError, "font system not initialized")

#include "include/pygame_font.h"

#define PYGAMEAPI_FONT_NUMSLOTS 3

#endif /* ~PGFONT_INTERNAL_H */
