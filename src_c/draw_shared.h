/*
 * An attempt to share draw functions between draw and gfxdraw with a view to
   eventually deprecating new code from gfxdraw.
 */

// validation of a draw color
#define CHECK_LOAD_COLOR(colorobj)                                         \
    if (PyLong_Check(colorobj))                                            \
        color = (Uint32)PyLong_AsLong(colorobj);                           \
    else if (pg_RGBAFromFuzzyColorObj(colorobj, rgba))                     \
        color =                                                            \
            SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]); \
    else                                                                   \
        return NULL; /* pg_RGBAFromFuzzyColorObj sets the exception for us */

int
set_at(SDL_Surface *surf, int x, int y, Uint32 color);

void
draw_rect(SDL_Surface *surf, int x1, int y1, int x2, int y2, int width,
          Uint32 color);

void
drawhorzline(SDL_Surface *surf, Uint32 color, int x1, int y1, int x2);
