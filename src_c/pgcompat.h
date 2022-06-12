/* Python 2.x/3.x compatibility tools (internal)
 */
#ifndef PGCOMPAT_INTERNAL_H
#define PGCOMPAT_INTERNAL_H

#include "include/pgcompat.h"

/* Module init function returns new module instance. */
#define MODINIT_DEFINE(mod_name) PyMODINIT_FUNC PyInit_##mod_name(void)

/* Defaults for unicode file path encoding */
#if defined(MS_WIN32)
#define UNICODE_DEF_FS_ERROR "replace"
#else
#define UNICODE_DEF_FS_ERROR "surrogateescape"
#endif

#define RELATIVE_MODULE(m) ("." m)

#ifndef Py_TPFLAGS_HAVE_NEWBUFFER
#define Py_TPFLAGS_HAVE_NEWBUFFER 0
#endif

#define Slice_GET_INDICES_EX(slice, length, start, stop, step, slicelength) \
    PySlice_GetIndicesEx(slice, length, start, stop, step, slicelength)

#if defined(SDL_VERSION_ATLEAST)
#if !(SDL_VERSION_ATLEAST(2, 0, 5))
/* These functions require SDL 2.0.5 or greater.

  https://wiki.libsdl.org/SDL_SetWindowResizable
*/
void
SDL_SetWindowResizable(SDL_Window *window, SDL_bool resizable);
int
SDL_GetWindowOpacity(SDL_Window *window, float *opacity);
int
SDL_SetWindowOpacity(SDL_Window *window, float opacity);
int
SDL_SetWindowModalFor(SDL_Window *modal_window, SDL_Window *parent_window);
int
SDL_SetWindowInputFocus(SDL_Window *window);
SDL_Surface *
SDL_CreateRGBSurfaceWithFormat(Uint32 flags, int width, int height, int depth,
                               Uint32 format);
#endif /* !(SDL_VERSION_ATLEAST(2, 0, 5)) */

/* SDL 2.0.22 provides some utility functions for FRects */
#if !(SDL_VERSION_ATLEAST(2, 0, 22))

#ifndef CODE_BOTTOM
#define CODE_BOTTOM 1
#endif
#ifndef CODE_TOP
#define CODE_TOP 2
#endif
#ifndef CODE_LEFT
#define CODE_LEFT 4
#endif
#ifndef CODE_RIGHT
#define CODE_RIGHT 8
#endif
//#endregion

static int
compat_ComputeOutCodeF(const SDL_FRect *rect, float x, float y)
{
    int code = 0;
    if (y < rect->y) {
        code |= CODE_TOP;
    }
    else if (y >= rect->y + rect->h) {
        code |= CODE_BOTTOM;
    }
    if (x < rect->x) {
        code |= CODE_LEFT;
    }
    else if (x >= rect->x + rect->w) {
        code |= CODE_RIGHT;
    }
    return code;
}

static SDL_bool
SDL_IntersectFRectAndLine(SDL_FRect *rect, float *X1, float *Y1, float *X2,
                          float *Y2)
{
    float x = 0;
    float y = 0;
    float x1, y1;
    float x2, y2;
    float rectx1;
    float recty1;
    float rectx2;
    float recty2;
    int outcode1, outcode2;

    /* Special case for empty rect */
    if ((!rect) || (rect->w <= 0) || (rect->h <= 0)) {
        return SDL_FALSE;
    }

    x1 = *X1;
    y1 = *Y1;
    x2 = *X2;
    y2 = *Y2;
    rectx1 = rect->x;
    recty1 = rect->y;
    rectx2 = rect->x + rect->w - 1;
    recty2 = rect->y + rect->h - 1;

    /* Check to see if entire line is inside rect */
    if (x1 >= rectx1 && x1 <= rectx2 && x2 >= rectx1 && x2 <= rectx2 &&
        y1 >= recty1 && y1 <= recty2 && y2 >= recty1 && y2 <= recty2) {
        return SDL_TRUE;
    }

    /* Check to see if entire line is to one side of rect */
    if ((x1 < rectx1 && x2 < rectx1) || (x1 > rectx2 && x2 > rectx2) ||
        (y1 < recty1 && y2 < recty1) || (y1 > recty2 && y2 > recty2)) {
        return SDL_FALSE;
    }

    if (y1 == y2) {
        /* Horizontal line, easy to clip */
        if (x1 < rectx1) {
            *X1 = rectx1;
        }
        else if (x1 > rectx2) {
            *X1 = rectx2;
        }
        if (x2 < rectx1) {
            *X2 = rectx1;
        }
        else if (x2 > rectx2) {
            *X2 = rectx2;
        }
        return SDL_TRUE;
    }

    if (x1 == x2) {
        /* Vertical line, easy to clip */
        if (y1 < recty1) {
            *Y1 = recty1;
        }
        else if (y1 > recty2) {
            *Y1 = recty2;
        }
        if (y2 < recty1) {
            *Y2 = recty1;
        }
        else if (y2 > recty2) {
            *Y2 = recty2;
        }
        return SDL_TRUE;
    }

    /* More complicated Cohen-Sutherland algorithm */
    outcode1 = compat_ComputeOutCodeF(rect, x1, y1);
    outcode2 = compat_ComputeOutCodeF(rect, x2, y2);
    while (outcode1 || outcode2) {
        if (outcode1 & outcode2) {
            return SDL_FALSE;
        }

        if (outcode1) {
            if (outcode1 & CODE_TOP) {
                y = recty1;
                x = x1 + ((x2 - x1) * (y - y1)) / (y2 - y1);
            }
            else if (outcode1 & CODE_BOTTOM) {
                y = recty2;
                x = x1 + ((x2 - x1) * (y - y1)) / (y2 - y1);
            }
            else if (outcode1 & CODE_LEFT) {
                x = rectx1;
                y = y1 + ((y2 - y1) * (x - x1)) / (x2 - x1);
            }
            else if (outcode1 & CODE_RIGHT) {
                x = rectx2;
                y = y1 + ((y2 - y1) * (x - x1)) / (x2 - x1);
            }
            x1 = x;
            y1 = y;
            outcode1 = compat_ComputeOutCodeF(rect, x, y);
        }
        else {
            if (outcode2 & CODE_TOP) {
                y = recty1;
                x = x1 + ((x2 - x1) * (y - y1)) / (y2 - y1);
            }
            else if (outcode2 & CODE_BOTTOM) {
                y = recty2;
                x = x1 + ((x2 - x1) * (y - y1)) / (y2 - y1);
            }
            else if (outcode2 & CODE_LEFT) {
                /* If this assertion ever fires, here's the static analysis
                   that warned about it:
                   http://buildbot.libsdl.org/sdl-static-analysis/sdl-macosx-static-analysis/sdl-macosx-static-analysis-1101/report-b0d01a.html#EndPath
                 */
                SDL_assert(x2 != x1); /* if equal: division by zero. */
                x = rectx1;
                y = y1 + ((y2 - y1) * (x - x1)) / (x2 - x1);
            }
            else if (outcode2 & CODE_RIGHT) {
                /* If this assertion ever fires, here's the static analysis
                   that warned about it:
                   http://buildbot.libsdl.org/sdl-static-analysis/sdl-macosx-static-analysis/sdl-macosx-static-analysis-1101/report-39b114.html#EndPath
                 */
                SDL_assert(x2 != x1); /* if equal: division by zero. */
                x = rectx2;
                y = y1 + ((y2 - y1) * (x - x1)) / (x2 - x1);
            }
            x2 = x;
            y2 = y;
            outcode2 = compat_ComputeOutCodeF(rect, x, y);
        }
    }
    *X1 = x1;
    *Y1 = y1;
    *X2 = x2;
    *Y2 = y2;
    return SDL_TRUE;
}
#endif /* !(SDL_VERSION_ATLEAST(2, 0, 22)) */
#endif /* defined(SDL_VERSION_ATLEAST) */

/* incase it is defined in the future by Python.h */
#ifndef PyFloat_FromFloat
#define PyFloat_FromFloat(x) \
    (PyFloat_FromDouble((double)(round((x)*100000) / 100000)))
#endif /* PyFloat_FromFloat */

#endif /* ~PGCOMPAT_INTERNAL_H */
