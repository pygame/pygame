/*
  pygame - Python Game Library
  Copyright (C) 2007-2008 Marcus von Appen

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Library General Public
  License as published by the Free Software Foundation; either
  version 2 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Library General Public License for more details.

  You should have received a copy of the GNU Library General Public
  License along with this library; if not, write to the Free
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

*/

#include "surface.h"

/* Simple weighted euclidian distance, which tries to get near to the
 * human eye reception using the weights.
 * It receives RGB values in the range 0-255 and returns a distance
 * value between 0.0 and 1.0.
 */
#define COLOR_DIFF_RGB(wr,wg,wb,r1,g1,b1,r2,g2,b2) \
    (sqrt (wr * (r1 - r2) * (r1 - r2) + \
           wg * (g1 - g2) * (g1 - g2) + \
           wb * (b1 - b2) * (b1 - b2)) / 255.0)

#define WR_NTSC 0.299
#define WG_NTSC 0.587
#define WB_NTSC 0.114

/**
 * Creates a new surface using the currently applied dimensions, step
 * size, etc.
 */
static PyObject*
_make_surface(PyPixelArray *array)
{
    PyObject *newsf;
    SDL_Surface *tmpsf;
    SDL_Surface *newsurf;
    Uint8 *pixels;
    Uint8 *origpixels;

    SDL_Surface *surface;
    int bpp;
    Uint32 x = 0;
    Uint32 y = 0;
    Uint32 vx = 0;
    Uint32 vy = 0;
    Uint32 posx = 0;
    Uint32 posy = 0;
    Uint32 absxstep;
    Uint32 absystep;

    surface = ((PySDLSurface*)array->surface)->surface;
    bpp = surface->format->BytesPerPixel;

    /* Create the second surface. */
    tmpsf = SDL_CreateRGBSurface (surface->flags,
        (int) (array->xlen / ABS (array->xstep)),
        (int) (array->ylen / ABS (array->ystep)), bpp, surface->format->Rmask,
        surface->format->Gmask, surface->format->Bmask, surface->format->Amask);
    if (!tmpsf)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }

    /* Guarantee an identical format. */
    newsurf = SDL_ConvertSurface (tmpsf, surface->format, surface->flags);
    if (!newsurf)
    {
        SDL_FreeSurface (tmpsf);
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    SDL_FreeSurface (tmpsf);
    
    newsf = PySDLSurface_NewFromSDLSurface (newsurf);
    if (!newsf)
    {
        SDL_FreeSurface (newsurf);
        return NULL;
    }

    /* Acquire a temporary lock. */
    if (SDL_MUSTLOCK (newsurf) == 0)
        SDL_LockSurface (newsurf);

    pixels = (Uint8 *) newsurf->pixels;
    origpixels = (Uint8 *) surface->pixels;

    absxstep = ABS (array->xstep);
    absystep = ABS (array->ystep);
    y = array->ystart;

    Py_BEGIN_ALLOW_THREADS;
    /* Single value assignment. */
    switch (bpp)
    {
    case 1:
        while (posy < array->ylen)
        {
            vx = 0;
            x = array->xstart;
            posx = 0;
            while (posx < array->xlen)
            {
                *((Uint8 *) pixels + vy * newsurf->pitch + vx) =
                    (Uint8)*((Uint8 *) origpixels + y * array->padding + x);
                vx++;
                x += array->xstep;
                posx += absxstep;
            }
            vy++;
            y += array->ystep;
            posy += absystep;
        }
        break;
    case 2:
        while (posy < array->ylen)
        {
            vx = 0;
            x = array->xstart;
            posx = 0;
            while (posx < array->xlen)
            {
                *((Uint16 *) (pixels + vy * newsurf->pitch) + vx) =
                    (Uint16)*((Uint16 *) (origpixels + y * array->padding) + x);
                vx++;
                x += array->xstep;
                posx += absxstep;
            }
            vy++;
            y += array->ystep;
            posy += absystep;
        }
        break;
    case 3:
    {
        Uint8 *px;
        Uint8 *vpx;
        SDL_PixelFormat *format = newsurf->format;
        SDL_PixelFormat *vformat = surface->format;

        while (posy < array->ylen)
        {
            vx = 0;
            x = array->xstart;
            posx = 0;
            while (posx < array->xlen)
            {
                px = ((Uint8 *) (pixels + vy * newsurf->pitch) + vx * 3);
                vpx = ((Uint8 *) (origpixels + y * array->padding) + x * 3);

#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
                *(px + (format->Rshift >> 3)) =
                    *(vpx + (vformat->Rshift >> 3));
                *(px + (format->Gshift >> 3)) =
                    *(vpx + (vformat->Gshift >> 3));
                *(px + (format->Bshift >> 3)) =
                    *(vpx + (vformat->Bshift >> 3));
#else
                *(px + 2 - (format->Rshift >> 3)) =
                    *(vpx + 2 - (vformat->Rshift >> 3));
                *(px + 2 - (format->Gshift >> 3)) =
                    *(vpx + 2 - (vformat->Gshift >> 3));
                *(px + 2 - (format->Bshift >> 3)) =
                    *(vpx + 2 - (vformat->Bshift >> 3));
#endif
                vx++;
                x += array->xstep;
                posx += absxstep;
            }
            vy++;
            y += array->ystep;
            posy += absystep;
        }
        break;
    }
    default:
        while (posy < array->ylen)
        {
            vx = 0;
            x = array->xstart;
            posx = 0;
            while (posx < array->xlen)
            {
                *((Uint32 *) (pixels + vy * newsurf->pitch) + vx) =
                    (Uint32)*((Uint32 *) (origpixels + y * array->padding) + x);
                vx++;
                x += array->xstep;
                posx += absxstep;
            }
            vy++;
            y += array->ystep;
            posy += absystep;
        }
        break;
    }
    Py_END_ALLOW_THREADS;

    if (SDL_MUSTLOCK (newsurf) == 0)
        SDL_UnlockSurface (newsurf);
    return newsf;
}

static int
_get_weights (PyObject *weights, double *wr, double *wg, double *wb)
{
    int success = 1;
    double rgb[3] = { 0 };
    
    if (!weights)
    {
        *wr = WR_NTSC;
        *wg = WG_NTSC;
        *wb = WB_NTSC;
        return 1;
    }
    
    if (!PySequence_Check (weights))
    {
        PyErr_SetString (PyExc_TypeError, "weights must be a sequence");
        success = 0;
    }
    else if (PySequence_Size (weights) < 3)
    {
        PyErr_SetString (PyExc_TypeError,
            "weights must contain at least 3 values");
        success = 0;
    }
    else
    {
        PyObject *item;
        int i;
        
        for (i = 0; i < 3; i++)
        {
            item = PySequence_ITEM (weights, i);
            if (item && PyNumber_Check (item))
            {
                PyObject *num = NULL;
                if ((num = PyNumber_Float (item)) != NULL)
                {
                    rgb[i] = PyFloat_AsDouble (num);
                    Py_DECREF (num);
                }
                else if ((num = PyNumber_Int (item)) != NULL)
                {
                    rgb[i] = (double) PyInt_AsLong (num);
                    if (rgb[i] == -1 && PyErr_Occurred ())
                        success = 0;
                    Py_DECREF (num);
                }
                else if ((num = PyNumber_Long (item)) != NULL)
                {
                    rgb[i] = (double) PyLong_AsLong (num);
                    if (PyErr_Occurred () &&
                        PyErr_ExceptionMatches (PyExc_OverflowError))
                        success = 0;
                    Py_DECREF (num);
                }
            }
            else
            {
                PyErr_SetString (PyExc_TypeError, "invalid weights");
                success = 0;
            }
            Py_XDECREF (item);
            if (!success)
                break;
        }
    }
    
    if (success)
    {
        double sum = 0;
        
        *wr = rgb[0];
        *wg = rgb[1];
        *wb = rgb[2];
        if ((*wr < 0 || *wg < 0 || *wb < 0) ||
            (*wr == 0 && *wg == 0 && *wb == 0))
        {
            PyErr_SetString (PyExc_ValueError,
                "weights must be positive and greater than 0");
            return 0;
        }
        /* Build the average weight values. */
        sum = *wr + *wg + *wb;
        *wr = *wr / sum;
        *wg = *wg / sum;
        *wb = *wb / sum;
        
        return success;
    }
    return 0;
}

static PyObject*
_replace_color (PyPixelArray *array, PyObject *args, PyObject *kwds)
{
    PyObject *weights = NULL;
    PyObject *delcolor = NULL;
    PyObject *replcolor = NULL;
    Uint32 dcolor;
    Uint32 rcolor;
    Uint8 r1, g1, b1, r2, g2, b2, a2;
    SDL_Surface *surface;
    double distance = 0;
    double wr, wg, wb;

    Uint32 x = 0;
    Uint32 y = 0;
    Uint32 posx = 0;
    Uint32 posy = 0;
    Sint32 absxstep;
    Sint32 absystep;
    Uint8 *pixels;

    static char *keys[] = { "color", "repcolor", "distance", "weights", NULL };
    
    if (!PyArg_ParseTupleAndKeywords (args, kwds, "OO|dO", keys, &delcolor,
            &replcolor, &distance, &weights))
        return NULL;

    if (distance < 0 || distance > 1)
    {
        PyErr_SetString (PyExc_ValueError,
            "distance must be in the range from 0.0 to 1.0");
        return NULL;
    }

    surface = ((PySDLSurface*)array->surface)->surface;
    if (!SDLColorFromObj (delcolor, surface->format, &dcolor) ||
        !SDLColorFromObj (replcolor, surface->format, &rcolor))
        return NULL;

    if (!_get_weights (weights, &wr, &wg, &wb))
        return NULL;

    surface = ((PySDLSurface*)array->surface)->surface;
    pixels = surface->pixels;
    absxstep = ABS (array->xstep);
    absystep = ABS (array->ystep);
    y = array->ystart;

    if (distance)
        SDL_GetRGB (dcolor, surface->format, &r1, &g1, &b1);

    Py_BEGIN_ALLOW_THREADS;
    switch (surface->format->BytesPerPixel)
    {
    case 1:
    {
        Uint8 *pixel;
        while (posy < array->ylen)
        {
            x = array->xstart;
            posx = 0;
            while (posx < array->xlen)
            {
                pixel = ((Uint8 *) pixels + y * surface->pitch + x);
                if (distance)
                {
                    GET_PALETTE_VALS (pixel, surface->format, r2, g2, b2, a2);
                    if (COLOR_DIFF_RGB (wr, wg, wb, r1, g1, b1, r2, g2, b2) <=
                        distance)
                        *pixel = (Uint8) rcolor;
                }
                else if (*pixel == dcolor)
                    *pixel = (Uint8) rcolor;
                x += array->xstep;
                posx += absxstep;
            }
            y += array->ystep;
            posy += absystep;
        }
        break;
    }
    case 2:
    {
        Uint16 *pixel;
        while (posy < array->ylen)
        {
            x = array->xstart;
            posx = 0;
            while (posx < array->xlen)
            {
                pixel = ((Uint16 *) (pixels + y * surface->pitch) + x);
                if (distance)
                {
                    GET_RGB_VALS ((Uint32) *pixel, surface->format,
                        r2, g2, b2, a2);
                    if (COLOR_DIFF_RGB (wr, wg, wb, r1, g1, b1, r2, g2, b2) <=
                        distance)
                        *pixel = (Uint16) rcolor;
                }
                else if (*pixel == dcolor)
                    *pixel = (Uint16) rcolor;
                x += array->xstep;
                posx += absxstep;
            }
            y += array->ystep;
            posy += absystep;
        }
        break;
    }
    case 3:
    {
        Uint8 *px;
        Uint32 pxcolor;
        while (posy < array->ylen)
        {
            x = array->xstart;
            posx = 0;
            while (posx < array->xlen)
            {
                px = ((Uint8 *) (pixels + y * surface->pitch) + x * 3);
                pxcolor = GET_PIXEL24 (px);
                if (distance)
                {
                    GET_RGB_VALS (pxcolor, surface->format, r2, g2, b2, a2);
                    if (COLOR_DIFF_RGB (wr, wg, wb, r1, g1, b1, r2, g2, b2) <=
                        distance)
                    {
                        SET_PIXEL24_RGB (px, surface->format,
                            ((Uint8) (rcolor >> 16)),
                            ((Uint8) (rcolor >> 8)),
                            ((Uint8) (rcolor)));
                    }
                }
                else if (pxcolor == dcolor)
                {
                    SET_PIXEL24_RGB (px,  surface->format,
                        ((Uint8) (rcolor >> 16)),
                        ((Uint8) (rcolor >> 8)),
                        ((Uint8) (rcolor)));
                }
                x += array->xstep;
                posx += absxstep;
            }
            y += array->ystep;
            posy += absystep;
        }
        break;
    }
    default:
    {
        Uint32 *pixel;
        while (posy < array->ylen)
        {
            x = array->xstart;
            posx = 0;
            while (posx < array->xlen)
            {
                pixel = ((Uint32 *) (pixels + y * surface->pitch) + x);
                if (distance)
                {
                    GET_RGB_VALS (*pixel, surface->format, r2, g2, b2, a2);
                    if (COLOR_DIFF_RGB (wr, wg, wb, r1, g1, b1, r2, g2, b2) <=
                        distance)
                        *pixel = rcolor;
                }
                else if (*pixel == dcolor)
                    *pixel = rcolor;
                x += array->xstep;
                posx += absxstep;
            }
            y += array->ystep;
            posy += absystep;
        }
        break;
    }
    }
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyObject*
_extract_color (PyPixelArray *array, PyObject *args, PyObject *kwds)
{
    PyObject *weights = NULL;
    PyObject *sf = NULL;
    PyObject *excolor = NULL;
    PyPixelArray *newarray = NULL;
    Uint32 color;
    Uint32 white;
    Uint32 black;
    SDL_Surface *surface;
    double distance = 0;
    double wr, wg, wb;
    Uint8 r1, g1, b1, r2, g2, b2, a2;

    Uint32 x = 0;
    Uint32 y = 0;
    Uint32 posx = 0;
    Uint32 posy = 0;
    Sint32 absxstep;
    Sint32 absystep;
    Uint8 *pixels;

    static char *keys[] = { "color", "distance", "weights", NULL };

    if (!PyArg_ParseTupleAndKeywords (args, kwds, "O|dO", keys, &excolor,
            &distance, &weights))
        return NULL;

    if (distance < 0 || distance > 1)
    {
        PyErr_SetString (PyExc_ValueError,
            "distance must be in the range from 0.0 to 1.0");
        return NULL;
    }
    if (!_get_weights (weights, &wr, &wg, &wb))
        return NULL;

    surface = ((PySDLSurface*)array->surface)->surface;
    if (!SDLColorFromObj (excolor, surface->format, &color))
        return NULL;

    /* Create the b/w mask surface. */
    sf = _make_surface (array);
    if (!sf)
        return NULL;
    newarray = (PyPixelArray *) PyPixelArray_New (sf);
    if (!newarray)
    {
        Py_DECREF (sf);
        return NULL;
    }
    surface = ((PySDLSurface*)newarray->surface)->surface;

    black = SDL_MapRGBA (surface->format, 0, 0, 0, 255);
    white = SDL_MapRGBA (surface->format, 255, 255, 255, 255);
    if (distance)
        SDL_GetRGB (color, surface->format, &r1, &g1, &b1);

    pixels = surface->pixels;
    absxstep = ABS (newarray->xstep);
    absystep = ABS (newarray->ystep);
    y = newarray->ystart;

    Py_BEGIN_ALLOW_THREADS;
    switch (surface->format->BytesPerPixel)
    {
    case 1:
    {
        Uint8 *pixel;
        while (posy < newarray->ylen)
        {
            x = newarray->xstart;
            posx = 0;
            while (posx < newarray->xlen)
            {
                pixel = ((Uint8 *) pixels + y * surface->pitch + x);
                if (distance)
                {
                    GET_PALETTE_VALS (pixel, surface->format, r2, g2, b2, a2);
                    if (COLOR_DIFF_RGB (wr, wg, wb, r1, g1, b1, r2, g2, b2) <=
                        distance)
                        *pixel = (Uint8) white;
                    else
                        *pixel = (Uint8) black;
                }
                else
                    *pixel = (*pixel == color) ? (Uint8) white : (Uint8) black;
                x += newarray->xstep;
                posx += absxstep;
            }
            y += newarray->ystep;
            posy += absystep;
        }
        break;
    }
    case 2:
    {
        Uint16 *pixel;
        while (posy < newarray->ylen)
        {
            x = newarray->xstart;
            posx = 0;
            while (posx < newarray->xlen)
            {
                pixel = ((Uint16 *) (pixels + y * surface->pitch) + x);
                if (distance)
                {
                    GET_RGB_VALS ((Uint32)*pixel, surface->format,
                        r2, g2, b2, a2);
                    if (COLOR_DIFF_RGB (wr, wg, wb, r1, g1, b1, r2, g2, b2) <=
                        distance)
                        *pixel = (Uint16) white;
                    else
                        *pixel = (Uint16) black;
                }
                else
                    *pixel = (*pixel == color) ? (Uint16) white :
                        (Uint16) black;
                x += newarray->xstep;
                posx += absxstep;
            }
            y += newarray->ystep;
            posy += absystep;
        }
        break;
    }
    case 3:
    {
        Uint8 *px;
        Uint32 pxcolor;
        while (posy < newarray->ylen)
        {
            x = newarray->xstart;
            posx = 0;
            while (posx < newarray->xlen)
            {
                px = ((Uint8 *) (pixels + y * surface->pitch) + x * 3);
                pxcolor = GET_PIXEL24 (px);
                if (distance)
                {
                    GET_RGB_VALS (pxcolor, surface->format, r2, g2, b2, a2);
                    if (COLOR_DIFF_RGB (wr, wg, wb, r1, g1, b1, r2, g2, b2) <=
                        distance)
                    {
                        SET_PIXEL24_RGB (px,  surface->format,
                            ((Uint8) (white >> 16)),
                            ((Uint8) (white >> 8)),
                            ((Uint8) (white)));
                    }
                    else
                    {
                        SET_PIXEL24_RGB (px, surface->format,
                            ((Uint8) (black >> 16)),
                            ((Uint8) (black >> 8)),
                            ((Uint8) (black)));
                    }
                }
                else if (pxcolor == color)
                {
                    SET_PIXEL24_RGB (px,  surface->format,
                        ((Uint8) (white >> 16)),
                        ((Uint8) (white >> 8)),
                        ((Uint8) (white)));
                }
                else
                {
                    SET_PIXEL24_RGB (px, surface->format,
                        ((Uint8) (black >> 16)),
                        ((Uint8) (black >> 8)),
                        ((Uint8) (black)));
                }
                x += newarray->xstep;
                posx += absxstep;
            }
            y += newarray->ystep;
            posy += absystep;
        }
        break;
    }
    default:
    {
        Uint32 *pixel;
        while (posy < newarray->ylen)
        {
            x = newarray->xstart;
            posx = 0;
            while (posx < newarray->xlen)
            {
                pixel = ((Uint32 *) (pixels + y * surface->pitch) + x);
                if (distance)
                {
                    GET_RGB_VALS (*pixel, surface->format, r2, g2, b2, a2);
                    if (COLOR_DIFF_RGB (wr, wg, wb, r1, g1, b1, r2, g2, b2) <=
                        distance)
                        *pixel = white;
                    else
                        *pixel = black;
                }
                else
                    *pixel = (*pixel == color) ? white : black;
                x += newarray->xstep;
                posx += absxstep;
            }
            y += newarray->ystep;
            posy += absystep;
        }
        break;
    }
    }
    Py_END_ALLOW_THREADS;
    return (PyObject *) newarray;
}

static PyObject*
_compare (PyPixelArray *array, PyObject *args, PyObject *kwds)
{
    PyPixelArray *array2 = NULL;
    PyPixelArray *newarray = NULL;
    PyObject *weights = NULL;
    PyObject *sf = NULL;
    SDL_Surface *surface1 = NULL;
    SDL_Surface *surface2 = NULL;
    Uint32 black;
    Uint32 white;
    double distance = 0;
    Uint8 r1, g1, b1, a1, r2, g2, b2, a2;
    double wr, wg, wb;

    Uint32 x = 0;
    Uint32 y = 0;
    Uint32 vx = 0;
    Uint32 vy = 0;
    Uint32 posx = 0;
    Uint32 posy = 0;
    Sint32 absxstep;
    Sint32 absystep;
    Uint8 *pixels1;
    Uint8 *pixels2;

    static char *keys[] = { "array", "distance", "weights", NULL };

    if (!PyArg_ParseTupleAndKeywords (args, kwds, "O|dO", keys, &newarray,
            &distance, &weights))
        return NULL;

    if (!PyPixelArray_Check (newarray))
    {
        PyErr_SetString (PyExc_TypeError, "invalid array type");
        return NULL;
    }

    array2 = (PyPixelArray *) newarray;
    newarray = NULL;
    if (distance < 0 || distance > 1)
    {
        PyErr_SetString (PyExc_ValueError,
            "distance must be in the range from 0.0 to 1.0");
        return NULL;
    }
    if (!_get_weights (weights, &wr, &wg, &wb))
        return NULL;

    if (array->ylen / ABS (array->ystep) != array2->ylen / ABS (array2->ystep)
        || array->xlen / ABS (array->xstep) != array2->xlen / ABS (array2->xstep))
    {
        /* Bounds do not match. */
        PyErr_SetString (PyExc_ValueError, "array sizes do not match");
        return NULL;
    }

    surface1 = ((PySDLSurface*) array->surface)->surface;
    surface2 = ((PySDLSurface*) array2->surface)->surface;
    if (surface2->format->BytesPerPixel != surface1->format->BytesPerPixel)
    {
        PyErr_SetString (PyExc_ValueError, "bit depths do not match");
        return NULL;
    }

    /* Create the b/w mask surface. */
    sf = _make_surface (array);
    if (!sf)
        return NULL;
    newarray = (PyPixelArray *) PyPixelArray_New (sf);
    if (!newarray)
    {
        Py_DECREF (sf);
        return NULL;
    }
    surface1 = ((PySDLSurface*)newarray->surface)->surface;
    
    black = SDL_MapRGBA (surface1->format, 0, 0, 0, 255);
    white = SDL_MapRGBA (surface1->format, 255, 255, 255, 255);

    pixels1 = surface1->pixels;
    pixels2 = surface2->pixels;
    absxstep = ABS (array2->xstep);
    absystep = ABS (array2->ystep);
    y = array2->ystart;

    Py_BEGIN_ALLOW_THREADS;
    switch (surface1->format->BytesPerPixel)
    {
    case 1:
    {
        Uint8 *pixel1, *pixel2;
        while (posy < newarray->ylen)
        {
            vx = array2->xstart;
            x = newarray->xstart;
            posx = 0;
            while (posx < newarray->xlen)
            {
                pixel1 = ((Uint8 *) pixels1 + y * surface1->pitch + x);
                pixel2 = ((Uint8 *) pixels2 + vy * surface2->pitch + vx);
                if (distance)
                {
                    GET_PALETTE_VALS (pixel1, surface1->format, r1, g1, b1, a1);
                    GET_PALETTE_VALS (pixel2, surface2->format, r2, g2, b2, a2);
                    if (COLOR_DIFF_RGB (wr, wg, wb, r1, g1, b1, r2, g2, b2) >
                        distance)
                        *pixel1 = (Uint8) white;
                    else
                        *pixel1 = (Uint8) black;
                }
                else
                    *pixel1 = (*pixel1 == *pixel2) ? (Uint8) white :
                        (Uint8) black;
                vx += array2->xstep;
                x += newarray->xstep;
                posx += absxstep;
            }
            vy += array2->ystep;
            y += newarray->ystep;
            posy += absystep;
        }
        break;
    }
    case 2:
    {
        Uint16 *pixel1, *pixel2;
        while (posy < newarray->ylen)
        {
            vx = array2->xstart;
            x = array->xstart;
            posx = 0;
            while (posx < newarray->xlen)
            {
                pixel1 = ((Uint16 *) (pixels1 + y * surface1->pitch) + x);
                pixel2 = ((Uint16 *) (pixels2 + vy * surface2->pitch) + vx);
                if (distance)
                {
                    GET_RGB_VALS ((Uint32) *pixel1, surface1->format,
                        r1, g1, b1, a1);
                    GET_RGB_VALS ((Uint32) *pixel2, surface2->format,
                        r2, g2, b2, a2);
                    if (COLOR_DIFF_RGB (wr, wg, wb, r1, g1, b1, r2, g2, b2) >
                        distance)
                        *pixel1 = (Uint16) white;
                    else
                        *pixel1 = (Uint16) black;
                }
                else
                    *pixel1 = (*pixel1 == *pixel2) ? (Uint16) white :
                        (Uint16) black;
                vx += array2->xstep;
                x += newarray->xstep;
                posx += absxstep;
            }
            vy += array2->ystep;
            y += newarray->ystep;
            posy += absystep;
        }
        break;
    }
    case 3:
    {
        Uint8 *px1, *px2;
        Uint32 pxcolor1, pxcolor2;
        while (posy < newarray->ylen)
        {
            vx = array2->xstart;
            x = newarray->xstart;
            posx = 0;
            while (posx < newarray->xlen)
            {
                px1 = ((Uint8 *) (pixels1 + y * surface1->pitch) + x * 3);
                px2 = ((Uint8 *) (pixels2 + vy * surface2->pitch) + vx * 3);
                pxcolor1 = GET_PIXEL24(px1);
                pxcolor2 = GET_PIXEL24(px2);
                if (distance)
                {
                    GET_RGB_VALS (pxcolor1, surface1->format, r1, g1, b1, a1);
                    GET_RGB_VALS (pxcolor2, surface2->format, r2, g2, b2, a2);
                    if (COLOR_DIFF_RGB (wr, wg, wb, r1, g1, b1, r2, g2, b2) >
                        distance)
                    {
                        SET_PIXEL24_RGB(px1, surface1->format,
                            ((Uint8) (white >> 16)),
                            ((Uint8) (white >> 8)),
                            ((Uint8) (white)));
                    }
                    else
                    {
                        SET_PIXEL24_RGB(px1, surface1->format,
                            ((Uint8) (black >> 16)),
                            ((Uint8) (black >> 8)),
                            ((Uint8) (black)));
                    }
                }
                else if (pxcolor1 != pxcolor2)
                {
                    SET_PIXEL24_RGB(px1, surface1->format,
                        ((Uint8) (white >> 16)),
                        ((Uint8) (white >> 8)),
                        ((Uint8) (white)));
                }
                else
                {
                    SET_PIXEL24_RGB(px1, surface1->format,
                        ((Uint8) (black >> 16)),
                        ((Uint8) (black >> 8)),
                        ((Uint8) (black)));
                }
                vx += array2->xstep;
                x += newarray->xstep;
                posx += absxstep;
            }
            vy += array2->ystep;
            y += newarray->ystep;
            posy += absystep;
        }
        break;
    }
    default:
    {
        Uint32 *pixel1, *pixel2;
        while (posy < newarray->ylen)
        {
            vx = array2->xstart;
            x = newarray->xstart;
            posx = 0;
            while (posx < newarray->xlen)
            {
                pixel1 = ((Uint32 *) (pixels1 + y * surface1->pitch) + x);
                pixel2 = ((Uint32 *) (pixels2 + vy * surface2->pitch) + vx);
                if (distance)
                {
                    GET_RGB_VALS (*pixel1, surface1->format, r1, g1, b1, a1);
                    GET_RGB_VALS (*pixel2, surface2->format, r2, g2, b2, a2);
                    if (COLOR_DIFF_RGB (wr, wg, wb, r1, g1, b1, r2, g2, b2) >
                        distance)
                        *pixel1 = white;
                    else
                        *pixel1 = black;
                }
                else
                    *pixel1 = (*pixel1 == *pixel2) ? white : black;
                vx += array2->xstep;
                x += newarray->xstep;
                posx += absxstep;
            }
            vy += array2->ystep;
            y += newarray->ystep;
            posy += absystep;
        }
        break;
    }
    }
    Py_END_ALLOW_THREADS;
    return (PyObject *) newarray;
}
