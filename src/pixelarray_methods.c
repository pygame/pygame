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

/**
 * Tries to retrieve a valid color for a Surface.
 */
static int
_get_color_from_object (PyObject *val, SDL_PixelFormat *format, Uint32 *color)
{
    Uint8 rgba[4];

    if (!val)
        return 0;

    if (PyInt_Check (val))
    {
        long intval = PyInt_AsLong (val);
        if ((intval < INT_MIN) || (intval > INT_MAX))
        {
            if (!PyErr_Occurred ())
                PyErr_SetString (PyExc_ValueError, "invalid color argument");
            return 0;
        }
        *color = (Uint32) intval;
        return 1;
    }
    else if (PyLong_Check (val))
    {
        long long longval = PyLong_AsLong (val);
        if ((longval < INT_MIN) || (longval > INT_MAX))
        {
            PyErr_SetString(PyExc_ValueError, "invalid color argument");
            return 0;
        }
        *color = (Uint32) longval;
        return 1;
    }
    else if (RGBAFromObj (val, rgba))
    {
        *color = (Uint32) SDL_MapRGBA
            (format, rgba[0], rgba[1], rgba[2], rgba[3]);
        return 1;
    }
    else
        PyErr_SetString (PyExc_ValueError, "invalid color argument");
    return 0;
}

/**
 * Retrieves a single pixel located at index from the surface pixel
 * array.
 */
static PyObject*
_get_single_pixel (Uint8 *pixels, int bpp, Uint32 _index, Uint32 row)
{
    Uint32 pixel;

    switch (bpp)
    {
    case 1:
        pixel = (Uint32)*((Uint8 *) pixels + row + _index);
        break;
    case 2:
        pixel = (Uint32)*((Uint16 *) (pixels + row) + _index);
        break;
    case 3:
    {
        Uint8 *px = ((Uint8 *) (pixels + row) + _index * 3);
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
        pixel = (px[0]) + (px[1] << 8) + (px[2] << 16);
#else
        pixel = (px[2]) + (px[1] << 8) + (px[0] << 16);
#endif
        break;
    }
    default: /* 4 bpp */
        pixel = *((Uint32 *) (pixels + row) + _index);
        break;
    }
    
    return PyInt_FromLong ((long)pixel);
}

/**
 * Sets a single pixel located at index from the surface pixel array.
 */
static void
_set_single_pixel (Uint8 *pixels, int bpp, Uint32 _index, Uint32 row,
    SDL_PixelFormat *format, Uint32 color)
{
    switch (bpp)
    {
    case 1:
        *((Uint8 *) pixels + row + _index) = (Uint8) color;
        break;
    case 2:
        *((Uint16 *) (pixels + row) + _index) = (Uint16) color;
        break;
    case 3:
#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
        *((Uint8 *) (pixels + row) + _index * 3 + (format->Rshift >> 3)) =
            (Uint8) (color >> 16);
        *((Uint8 *) (pixels + row) + _index * 3 + (format->Gshift >> 3)) =
            (Uint8) (color >> 8);
        *((Uint8 *) (pixels + row) + _index * 3 + (format->Bshift >> 3)) =
            (Uint8) color;
#else
        *((Uint8 *) (pixels + row) + _index * 3 + 2 - (format->Rshift >> 3)) =
            (Uint8) (color >> 16);
        *((Uint8 *) (pixels + row) + _index * 3 + 2 - (format->Gshift >> 3)) =
            (Uint8) (color >> 8);
        *((Uint8 *) (pixels + row) + _index * 3 + 2 - (format->Bshift >> 3)) =
            (Uint8) color;
#endif
        break;
    default: /* 4 bpp */
        *((Uint32 *) (pixels + row) + _index) = color;
        break;
    }
}

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

    surface = PySurface_AsSurface (array->surface);
    bpp = surface->format->BytesPerPixel;

    /* Create the second surface. */
    tmpsf = SDL_CreateRGBSurface (surface->flags,
        (int) (array->xlen / ABS (array->xstep)),
        (int) (array->ylen / ABS (array->ystep)), bpp, surface->format->Rmask,
        surface->format->Gmask, surface->format->Bmask, surface->format->Amask);
    if (!tmpsf)
        return RAISE (PyExc_SDLError, SDL_GetError ());

    /* Guarantee an identical format. */
    newsurf = SDL_ConvertSurface (tmpsf, surface->format, surface->flags);
    if (!newsurf)
    {
        SDL_FreeSurface (tmpsf);
        return RAISE (PyExc_SDLError, SDL_GetError ());
    }
    SDL_FreeSurface (tmpsf);

    /* Acquire a temporary lock. */
    if (SDL_MUSTLOCK (newsurf) == 0)
        SDL_LockSurface (newsurf);

    pixels = (Uint8 *) newsurf->pixels;
    origpixels = (Uint8 *) surface->pixels;

    absxstep = ABS (array->xstep);
    absystep = ABS (array->ystep);
    y = array->ystart;

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
                px = (Uint8 *) (pixels + vy * newsurf->pitch) + vx * 3;
                vpx = (Uint8 *) ((Uint8*) origpixels + y * array->padding) +
                    x * 3;

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
    if (SDL_MUSTLOCK (newsurf) == 0)
        SDL_UnlockSurface (newsurf);
    newsf = PySurface_New (newsurf);
    if (!newsf)
        return NULL;
    return newsf;
}

static PyObject*
_replace_color (PyPixelArray *array, PyObject *args)
{
    PyObject *delcolor = NULL;
    PyObject *replcolor = NULL;
    Uint32 dcolor;
    Uint32 rcolor;
    SDL_Surface *surface;

    Uint32 x = 0;
    Uint32 y = 0;
    Uint32 posx = 0;
    Uint32 posy = 0;
    Sint32 absxstep;
    Sint32 absystep;
    Uint8 *pixels;

    if (!PyArg_ParseTuple (args, "OO", &delcolor, &replcolor))
        return NULL;

    surface = PySurface_AsSurface (array->surface);
    if (!_get_color_from_object (delcolor, surface->format, &dcolor) ||
        !_get_color_from_object (replcolor, surface->format, &rcolor))
        return NULL;

    surface = PySurface_AsSurface (array->surface);
    pixels = surface->pixels;
    absxstep = ABS (array->xstep);
    absystep = ABS (array->ystep);
    y = array->ystart;

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
                if (*pixel == dcolor)
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
                if (*pixel == dcolor)
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
        SDL_PixelFormat *format = surface->format;
        while (posy < array->ylen)
        {
            x = array->xstart;
            posx = 0;
            while (posx < array->xlen)
            {
                px = (Uint8 *) (pixels + y * surface->pitch) + x * 3;
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                pxcolor = (px[0]) + (px[1] << 8) + (px[2] << 16);
                if (pxcolor == dcolor)
                {
                    *(px + (format->Rshift >> 3)) = (Uint8) (rcolor >> 16);
                    *(px + (format->Gshift >> 3)) = (Uint8) (rcolor >> 8);
                    *(px + (format->Bshift >> 3)) = (Uint8) rcolor;
                }
#else
                pxcolor = (px[2]) + (px[1] << 8) + (px[0] << 16);
                if (pxcolor == dcolor)
                {
                    *(px + 2 - (format->Rshift >> 3)) = (Uint8) (rcolor >> 16);
                    *(px + 2 - (format->Gshift >> 3)) = (Uint8) (rcolor >> 8);
                    *(px + 2 - (format->Bshift >> 3)) = (Uint8) rcolor;
                }
#endif
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
                if (*pixel == dcolor)
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
    Py_RETURN_NONE;
}
