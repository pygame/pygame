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

/* Modified RGBAFromColorObj that only accepts pygame.Color or tuple objects.
 */
static int
_RGBAFromColorObj (PyObject *obj, Uint8 rgba[4]) {
    if (PyObject_IsInstance (obj, &PyColor_Type) || PyTuple_Check (obj)) {
        return RGBAFromColorObj (obj, rgba);
    }
    PyErr_SetString(PyExc_ValueError, "invalid color argument");
    return 0;
}

/**
 * Tries to retrieve a valid color for a Surface.
 */
static int
_get_color_from_object(PyObject *val, SDL_PixelFormat *format, Uint32 *color)
{
    Uint8 rgba[4];

    if (!val) {
        return 0;
    }

    if (PyInt_Check(val)) {
        long intval = PyInt_AsLong (val);
        if (intval == -1 && PyErr_Occurred()) {
            PyErr_SetString (PyExc_ValueError, "invalid color argument");
            return 0;
        }
        *color = (Uint32) intval;
        return 1;
    }
    else if (PyLong_Check(val)) {
        unsigned long longval = PyLong_AsUnsignedLong (val);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "invalid color argument");
            return 0;
        }
        *color = (Uint32) longval;
        return 1;
    }
    else if (_RGBAFromColorObj(val, rgba)) {
      *color = (Uint32) SDL_MapRGBA(format, rgba[0], rgba[1], rgba[2], rgba[3]);
        return 1;
    }
    else {
        PyErr_SetString(PyExc_ValueError, "invalid color argument");
    }
    return 0;
}


/**
 * Retrieves a single pixel located at index from the surface pixel
 * array.
 */
static PyObject *
_get_single_pixel(PyPixelArray *array, Uint32 x, Uint32 y)
{
    Uint8 *pixel_p = array->pixels + x * array->stride0 + y * array->stride1;
    SDL_Surface *surf = PySurface_AsSurface(array->surface);
    int bpp;
    Uint32 pixel;

    bpp = surf->format->BytesPerPixel;

    switch (bpp) {

    case 1:
        pixel = (Uint32)*pixel_p;
        break;
    case 2:
        pixel = (Uint32)*((Uint16 *)pixel_p);
        break;
    case 3:
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
        pixel = (pixel_p[0]) + (pixel_p[1] << 8) + (pixel_p[2] << 16);
#else
        pixel = (pixel_p[2]) + (pixel_p[1] << 8) + (pixel_p[0] << 16);
#endif
        break;
    case 4:
        pixel = *((Uint32 *)pixel_p);
        break;
    default: /* Should not get here. */
        PyErr_Format(PyExc_SystemError,
                     "Pygame internal error in _get_single_pixel: "
                     "unexpected pixel size %i", bpp);
        return 0;
    }
    
    return PyInt_FromLong((long)pixel);
}

/**
 * Creates a new surface using the currently applied dimensions, step
 * size, etc.
 */
static PyObject *
_make_surface(PyPixelArray *array)
{
    SDL_Surface *surf = PySurface_AsSurface(array->surface);
    int bpp;
    Py_ssize_t dim0 = array->dim0;
    Py_ssize_t dim1 = array->dim1 ? array->dim1 : 1;
    Py_ssize_t stride0 = array->stride0;
    Py_ssize_t stride1 = array->stride1;
    PyObject *new_surface;
    SDL_Surface *temp_surf;
    SDL_Surface *new_surf;
    Py_ssize_t new_stride0;
    Py_ssize_t new_stride1;
    Uint8 *pixels = array->pixels;
    Uint8 *new_pixels;
    Py_ssize_t x;
    Py_ssize_t y;
    Uint8 *pixelrow;
    Uint8 *pixel_p;
    Uint8 *new_pixelrow;
    Uint8 *new_pixel_p;

    bpp = surf->format->BytesPerPixel;

    /* Create the second surface. */

    temp_surf = SDL_CreateRGBSurface(surf->flags,
				     (int)dim0,
				     (int)dim1,
				     bpp, 
				     surf->format->Rmask,
				     surf->format->Gmask,
				     surf->format->Bmask,
				     surf->format->Amask);
    if (!temp_surf) {
        return RAISE(PyExc_SDLError, SDL_GetError());
    }

    /* Guarantee an identical format. */
    new_surf = SDL_ConvertSurface(temp_surf, surf->format, surf->flags);
    SDL_FreeSurface(temp_surf);
    if (!new_surf) {
        return RAISE(PyExc_SDLError, SDL_GetError());
    }

    new_surface = PySurface_New(new_surf);
    if (!new_surface) {
        SDL_FreeSurface(new_surf);
        return 0;
    }

    /* Acquire a temporary lock. */
    if (SDL_MUSTLOCK(new_surf) == 0) {
        SDL_LockSurface(new_surf);
    }

    new_pixels = (Uint8 *)new_surf->pixels;
    new_stride0 = new_surf->format->BytesPerPixel;
    new_stride1 = new_surf->pitch;
    pixelrow = pixels;
    new_pixelrow = new_pixels;

    Py_BEGIN_ALLOW_THREADS;
    switch (bpp) {

    case 1:
        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            new_pixel_p = new_pixelrow;
            for (x = 0; x < dim0; ++x) {
                *new_pixel_p = *pixel_p;
                pixel_p += stride0;
                new_pixel_p += new_stride0;
            }
            pixelrow += stride1;
            new_pixelrow += new_stride1;
        }
        break;
    case 2:
        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            new_pixel_p = new_pixelrow;
            for (x = 0; x < dim0; ++x) {
                *((Uint16 *)new_pixel_p) = *((Uint16 *)pixel_p);
                pixel_p += stride0;
                new_pixel_p += new_stride0;
            }
            pixelrow += stride1;
            new_pixelrow += new_stride1;
        }
        break;
    case 3:
        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            new_pixel_p = new_pixelrow;
            for (x = 0; x < dim0; ++x) {
                new_pixel_p[0] = pixel_p[0];
                new_pixel_p[1] = pixel_p[1];
                new_pixel_p[2] = pixel_p[2];
                pixel_p += stride0;
                new_pixel_p += new_stride0;
            }
            pixelrow += stride1;
            new_pixelrow += new_stride1;
        }
        break;
    default: /* case: 4 */
        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            new_pixel_p = new_pixelrow;
            for (x = 0; x < dim0; ++x) {
                *((Uint32 *)new_pixel_p) = *((Uint32 *)pixel_p);
                pixel_p += stride0;
                new_pixel_p += new_stride0;
            }
            pixelrow += stride1;
            new_pixelrow += new_stride1;
        }
    }
    Py_END_ALLOW_THREADS;

    if (SDL_MUSTLOCK(new_surf) == 0) {
        SDL_UnlockSurface(new_surf);
    }
    return new_surface;
}

static int
_get_weights(PyObject *weights, float *wr, float *wg, float *wb)
{
    int success = 1;
    float rgb[3] = { 0 };
    
    if (!weights) {
        *wr = WR_NTSC;
        *wg = WG_NTSC;
        *wb = WB_NTSC;
        return 1;
    }
    
    if (!PySequence_Check(weights)) {
        PyErr_SetString(PyExc_TypeError, "weights must be a sequence");
        success = 0;
    } 
    else if (PySequence_Size(weights) < 3) {
        PyErr_SetString(PyExc_TypeError,
			"weights must contain at least 3 values");
        success = 0;
    }
    else {
        PyObject *item;
        int i;
        
        for (i = 0; i < 3; ++i) {
            item = PySequence_GetItem(weights, i);
            if (PyNumber_Check(item)) {
                PyObject *num;

                if ((num = PyNumber_Float(item))) {
                    rgb[i] = (float)PyFloat_AsDouble(num);
                    Py_DECREF(num);
                }
                else if (PyErr_Clear(), (num = PyNumber_Int(item))) {
                    rgb[i] = (float)PyInt_AsLong (num);
                    success = rgb[i] != -1 || !PyErr_Occurred();
                    Py_DECREF(num);
                }
                else if (PyErr_Clear(), (num = PyNumber_Long(item))) {
                    rgb[i] = (float)PyLong_AsLong(num);
                    success = (!PyErr_Occurred() ||
                               !PyErr_ExceptionMatches(PyExc_OverflowError));
                    Py_DECREF(num);
                }
                else {
                    PyErr_Clear();
                    PyErr_Format(PyExc_TypeError,
                                 "Unrecognized number type %s",
                                 Py_TYPE(item)->tp_name);
                    success = 0;
                }
            }
            else {
                PyErr_SetString(PyExc_TypeError, "invalid weights");
                success = 0;
            }
            Py_XDECREF(item);
            if (!success) {
                break;
            }
        }
    }
    
    if (success)
    {
        float sum = 0;
        
        *wr = rgb[0];
        *wg = rgb[1];
        *wb = rgb[2];
        if ((*wr < 0 || *wg < 0 || *wb < 0) ||
            (*wr == 0 && *wg == 0 && *wb == 0))
        {
            PyErr_SetString(PyExc_ValueError,
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

static PyObject *
_replace_color(PyPixelArray *array, PyObject *args, PyObject *kwds)
{
    PyObject *weights = 0;
    PyObject *delcolor = 0;
    PyObject *replcolor = 0;
    SDL_Surface *surf = PySurface_AsSurface(array->surface);
    SDL_PixelFormat *format;
    Py_ssize_t dim0 = array->dim0;
    Py_ssize_t dim1 = array->dim1;
    Py_ssize_t stride0 = array->stride0;
    Py_ssize_t stride1 = array->stride1;
    Uint8 *pixels = array->pixels;
    int bpp;
    Uint32 dcolor;
    Uint32 rcolor;
    Uint8 r1, g1, b1, r2, g2, b2, a2;
    Uint8 *pixelrow;
    Uint8 *pixel_p;
    float distance = 0;
    float wr, wg, wb;
    Py_ssize_t x;
    Py_ssize_t y;
    static char *keys[] = { "color", "repcolor", "distance", "weights", NULL };
    
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|fO", keys, &delcolor,
                                     &replcolor, &distance, &weights)) {
        return 0;
    }

    if (distance < 0 || distance > 1) {
        return RAISE(PyExc_ValueError,
                     "distance must be in the range from 0.0 to 1.0");
    }

    format = surf->format;
    bpp = surf->format->BytesPerPixel;

    if (!_get_color_from_object(delcolor, format, &dcolor) ||
        !_get_color_from_object(replcolor, format, &rcolor)   ) {
        return 0;
    }

    if (!_get_weights(weights, &wr, &wg, &wb)) {
        return 0;
    }

    if (distance != 0.0) {
        SDL_GetRGB(dcolor, format, &r1, &g1, &b1);
    }

    if (!dim1) {
        dim1 = 1;
    }
    pixelrow = pixels;

    Py_BEGIN_ALLOW_THREADS;
    switch (bpp) {

    case 1:
    {
        Uint8 *px_p;

        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            for (x = 0; x < dim0; ++x) {
                px_p = pixel_p;
                if (distance != 0.0) {
                    GET_PIXELVALS_1(r2, g2, b2, a2, px_p, format);
                    if (COLOR_DIFF_RGB(wr, wg, wb, r1, g1, b1, r2, g2, b2) <=
                        distance) {
                        *px_p = (Uint8)rcolor;
                    }
                }
                else if (*px_p == dcolor) {
                    *px_p = (Uint8)rcolor;
                }
                pixel_p += stride0;
            }
            pixelrow += stride1;
        }
    }
        break;
    case 2:
    {
        Uint16 *px_p;
	int ppa = (surf->flags & SDL_SRCALPHA && format->Amask);

        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            for (x = 0; x < dim0; ++x) {
                px_p = (Uint16 *)pixel_p;
                if (distance != 0.0) {
                    GET_PIXELVALS(r2, g2, b2, a2, (Uint32)*px_p, format, ppa);
                    if (COLOR_DIFF_RGB(wr, wg, wb, r1, g1, b1, r2, g2, b2) <=
                        distance) {
                        *px_p = (Uint16)rcolor;
                    }
                }
                else if (*px_p == dcolor) {
                    *px_p = (Uint16)rcolor;
                }
                pixel_p += stride0;
            }
            pixelrow += stride1;
        }
    }
        break;
    case 3:
    {
#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
        Uint32 Roffset = format->Rshift >> 3;
        Uint32 Goffset = format->Gshift >> 3;
        Uint32 Boffset = format->Bshift >> 3;
#else
        Uint32 Roffset = 2 - (format->Rshift >> 3);
        Uint32 Goffset = 2 - (format->Gshift >> 3);
        Uint32 Boffset = 2 - (format->Bshift >> 3);
#endif
        Uint32 pxcolor;
	int ppa = (surf->flags & SDL_SRCALPHA && format->Amask);

        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            for (x = 0; x < dim0; ++x) {
                pxcolor = (((Uint32)pixel_p[Roffset] << 16) +
                           ((Uint32)pixel_p[Goffset] << 8)  +
                           ((Uint32)pixel_p[Boffset])         );
                if (distance != 0.0) {
                    GET_PIXELVALS(r2, g2, b2, a2, pxcolor, format, ppa);
                    if (COLOR_DIFF_RGB(wr, wg, wb, r1, g1, b1, r2, g2, b2) <=
                        distance) {
                        pixel_p[Roffset] = (Uint8)(rcolor >> 16);
                        pixel_p[Goffset] = (Uint8)(rcolor >> 8);
                        pixel_p[Boffset] = (Uint8)rcolor;
                    }
                }
                else if (pxcolor == dcolor) {
                    pixel_p[Roffset] = (Uint8)(rcolor >> 16);
                    pixel_p[Goffset] = (Uint8)(rcolor >> 8);
                    pixel_p[Boffset] = (Uint8)rcolor;
                }
                pixel_p += stride0;
            }
            pixelrow += stride1;
        }
    }
        break;
    default: /* case 4: */
    {
        Uint32 *px_p;
	int ppa = (surf->flags & SDL_SRCALPHA && surf->format->Amask);

        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            for (x = 0; x < dim0; ++x) {
                px_p = (Uint32 *)pixel_p;
                if (distance != 0.0) {
                    GET_PIXELVALS(r2, g2, b2, a2, *px_p, format, ppa);
                    if (COLOR_DIFF_RGB(wr, wg, wb, r1, g1, b1, r2, g2, b2) <=
                        distance) {
                        *px_p = rcolor;
                    }
                }
                else if (*px_p == dcolor) {
                    *px_p = rcolor;
                }
                pixel_p += stride0;
            }
            pixelrow += stride1;
        }
    }
        break;
    }
    Py_END_ALLOW_THREADS;

    Py_RETURN_NONE;
}

static PyObject*
_extract_color (PyPixelArray *array, PyObject *args, PyObject *kwds)
{
    PyObject *weights = 0;
    PyObject *excolor = 0;
    int bpp;
    Uint32 black;
    Uint32 white;
    Uint32 color;
    Uint8 r1, g1, b1, r2, g2, b2, a2;
    Uint8 *pixelrow;
    Uint8 *pixel_p;
    float distance = 0;
    float wr, wg, wb;
    Py_ssize_t x;
    Py_ssize_t y;
    PyObject *surface;
    SDL_Surface *surf;
    SDL_PixelFormat *format;
    PyPixelArray *new_array;
    Py_ssize_t dim0;
    Py_ssize_t dim1;
    Py_ssize_t stride0;
    Py_ssize_t stride1;
    Uint8 *pixels;
    static char *keys[] = { "color", "distance", "weights", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|fO", keys, &excolor,
                                     &distance, &weights)) {
        return 0;
    }

    if (distance < 0 || distance > 1) {
        return RAISE(PyExc_ValueError,
                     "distance must be in the range from 0.0 to 1.0");
    }

    if (!_get_weights(weights, &wr, &wg, &wb)) {
        return 0;
    }

    /* Create the b/w mask surface. */
    surface = _make_surface(array);
    if (!surface) {
        return 0;
    }

    new_array = (PyPixelArray *)PyPixelArray_New(surface);
    if (!new_array) {
        Py_DECREF(surface);
        return 0;
    }

    surf = PySurface_AsSurface(surface);
    format = surf->format;
    bpp = surf->format->BytesPerPixel;
    dim0 = new_array->dim0;
    dim1 = new_array->dim1;
    stride0 = new_array->stride0;
    stride1 = new_array->stride1;
    pixels = new_array->pixels;

    black = SDL_MapRGBA(format, 0, 0, 0, 255);
    white = SDL_MapRGBA(format, 255, 255, 255, 255);

    if (!_get_color_from_object(excolor, format, &color)) {
        Py_DECREF(new_array);
        return 0;
    }

    if (distance != 0.0) {
        SDL_GetRGB(color, format, &r1, &g1, &b1);
    }

    if (!dim1) {
        dim1 = 1;
    }
    pixelrow = pixels;

    Py_BEGIN_ALLOW_THREADS;
    switch (bpp) {

    case 1:
    {
        Uint8 *px_p;

        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            for (x = 0; x < dim0; ++x) {
                px_p = pixel_p;
                if (distance != 0.0) {
                    GET_PIXELVALS_1(r2, g2, b2, a2, px_p, format);
                    if (COLOR_DIFF_RGB(wr, wg, wb, r1, g1, b1, r2, g2, b2) <=
                        distance) {
                        *px_p = (Uint8)white;
                    }
                    else {
                        *px_p = (Uint8)black;
                    }
                }
                else {
                    *px_p = (Uint8)(*px_p == color ? white : black);
                }
                pixel_p += stride0;
            }
            pixelrow += stride1;
        }
    }
        break;
    case 2:
    {
        Uint16 *px_p;
	int ppa = (surf->flags & SDL_SRCALPHA && format->Amask);

        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            for (x = 0; x < dim0; ++x) {
                px_p = (Uint16 *)pixel_p;
                if (distance != 0.0) {
                    GET_PIXELVALS(r2, g2, b2, a2, (Uint32)*px_p, format, ppa);
                    if (COLOR_DIFF_RGB(wr, wg, wb, r1, g1, b1, r2, g2, b2) <=
                        distance) {
                        *px_p = (Uint16)white;
                    }
                    else {
                        *px_p = (Uint16)black;
                    }
                }
                else {
                    *px_p = (Uint16)(*px_p == color ? white : black);
                }
                pixel_p += stride0;
            }
            pixelrow += stride1;
        }
    }
        break;
    case 3:
    {
#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
        Uint32 Roffset = format->Rshift >> 3;
        Uint32 Goffset = format->Gshift >> 3;
        Uint32 Boffset = format->Bshift >> 3;
#else
        Uint32 Roffset = 2 - (format->Rshift >> 3);
        Uint32 Goffset = 2 - (format->Gshift >> 3);
        Uint32 Boffset = 2 - (format->Bshift >> 3);
#endif
        Uint8 white_r = (Uint8)(white >> 16);
        Uint8 white_g = (Uint8)(white >> 8);
        Uint8 white_b = (Uint8)white;
        Uint8 black_r = (Uint8)(black >> 16);
        Uint8 black_g = (Uint8)(black >> 8);
        Uint8 black_b = (Uint8)black;
        Uint32 pxcolor;
	int ppa = (surf->flags & SDL_SRCALPHA && format->Amask);

        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            for (x = 0; x < dim0; ++x) {
                pxcolor = (((Uint32)pixel_p[Roffset] << 16) +
                           ((Uint32)pixel_p[Goffset] << 8)  +
                           ((Uint32)pixel_p[Boffset])         );
                if (distance != 0.0) {
                    GET_PIXELVALS(r2, g2, b2, a2, pxcolor, format, ppa);
                    if (COLOR_DIFF_RGB(wr, wg, wb, r1, g1, b1, r2, g2, b2) <=
                        distance) {
                        pixel_p[Roffset] = white_r;
                        pixel_p[Goffset] = white_g;
                        pixel_p[Boffset] = white_b;
                    }
                    else {
                        pixel_p[Roffset] = black_r;
                        pixel_p[Goffset] = black_g;
                        pixel_p[Boffset] = black_b;
                    }
                }
                else if (pxcolor == color) {
                    pixel_p[Roffset] = white_r;
                    pixel_p[Goffset] = white_g;
                    pixel_p[Boffset] = white_b;
                }
                else {
                    pixel_p[Roffset] = black_r;
                    pixel_p[Goffset] = black_g;
                    pixel_p[Boffset] = black_b;
                }
                pixel_p += stride0;
            }
            pixelrow += stride1;
        }
    }
        break;
    default: /* case 4: */
    {
        Uint32 *px_p;
	int ppa = (surf->flags & SDL_SRCALPHA && surf->format->Amask);

        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            for (x = 0; x < dim0; ++x) {
                px_p = (Uint32 *)pixel_p;
                if (distance != 0.0) {
                    GET_PIXELVALS(r2, g2, b2, a2, *px_p, format, ppa);
                    if (COLOR_DIFF_RGB(wr, wg, wb, r1, g1, b1, r2, g2, b2) <=
                        distance) {
                        *px_p = white;
                    }
                    else {
                        *px_p = black;
                    }
                }
                else {
                    *px_p = *px_p == color ? white : black;
                }
                pixel_p += stride0;
            }
            pixelrow += stride1;
        }
    }
        break;
    }
    Py_END_ALLOW_THREADS;

    return (PyObject *)new_array;
}

static PyObject *
_compare(PyPixelArray *array, PyObject *args, PyObject *kwds)
{
    Py_ssize_t dim0 = array->dim0;
    Py_ssize_t dim1 = array->dim1;
    SDL_Surface *surf = PySurface_AsSurface(array->surface);
    SDL_PixelFormat *format;
    PyPixelArray *other_array;
    PyObject *weights = 0;
    SDL_Surface *other_surf;
    SDL_PixelFormat *other_format;
    Py_ssize_t other_stride0;
    Py_ssize_t other_stride1;
    Uint8 *other_pixels;
    int bpp;
    Uint32 black;
    Uint32 white;
    Uint8 r1, g1, b1, a1, r2, g2, b2, a2;
    Uint8 *row_p;
    Uint8 *byte_p;
    Uint8 *other_row_p;
    Uint8 *other_byte_p;
    float distance = 0;
    float wr, wg, wb;
    Py_ssize_t x;
    Py_ssize_t y;
    PyPixelArray *new_array;
    PyObject *new_surface;
    SDL_Surface *new_surf;
    SDL_PixelFormat *new_format;
    Py_ssize_t stride0;
    Py_ssize_t stride1;
    Uint8 *pixels;

    static char *keys[] = { "array", "distance", "weights", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|fO", keys, 
				     &PyPixelArray_Type, &other_array,
                                     &distance, &weights)) {
        return 0;
    }


    if (distance < 0.0 || distance > 1.0) {
        return RAISE(PyExc_ValueError,
                     "distance must be in the range from 0.0 to 1.0");
    }

    if (!_get_weights(weights, &wr, &wg, &wb)) {
        return 0;
    }

    if (other_array->dim0 != dim0 || other_array->dim1 != dim1) {
         /* Bounds do not match. */
        PyErr_SetString(PyExc_ValueError, "array sizes do not match");
        return 0;
    }

    format = surf->format;
    bpp = surf->format->BytesPerPixel;
    other_surf = PySurface_AsSurface(other_array->surface);
    other_format = other_surf->format;

    if (other_format->BytesPerPixel != bpp) {
        /* bpp do not match. We cannot guarantee that the padding and co
         * would be set correctly. */
        PyErr_SetString(PyExc_ValueError, "bit depths do not match");
        return 0;
    }

    other_stride0 = other_array->stride0;
    other_stride1 = other_array->stride1;
    other_pixels = other_array->pixels;

    /* Create the b/w mask surface. */
    new_surface = _make_surface(array);
    if (!new_surface) {
        return 0;
    }

    new_array = (PyPixelArray *)PyPixelArray_New(new_surface);
    if (!new_array) {
        Py_DECREF(new_surface);
        return 0;
    }

    new_surf = PySurface_AsSurface(new_surface);
    new_format = surf->format;
    stride0 = new_array->stride0;
    stride1 = new_array->stride1;
    pixels = new_array->pixels;

    black = SDL_MapRGBA(format, 0, 0, 0, 255);
    white = SDL_MapRGBA(format, 255, 255, 255, 255);

    Py_BEGIN_ALLOW_THREADS;
    if (!dim1) {
        dim1 = 1;
    }
    row_p = pixels;
    other_row_p = other_pixels;

    switch (bpp) {

    case 1:
    {
        Uint8 *pixel_p;
        Uint8 *other_pixel_p;

        for (y = 0; y < dim1; ++y) {
            byte_p = row_p;
            other_byte_p = other_row_p;
            for (x = 0; x < dim0; ++x) {
                pixel_p = byte_p;
                other_pixel_p = other_byte_p;
                if (distance != 0.0) {
                    GET_PIXELVALS_1(r1, g1, b1, a1, pixel_p, new_format);
                    GET_PIXELVALS_1(r2, g2, b2, a2, other_pixel_p,
                                    other_format);
                    if (COLOR_DIFF_RGB(wr, wg, wb, r1, g1, b1, r2, g2, b2) <=
                        distance) {
                        *pixel_p = (Uint8)white;
                    }
                    else {
                        *pixel_p = (Uint8)black;
                    }
                }
                else {
                    *pixel_p = (Uint8)(*pixel_p == *other_pixel_p ?
                                       white : black);
                }
                byte_p += stride0;
                other_byte_p += other_stride0;
            }
            row_p += stride1;
            other_row_p += other_stride1;
        }
    }
        break;
    case 2:
    {
        Uint16 *pixel_p;
        Uint16 *other_pixel_p;
	int ppa = (surf->flags & SDL_SRCALPHA && format->Amask);
        int other_ppa = (other_surf->flags & SDL_SRCALPHA &&
                         other_format->Amask);

        for (y = 0; y < dim1; ++y) {
            byte_p = row_p;
            other_byte_p = other_row_p;
            for (x = 0; x < dim0; ++x) {
                pixel_p = (Uint16 *)byte_p;
                other_pixel_p = (Uint16 *)other_byte_p;
                if (distance != 0.0) {
                    GET_PIXELVALS(r1, g1, b1, a1,
                                  (Uint32)*pixel_p, format, ppa);
                    GET_PIXELVALS(r2, g2, b2, a2,
                                  (Uint32)*pixel_p, other_format, other_ppa);
                    if (COLOR_DIFF_RGB(wr, wg, wb, r1, g1, b1, r2, g2, b2) <=
                        distance) {
                        *pixel_p = (Uint16)white;
                    }
                    else {
                        *pixel_p = (Uint16)black;
                    }
                }
                else {
                    *pixel_p = (Uint16)(*pixel_p == *other_pixel_p ?
                                        white : black);
                }
                byte_p += stride0;
                other_byte_p += other_stride0;
            }
            row_p += stride1;
            other_row_p += other_stride1;
        }
    }
        break;
    case 3:
    {
#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
        Uint32 Roffset = format->Rshift >> 3;
        Uint32 Goffset = format->Gshift >> 3;
        Uint32 Boffset = format->Bshift >> 3;
        Uint32 oRoffset = other_format->Rshift >> 3;
        Uint32 oGoffset = other_format->Gshift >> 3;
        Uint32 oBoffset = other_format->Bshift >> 3;
#else
        Uint32 Roffset = 2 - (format->Rshift >> 3);
        Uint32 Goffset = 2 - (format->Gshift >> 3);
        Uint32 Boffset = 2 - (format->Bshift >> 3);
        Uint32 oRoffset = 2 - (other_format->Rshift >> 3);
        Uint32 oGoffset = 2 - (other_format->Gshift >> 3);
        Uint32 oBoffset = 2 - (other_format->Bshift >> 3);
#endif
        Uint8 white_r = (Uint8)(white >> 16);
        Uint8 white_g = (Uint8)(white >> 8);
        Uint8 white_b = (Uint8)white;
        Uint8 black_r = (Uint8)(black >> 16);
        Uint8 black_g = (Uint8)(black >> 8);
        Uint8 black_b = (Uint8)black;

        for (y = 0; y < dim1; ++y) {
            byte_p = row_p;
            other_byte_p = other_row_p;
            for (x = 0; x < dim0; ++x) {
                r1 = byte_p[Roffset];
                g1 = byte_p[Goffset];
                b1 = byte_p[Boffset];
                r2 = other_byte_p[oRoffset];
                g2 = other_byte_p[oGoffset];
                b2 = other_byte_p[oBoffset];
                if (distance != 0.0) {
                    if (COLOR_DIFF_RGB(wr, wg, wb,
                                       r1, g1, b1,
                                       r2, g2, b2) <= distance) {
                        byte_p[Roffset] = white_r;
                        byte_p[Goffset] = white_g;
                        byte_p[Boffset] = white_b;
                    }
                    else {
                        byte_p[Roffset] = black_r;
                        byte_p[Goffset] = black_g;
                        byte_p[Boffset] = black_b;
                    }
                }
                else if (r1 == r2 && g1 == g2 && b1 == b2) {
                    byte_p[Roffset] = white_r;
                    byte_p[Goffset] = white_g;
                    byte_p[Boffset] = white_b;
                }
                else {
                    byte_p[Roffset] = black_r;
                    byte_p[Goffset] = black_g;
                    byte_p[Boffset] = black_b;
                }
                byte_p += stride0;
                other_byte_p += other_stride0;
            }
            row_p += stride1;
            other_row_p += other_stride1;
        }
    }
        break;
    default: /* case 4: */
    {
        Uint32 *pixel_p;
        Uint32 *other_pixel_p;
	int ppa = (surf->flags & SDL_SRCALPHA && surf->format->Amask);
        int other_ppa = (other_surf->flags & SDL_SRCALPHA &&
                         other_format->Amask);

        for (y = 0; y < dim1; ++y) {
            byte_p = row_p;
            other_byte_p = other_row_p;
            for (x = 0; x < dim0; ++x) {
                pixel_p = (Uint32 *)byte_p;
                other_pixel_p = (Uint32 *)other_byte_p;
                if (distance != 0.0) {
                    GET_PIXELVALS(r1, g1, b1, a1, *pixel_p, format, ppa);
                    GET_PIXELVALS(r2, g2, b2, a2,
                                  *pixel_p, other_format, other_ppa);
                    if (COLOR_DIFF_RGB(wr, wg, wb, r1, g1, b1, r2, g2, b2) <=
                        distance) {
                        *pixel_p = white;
                    }
                    else {
                        *pixel_p = black;
                    }
                }
                else {
                    *pixel_p = *pixel_p == *other_pixel_p ? white : black;
                }
                byte_p += stride0;
                other_byte_p += other_stride0;
            }
            row_p += stride1;
            other_row_p += other_stride1;
        }
    }
        break;
    }
    Py_END_ALLOW_THREADS;

    return (PyObject *)new_array;
}

static PyObject *
_transpose(PyPixelArray *array)
{
    SDL_Surface *surf = PySurface_AsSurface(array->surface);
    Py_ssize_t dim0 = array->dim1 ? array->dim1 : 1;
    Py_ssize_t dim1 = array->dim0;
    Py_ssize_t stride0;
    Py_ssize_t stride1 = array->stride0;

    stride0 = array->dim1 ?
        array->stride1 : array->dim0 * surf->format->BytesPerPixel;

    return (PyObject *)_pxarray_new_internal(&PyPixelArray_Type,
                                             0, array, array->pixels,
                                             dim0, dim1, stride0, stride1);
}
