/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners

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

  Pete Shinners
  pete@shinners.org
*/

#include <stddef.h>
#include "pygame.h"

#include "palette.h"

#include "pgcompat.h"

#include "doc/pixelcopy_doc.h"

#include <SDL_endian.h>

typedef enum {
    VIEWKIND_RED,
    VIEWKIND_GREEN,
    VIEWKIND_BLUE,
    VIEWKIND_ALPHA,
    VIEWKIND_COLORKEY,
    VIEWKIND_RGB
} _pc_view_kind_t;

typedef union {
    Uint32 value;
    Uint8 bytes[sizeof(Uint32)];
} _pc_pixel_t;

static int
_validate_view_format(const char *format)
{
    int i = 0;

    /* Skip size/byte order prefix or item count */
    switch (format[i]) {
        case '@':
        case '=':
        case '<':
        case '>':
        case '!':
            ++i;
            break;
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            /* Only allowed for fill bytes */
            if (format[i + 1] == 'x') {
                ++i;
            }
            break;

            /* default: assume it is a format character */
    }
    /* an item count of 1 is accepted */
    if (format[i] == '1') {
        ++i;
    }
    switch (format[i]) {
        case 'x':
        case 'b':
        case 'B':
        case 'h':
        case 'H':
        case 'i':
        case 'I':
        case 'l':
        case 'L':
        case 'q':
        case 'Q':
            ++i;
            break;

            /* default: unrecognized character; raise error later */
    }
    if (format[i] != '\0') {
        PyErr_SetString(PyExc_ValueError, "Unsupport array item type");
        return -1;
    }

    return 0;
}

static int
_is_swapped(Py_buffer *view_p)
{
    char ch = view_p->format[0];

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
    if (ch == '>' || ch == '!') {
        return 1;
    }
#else
    if (ch == '<') {
        return 1;
    }
#endif
    return 0;
}

static int
_view_kind(PyObject *obj, void *view_kind_vptr)
{
    unsigned long ch;
    _pc_view_kind_t *view_kind_ptr = (_pc_view_kind_t *)view_kind_vptr;

    if (PyUnicode_Check(obj)) {
        if (PyUnicode_GET_LENGTH(obj) != 1) {
            PyErr_SetString(PyExc_TypeError,
                            "expected a length 1 string for argument 3");
            return 0;
        }
        ch = PyUnicode_READ_CHAR(obj, 0);
    }
    else if (PyBytes_Check(obj)) {
        if (PyBytes_GET_SIZE(obj) != 1) {
            PyErr_SetString(PyExc_TypeError,
                            "expected a length 1 string for argument 3");
            return 0;
        }
        ch = *PyBytes_AS_STRING(obj);
    }
    else {
        PyErr_Format(PyExc_TypeError,
                     "expected a length one string for argument 3: got '%s'",
                     Py_TYPE(obj)->tp_name);
        return 0;
    }
    switch (ch) {
        case 'R':
        case 'r':
            *view_kind_ptr = VIEWKIND_RED;
            break;
        case 'G':
        case 'g':
            *view_kind_ptr = VIEWKIND_GREEN;
            break;
        case 'B':
        case 'b':
            *view_kind_ptr = VIEWKIND_BLUE;
            break;
        case 'A':
        case 'a':
            *view_kind_ptr = VIEWKIND_ALPHA;
            break;
        case 'C':
        case 'c':
            *view_kind_ptr = VIEWKIND_COLORKEY;
            break;
        case 'P':
        case 'p':
            *view_kind_ptr = VIEWKIND_RGB;
            break;
        default:
            PyErr_Format(PyExc_TypeError,
                         "unrecognized view kind '%c' for argument 3",
                         (int)ch);
            return 0;
    }
    return 1;
}

static int
_copy_mapped(Py_buffer *view_p, SDL_Surface *surf)
{
    Uint8 pixelsize = surf->format->BytesPerPixel;
    Py_ssize_t intsize = view_p->itemsize;
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
    char *src = (char *)surf->pixels;
#else
    char *src = (char *)surf->pixels + pixelsize - 1;
#endif
    char *dst = (char *)view_p->buf;
    int w = surf->w;
    int h = surf->h;
    Py_intptr_t dx_src = surf->format->BytesPerPixel;
    Py_intptr_t dy_src = surf->pitch;
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
    Py_intptr_t dz_src = 1;
#else
    Py_intptr_t dz_src = -1;
#endif
    Py_intptr_t dx_dst = view_p->strides[0];
    Py_intptr_t dy_dst = view_p->strides[1];
    Py_intptr_t dz_dst = 1;
    Py_intptr_t x, y, z;

    if (view_p->shape[0] != w || view_p->shape[1] != h) {
        PyErr_Format(PyExc_ValueError,
                     "Expected a (%d, %d) target: got (%d, %d)", w, h,
                     (int)view_p->shape[0], (int)view_p->shape[1]);
        return -1;
    }
    if (intsize < pixelsize) {
        PyErr_Format(PyExc_ValueError,
                     "Expected at least a target byte size of %d: got %d",
                     pixelsize, intsize);
        return -1;
    }
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
    if (_is_swapped(view_p)) {
        dst += intsize - 1;
        dz_dst = -1;
    }
#else
    if (!_is_swapped(view_p)) {
        dst += intsize - 1;
        dz_dst = -1;
    }
#endif
    for (x = 0; x < w; ++x) {
        for (y = 0; y < h; ++y) {
            for (z = 0; z < pixelsize; ++z) {
                dst[dx_dst * x + dy_dst * y + dz_dst * z] =
                    src[dx_src * x + dy_src * y + dz_src * z];
            }
            while (z < intsize) {
                dst[dx_dst * x + dy_dst * y + dz_dst * z] = 0;
                ++z;
            }
        }
    }

    return 0;
}

static int
_copy_colorplane(Py_buffer *view_p, SDL_Surface *surf,
                 _pc_view_kind_t view_kind, Uint8 opaque, Uint8 clear)
{
    SDL_PixelFormat *format = surf->format;
    int pixelsize = surf->format->BytesPerPixel;
    SDL_BlendMode mode;
    int intsize = (int)view_p->itemsize;
    char *src = (char *)surf->pixels;
    char *dst = (char *)view_p->buf;
    int w = surf->w;
    int h = surf->h;
    Py_intptr_t dx_src = surf->format->BytesPerPixel;
    Py_intptr_t dy_src = surf->pitch;
    Py_intptr_t dx_dst = view_p->strides[0];
    Py_intptr_t dy_dst = view_p->strides[1];
    Py_intptr_t dz_dst = 1;
    Py_intptr_t dz_pix;
    Py_intptr_t x, y, z;
    Uint8 r, g, b, a;
    Uint8 *element = 0;
    _pc_pixel_t pixel = {0};
    Uint32 colorkey;

    if (view_p->shape[0] != w || view_p->shape[1] != h) {
        PyErr_Format(PyExc_ValueError,
                     "Expected a (%d, %d) target: got (%d, %d)", w, h,
                     (int)view_p->shape[0], (int)view_p->shape[1]);
        return -1;
    }
    if (intsize < 1) {
        PyErr_Format(PyExc_ValueError,
                     "Expected at least a target byte size of 1: got %d",
                     intsize);
        return -1;
    }
    if (SDL_GetSurfaceBlendMode(surf, &mode) < 0) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return -1;
    }
    /* Select appropriate color plane element within the pixel */
    switch (view_kind) {
        case VIEWKIND_RED:
            element = &r;
            break;
        case VIEWKIND_GREEN:
            element = &g;
            break;
        case VIEWKIND_BLUE:
            element = &b;
            break;
        default: /* VIEWKIND_ALPHA or VIEWKIND_COLORKEY */
            /* element is unused for VIEWKIND_COLORKEY */
            assert(view_kind == VIEWKIND_ALPHA ||
                   view_kind == VIEWKIND_COLORKEY);
            element = &a;
    }
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
    dz_pix = 0;
    if (_is_swapped(view_p)) {
        dst += intsize - 1;
        dz_dst = -1;
    }
#else
    dz_pix = (unsigned)(sizeof(Uint32) - pixelsize);
    if (!_is_swapped(view_p)) {
        dst += intsize - 1;
        dz_dst = -1;
    }
#endif
    if (view_kind == VIEWKIND_COLORKEY &&
        SDL_GetColorKey(surf, &colorkey) == 0) {
        for (x = 0; x < w; ++x) {
            for (y = 0; y < h; ++y) {
                for (z = 0; z < pixelsize; ++z) {
                    pixel.bytes[dz_pix + z] = src[dx_src * x + dy_src * y + z];
                }
                dst[dx_dst * x + dy_dst * y] =
                    pixel.value == colorkey ? clear : opaque;
                for (z = 1; z < intsize; ++z) {
                    dst[dx_dst * x + dy_dst * y + dz_dst * z] = 0;
                }
            }
        }
    }
    else if ((view_kind != VIEWKIND_COLORKEY) &&
             (view_kind != VIEWKIND_ALPHA || mode != SDL_BLENDMODE_NONE)) {
        for (x = 0; x < w; ++x) {
            for (y = 0; y < h; ++y) {
                for (z = 0; z < pixelsize; ++z) {
                    pixel.bytes[dz_pix + z] = src[dx_src * x + dy_src * y + z];
                }
                SDL_GetRGBA(pixel.value, format, &r, &g, &b, &a);
                dst[dx_dst * x + dy_dst * y] = *element;
                for (z = 1; z < intsize; ++z) {
                    dst[dx_dst * x + dy_dst * y + dz_dst * z] = 0;
                }
            }
        }
    }
    else {
        for (x = 0; x < w; ++x) {
            for (y = 0; y < h; ++y) {
                dst[dx_dst * x + dy_dst * y] = opaque;
                for (z = 1; z < intsize; ++z) {
                    dst[dx_dst * x + dy_dst * y + dz_dst * z] = 0;
                }
            }
        }
    }

    return 0;
}

static int
_copy_unmapped(Py_buffer *view_p, SDL_Surface *surf)
{
    SDL_PixelFormat *format = surf->format;
    int pixelsize = surf->format->BytesPerPixel;
    int intsize = (int)view_p->itemsize;
    char *src = (char *)surf->pixels;
    char *dst = (char *)view_p->buf;
    int w = surf->w;
    int h = surf->h;
    Py_intptr_t dx_src = surf->format->BytesPerPixel;
    Py_intptr_t dy_src = surf->pitch;
    Py_intptr_t dx_dst = view_p->strides[0];
    Py_intptr_t dy_dst = view_p->strides[1];
    Py_intptr_t dp_dst = view_p->strides[2];
    Py_intptr_t dz_dst = 1;
    Py_intptr_t dz_pix;
    Py_intptr_t x, y, z;
    _pc_pixel_t pixel = {0};
    Uint8 r, g, b;

    if (view_p->shape[0] != w || view_p->shape[1] != h ||
        view_p->shape[2] != 3) {
        PyErr_Format(PyExc_ValueError,
                     "Expected a (%d, %d, 3) target: got (%d, %d, %d)", w, h,
                     (int)view_p->shape[0], (int)view_p->shape[1],
                     (int)view_p->shape[2]);
        return -1;
    }
    if (intsize < 1) {
        PyErr_Format(PyExc_ValueError,
                     "Expected at least a target byte size of 1: got %d",
                     intsize);
        return -1;
    }
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
    dz_pix = 0;
    if (_is_swapped(view_p)) {
        dst += intsize - 1;
        dz_dst = -1;
    }
#else
    dz_pix = (unsigned)(sizeof(Uint32) - pixelsize);
    if (!_is_swapped(view_p)) {
        dst += intsize - 1;
        dz_dst = -1;
    }
#endif
    for (x = 0; x < w; ++x) {
        for (y = 0; y < h; ++y) {
            for (z = 0; z < pixelsize; ++z) {
                pixel.bytes[dz_pix + z] = src[dx_src * x + dy_src * y + z];
            }
            SDL_GetRGB(pixel.value, format, &r, &g, &b);
            dst[dx_dst * x + dy_dst * y] = r;
            for (z = 1; z < intsize; ++z) {
                dst[dx_dst * x + dy_dst * y + dz_dst * z] = 0;
            }
            dst[dx_dst * x + dy_dst * y + dp_dst] = g;
            for (z = 1; z < intsize; ++z) {
                dst[dx_dst * x + dy_dst * y + dz_dst * z] = 0;
            }
            dst[dx_dst * x + dy_dst * y + 2 * dp_dst] = b;
            for (z = 1; z < intsize; ++z) {
                dst[dx_dst * x + dy_dst * y + dz_dst * z] = 0;
            }
        }
    }

    return 0;
}

/*macros used to blit arrays*/
#define COPYMACRO_2D(DST, SRC)                                               \
    for (loopy = 0; loopy < sizey; ++loopy) {                                \
        DST *imgrow = (DST *)(((char *)surf->pixels) + loopy * surf->pitch); \
        Uint8 *datarow = (Uint8 *)array_data + stridey * loopy;              \
        for (loopx = 0; loopx < sizex; ++loopx)                              \
            *(imgrow + loopx) = (DST) * (SRC *)(datarow + stridex * loopx);  \
    }

#define COPYMACRO_3D(DST, SRC)                                              \
    for (loopy = 0; loopy < sizey; ++loopy) {                               \
        DST *pix = (DST *)(((char *)surf->pixels) + surf->pitch * loopy);   \
        char *data = array_data + stridey * loopy;                          \
        for (loopx = 0; loopx < sizex; ++loopx) {                           \
            *pix++ = (DST)((*(SRC *)(data) >> Rloss << Rshift) |            \
                           (*(SRC *)(data + stridez) >> Gloss << Gshift) |  \
                           (*(SRC *)(data + stridez2) >> Bloss << Bshift) | \
                           alpha);                                          \
            data += stridex;                                                \
        }                                                                   \
    }

static PyObject *
array_to_surface(PyObject *self, PyObject *arg)
{
    pgSurfaceObject *surfobj;
    PyObject *arrayobj;
    pg_buffer pg_view;
    Py_buffer *view_p = (Py_buffer *)&pg_view;
    char *array_data;
    SDL_Surface *surf;
    SDL_PixelFormat *format;
    int loopx, loopy;
    Py_ssize_t stridex, stridey, stridez = 0, stridez2 = 0, sizex, sizey;
    int Rloss, Gloss, Bloss, Rshift, Gshift, Bshift;

    if (!PyArg_ParseTuple(arg, "O!O", &pgSurface_Type, &surfobj, &arrayobj)) {
        return NULL;
    }
    surf = pgSurface_AsSurface(surfobj);
    format = surf->format;

    if (pgObject_GetBuffer(arrayobj, &pg_view, PyBUF_RECORDS_RO)) {
        return 0;
    }

    if (_validate_view_format(view_p->format)) {
        return 0;
    }

    if (!(view_p->ndim == 2 || (view_p->ndim == 3 && view_p->shape[2] == 3))) {
        return RAISE(PyExc_ValueError, "must be a valid 2d or 3d array\n");
    }

    if (surf->format->BytesPerPixel == 0 || surf->format->BytesPerPixel > 4)
        return RAISE(PyExc_ValueError, "unsupport bit depth for surface");

    stridex = view_p->strides[0];
    stridey = view_p->strides[1];
    if (view_p->ndim == 3) {
        stridez = view_p->strides[2];
        stridez2 = stridez * 2;
    }
    else {
        stridez = 1;
        stridez2 = 2;
    }
    sizex = view_p->shape[0];
    sizey = view_p->shape[1];
    Rloss = format->Rloss;
    Gloss = format->Gloss;
    Bloss = format->Bloss;
    Rshift = format->Rshift;
    Gshift = format->Gshift;
    Bshift = format->Bshift;

    /* Do any required broadcasting. */
    if (sizex == 1) {
        sizex = surf->w;
        stridex = 0;
    }
    if (sizey == 1) {
        sizey = surf->h;
        stridey = 0;
    }

    if (sizex != surf->w || sizey != surf->h) {
        pgBuffer_Release(&pg_view);
        return RAISE(PyExc_ValueError, "array must match surface dimensions");
    }
    if (!pgSurface_LockBy(surfobj, arrayobj)) {
        pgBuffer_Release(&pg_view);
        return NULL;
    }

    array_data = (char *)view_p->buf;

    switch (surf->format->BytesPerPixel) {
        case 1:
            if (view_p->ndim == 2) {
                switch (view_p->itemsize) {
                    case sizeof(Uint8):
                        COPYMACRO_2D(Uint8, Uint8);
                        break;
                    case sizeof(Uint16):
                        COPYMACRO_2D(Uint8, Uint16);
                        break;
                    case sizeof(Uint32):
                        COPYMACRO_2D(Uint8, Uint32);
                        break;
                    case sizeof(Uint64):
                        COPYMACRO_2D(Uint8, Uint64);
                        break;
                    default:
                        pgBuffer_Release(&pg_view);
                        if (!pgSurface_UnlockBy(surfobj, arrayobj)) {
                            return NULL;
                        }
                        return RAISE(PyExc_ValueError,
                                     "unsupported datatype for array\n");
                }
            }
            else {
                pgBuffer_Release(&pg_view);
                if (!pgSurface_UnlockBy(surfobj, arrayobj)) {
                    return NULL;
                }
                return RAISE(PyExc_ValueError,
                             "unsupported datatype for array\n");
            }
            break;
        case 2:
            if (view_p->ndim == 2) {
                switch (view_p->itemsize) {
                    case sizeof(Uint16):
                        COPYMACRO_2D(Uint16, Uint16);
                        break;
                    case sizeof(Uint32):
                        COPYMACRO_2D(Uint16, Uint32);
                        break;
                    case sizeof(Uint64):
                        COPYMACRO_2D(Uint16, Uint64);
                        break;
                    default:
                        pgBuffer_Release(&pg_view);
                        if (!pgSurface_UnlockBy(surfobj, arrayobj)) {
                            return NULL;
                        }
                        return RAISE(PyExc_ValueError,
                                     "unsupported datatype for array\n");
                }
            }
            else {
                Uint16 alpha = 0;
                if (format->Amask) {
                    alpha = 255 >> format->Aloss << format->Ashift;
                }
                switch (view_p->itemsize) {
                    case sizeof(Uint8):
                        COPYMACRO_3D(Uint16, Uint8);
                        break;
                    case sizeof(Uint16):
                        COPYMACRO_3D(Uint16, Uint16);
                        break;
                    case sizeof(Uint32):
                        COPYMACRO_3D(Uint16, Uint32);
                        break;
                    case sizeof(Uint64):
                        COPYMACRO_3D(Uint16, Uint64);
                        break;
                    default:
                        pgBuffer_Release(&pg_view);
                        if (!pgSurface_UnlockBy(surfobj, arrayobj)) {
                            return NULL;
                        }
                        return RAISE(PyExc_ValueError,
                                     "unsupported datatype for array\n");
                }
            }
            break;
        case 3:
            /* Assumption: The rgb components of a 24 bit pixel are in
               separate bytes.
            */
            if (view_p->itemsize >= (view_p->ndim == 2 ? 3 : 1) &&
                view_p->itemsize <= 9) {
                size_t stridez_0 = 0;
                size_t stridez_1 = 0;
                size_t stridez_2 = 0;
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                if (view_p->ndim == 2) {
                    stridez_1 = 1;
                    stridez_2 = 2;
                }
                else {
                    size_t offset =
                        _is_swapped(view_p) ? view_p->itemsize - 1 : 0;
                    stridez_0 =
                        ((Rshift == 0 ? 0
                                      : (Gshift == 0 ? stridez : stridez2)) +
                         offset);
                    stridez_1 =
                        ((Rshift == 8 ? 0
                                      : (Gshift == 8 ? stridez : stridez2)) +
                         offset);
                    stridez_2 =
                        ((Rshift == 16 ? 0
                                       : (Gshift == 16 ? stridez : stridez2)) +
                         offset);
                }
#else
                if (view_p->ndim == 2) {
                    stridez_0 = view_p->itemsize - 3;
                    stridez_1 = stridez_0 + 1;
                    stridez_2 = stridez_1 + 1;
                }
                else {
                    size_t offset =
                        _is_swapped(view_p) ? 0 : view_p->itemsize - 1;
                    stridez_2 =
                        ((Rshift == 0 ? 0
                                      : (Gshift == 0 ? stridez : stridez2)) +
                         offset);
                    stridez_1 =
                        ((Rshift == 8 ? 0
                                      : (Gshift == 8 ? stridez : stridez2)) +
                         offset);
                    stridez_0 =
                        ((Rshift == 16 ? 0
                                       : (Gshift == 16 ? stridez : stridez2)) +
                         offset);
                }
#endif
                for (loopy = 0; loopy < sizey; ++loopy) {
                    Uint8 *pix = ((Uint8 *)surf->pixels) + surf->pitch * loopy;
                    Uint8 *data = (Uint8 *)array_data + stridey * loopy;
                    Uint8 *end = pix + 3 * sizex;
                    while (pix != end) {
                        *pix++ = *(data + stridez_0);
                        *pix++ = *(data + stridez_1);
                        *pix++ = *(data + stridez_2);
                        data += stridex;
                    }
                }
            }
            else {
                pgBuffer_Release(&pg_view);
                if (!pgSurface_UnlockBy(surfobj, arrayobj)) {
                    return NULL;
                }
                return RAISE(PyExc_ValueError,
                             "unsupported datatype for array\n");
            }
            break;
        case 4:
            if (view_p->ndim == 2) {
                switch (view_p->itemsize) {
                    case sizeof(Uint32):
                        COPYMACRO_2D(Uint32, Uint32);
                        break;
                    case sizeof(Uint64):
                        COPYMACRO_2D(Uint32, Uint64);
                        break;
                    default:
                        pgBuffer_Release(&pg_view);
                        if (!pgSurface_UnlockBy(surfobj, arrayobj)) {
                            return NULL;
                        }
                        return RAISE(PyExc_ValueError,
                                     "unsupported datatype for array\n");
                }
            }
            else {
                Uint32 alpha = 0;
                if (format->Amask) {
                    alpha = 255 >> format->Aloss << format->Ashift;
                }
                switch (view_p->itemsize) {
                    case sizeof(Uint8):
                        COPYMACRO_3D(Uint32, Uint8);
                        break;
                    case sizeof(Uint16):
                        COPYMACRO_3D(Uint32, Uint16);
                        break;
                    case sizeof(Uint32):
                        COPYMACRO_3D(Uint32, Uint32);
                        break;
                    case sizeof(Uint64):
                        COPYMACRO_3D(Uint32, Uint64);
                        break;
                    default:
                        pgBuffer_Release(&pg_view);
                        if (!pgSurface_UnlockBy(surfobj, arrayobj)) {
                            return NULL;
                        }
                        return RAISE(PyExc_ValueError,
                                     "unsupported datatype for array\n");
                }
            }
            break;
        default:
            pgBuffer_Release(&pg_view);
            if (!pgSurface_UnlockBy(surfobj, arrayobj)) {
                return NULL;
            }
            return RAISE(PyExc_RuntimeError,
                         "unsupported bit depth for image");
    }

    pgBuffer_Release(&pg_view);
    if (!pgSurface_UnlockBy(surfobj, arrayobj)) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
surface_to_array(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *arrayobj;
    pgSurfaceObject *surfobj;
    pg_buffer pg_view;
    Py_buffer *view_p = (Py_buffer *)&pg_view;
    _pc_view_kind_t view_kind = VIEWKIND_RGB;
    Uint8 opaque = 255;
    Uint8 clear = 0;
    SDL_Surface *surf;
    char *keywords[] = {"array", "surface", "kind", "opaque", "clear", 0};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OO!|O&BB", keywords, &arrayobj, &pgSurface_Type,
            &surfobj, _view_kind, (void *)&view_kind, &opaque, &clear)) {
        return 0;
    }
    if (!pgSurface_Lock(surfobj)) {
        return 0;
    }
    surf = pgSurface_AsSurface(surfobj);

    if (pgObject_GetBuffer(arrayobj, &pg_view, PyBUF_RECORDS)) {
        pgSurface_Unlock(surfobj);
        return 0;
    }
    if (_validate_view_format(view_p->format)) {
        pgBuffer_Release(&pg_view);
        pgSurface_Unlock(surfobj);
        return 0;
    }

    if (view_p->ndim == 2) {
        if (view_kind == VIEWKIND_RGB) {
            if (_copy_mapped(view_p, surf)) {
                pgBuffer_Release(&pg_view);
                pgSurface_Unlock(surfobj);
                return 0;
            }
        }
        else {
            if (_copy_colorplane(view_p, surf, view_kind, opaque, clear)) {
                pgBuffer_Release(&pg_view);
                pgSurface_Unlock(surfobj);
                return 0;
            }
        }
    }
    else if (view_p->ndim == 3) {
        if (view_kind != VIEWKIND_RGB) {
            PyErr_SetString(PyExc_ValueError,
                            "color planes only supported for 2d targets");
            pgBuffer_Release(&pg_view);
            pgSurface_Unlock(surfobj);
            return 0;
        }
        if (_copy_unmapped(view_p, surf)) {
            pgBuffer_Release(&pg_view);
            pgSurface_Unlock(surfobj);
            return 0;
        }
    }
    else {
        pgBuffer_Release(&pg_view);
        pgSurface_Unlock(surfobj);
        PyErr_Format(PyExc_ValueError, "Unsupported array depth %d",
                     (int)view_p->ndim);
        return 0;
    }

    pgBuffer_Release(&pg_view);
    if (!pgSurface_Unlock(surfobj)) {
        return 0;
    }
    Py_RETURN_NONE;
}

static PyObject *
map_array(PyObject *self, PyObject *args)
{
#define PIXELCOPY_MAX_DIM 10
    PyObject *src_array;
    PyObject *tar_array;
    pgSurfaceObject *format_surf;
    SDL_PixelFormat *format;
    pg_buffer src_pg_view;
    Py_buffer *src_view_p;
    Uint8 is_src_alloc = 0;
    Uint8 *src;
    int src_ndim;
    Py_intptr_t src_strides[PIXELCOPY_MAX_DIM];
    const int src_red = 0;
    int src_green;
    int src_blue;
    pg_buffer tar_pg_view;
    Py_buffer *tar_view_p;
    Uint8 is_tar_alloc = 0;
    Uint8 *tar;
    int ndim;
    Py_intptr_t *shape;
    Py_intptr_t *tar_strides;
    Py_ssize_t tar_itemsize;
    int tar_byte0 = 0;
    int tar_byte1 = 0;
    int tar_byte2 = 0;
    int tar_byte3 = 0;
    Py_ssize_t tar_padding_start;
    Py_ssize_t tar_padding_end;
    Py_intptr_t counters[PIXELCOPY_MAX_DIM];
    Py_ssize_t src_advances[PIXELCOPY_MAX_DIM] = {0};
    Py_ssize_t tar_advances[PIXELCOPY_MAX_DIM] = {0};
    int dim_diff;
    int dim;
    int topdim;
    _pc_pixel_t pixel = {0};
    int pix_bytesize;
    Py_ssize_t i;

    if (!PyArg_ParseTuple(args, "OOO!", &tar_array, &src_array,
                          &pgSurface_Type, &format_surf)) {
        return 0;
    }

    if (!pgSurface_Lock(format_surf)) {
        return 0;
    }

    /* Determine array shapes and check validity
     */
    if (pgObject_GetBuffer(tar_array, &tar_pg_view, PyBUF_RECORDS)) {
        goto fail;
    }
    is_tar_alloc = 1;
    tar_view_p = (Py_buffer *)&tar_pg_view;
    tar = (Uint8 *)tar_view_p->buf;
    if (_validate_view_format(tar_view_p->format)) {
        PyErr_SetString(PyExc_ValueError, "expected an integer target array");
        goto fail;
    }
    ndim = tar_view_p->ndim;
    tar_itemsize = tar_view_p->itemsize;
    shape = tar_view_p->shape;
    tar_strides = tar_view_p->strides;
    if (ndim < 1) {
        PyErr_SetString(PyExc_ValueError, "target array must be at least 1D");
        goto fail;
    }
    if (ndim > PIXELCOPY_MAX_DIM) {
        PyErr_Format(PyExc_ValueError, "target array exceeds %d dimensions",
                     (int)PIXELCOPY_MAX_DIM);
        goto fail;
    }
    if (pgObject_GetBuffer(src_array, &src_pg_view, PyBUF_RECORDS_RO)) {
        goto fail;
    }
    is_src_alloc = 1;
    src_view_p = (Py_buffer *)&src_pg_view;
    if (_validate_view_format(src_view_p->format)) {
        goto fail;
    }
    src = (Uint8 *)src_view_p->buf;
    src_ndim = src_view_p->ndim;
    if (src_ndim < 1) {
        PyErr_SetString(PyExc_ValueError, "source array must be at least 1D");
        goto fail;
    }
    if (src_view_p->shape[src_ndim - 1] != 3) {
        PyErr_Format(PyExc_ValueError,
                     "Expected a (..., 3) source array: got (..., %zd)",
                     src_view_p->shape[src_ndim - 1]);
        goto fail;
    }
    if (ndim < src_ndim - 1) {
        PyErr_Format(PyExc_ValueError,
                     "%d dimensional target has too few dimensions for"
                     " %d dimensional source",
                     ndim, src_ndim);
        goto fail;
    }
    for (dim = 0; dim != ndim; ++dim) {
        src_strides[dim] = 0;
    }
    dim_diff = ndim - src_ndim + 1;
    for (dim = dim_diff; dim != ndim; ++dim) {
        if (src_view_p->shape[dim - dim_diff] == 1) {
            src_strides[dim] = 0;
        }
        else if (src_view_p->shape[dim - dim_diff] == shape[dim]) {
            src_strides[dim] = src_view_p->strides[dim - dim_diff];
        }
        else {
            PyErr_Format(PyExc_ValueError,
                         "size mismatch between dimension %d of source and"
                         " dimension %d of target",
                         dim - dim_diff, dim);
            goto fail;
        }
    }
    for (dim = 0; dim != ndim - 1; ++dim) {
        tar_advances[dim] =
            tar_strides[dim] - shape[dim + 1] * tar_strides[dim + 1];
        src_advances[dim] =
            src_strides[dim] - shape[dim + 1] * src_strides[dim + 1];
    }

    /* Determine source and destination pixel formats
     */
    format = pgSurface_AsSurface(format_surf)->format;
    pix_bytesize = format->BytesPerPixel;
    if (tar_itemsize < pix_bytesize) {
        PyErr_SetString(PyExc_ValueError,
                        "target array itemsize is too small for pixel format");
        goto fail;
    }
    src_green = (int)src_view_p->strides[src_ndim - 1];
    src_blue = 2 * src_green;
    switch (pix_bytesize) {
        case 1:
            break;
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
        case 2:
            if (_is_swapped(tar_view_p)) {
                tar_byte2 = 1;
            }
            else {
                tar_byte3 = 1;
            }
            break;
        case 3:
            if (_is_swapped(tar_view_p)) {
                tar_byte1 = 2;
                tar_byte2 = 1;
            }
            else {
                tar_byte2 = 1;
                tar_byte3 = 2;
            }
            break;
#else
        case 2:
            if (_is_swapped(tar_view_p)) {
                tar_byte0 = 1;
            }
            else {
                tar_byte1 = 1;
            }
            break;
        case 3:
            if (_is_swapped(tar_view_p)) {
                tar_byte0 = 2;
                tar_byte1 = 1;
            }
            else {
                tar_byte1 = 1;
                tar_byte2 = 2;
            }
            break;
#endif
        case 4:
            if (_is_swapped(tar_view_p)) {
                tar_byte0 = 3;
                tar_byte1 = 2;
                tar_byte2 = 1;
            }
            else {
                tar_byte1 = 1;
                tar_byte2 = 2;
                tar_byte3 = 3;
            }
            break;
        default:
            PyErr_Format(PyExc_ValueError,
                         "%d bytes per pixel target format not supported",
                         pix_bytesize);
            goto fail;
    }

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
    if (_is_swapped(tar_view_p)) {
        tar += tar_itemsize - pix_bytesize;
        tar_padding_start = pix_bytesize - tar_itemsize;
        tar_padding_end = 0;
    }
    else {
        tar_padding_start = pix_bytesize;
        tar_padding_end = tar_itemsize;
    }
#else
    if (_is_swapped(tar_view_p)) {
        tar_padding_start = pix_bytesize;
        tar_padding_end = tar_itemsize;
    }
    else {
        tar += tar_itemsize - pix_bytesize;
        tar_padding_start = pix_bytesize - tar_itemsize;
        tar_padding_end = 0;
    }
#endif

    /* Iterate over arrays, left index varying slowest, copying pixels
     */
    dim = 0;
    topdim = ndim - 1;
    counters[0] = shape[0];
    while (counters[0]) {
        if (!counters[dim]) {
            /* Leave loop, moving left one index
             */
            --dim;
            if (dim < 0) {
                /* Should not happen, but handle case for extra safety */
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "internal pygame error in pixelcopy map_array");
                goto fail;
            }
            tar += tar_advances[dim];
            src += src_advances[dim];
            --counters[dim];
        }
        else if (dim == topdim) {
            /* Next iteration of inner most loop: copy pixel
             */
            pixel.value = SDL_MapRGB(format, src[src_red], src[src_green],
                                     src[src_blue]);
            /* Bytes are copied from the pixel in most to least significant
             * byte order. If destination bytes get overwritten, when the
             * destination size is less than 4 bytes, only zero pad bytes
             * of a pixel get clobbered in the destination write.
             */
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
            tar[tar_byte3] = pixel.bytes[3];
            tar[tar_byte2] = pixel.bytes[2];
            tar[tar_byte1] = pixel.bytes[1];
            tar[tar_byte0] = pixel.bytes[0];
#else
            tar[tar_byte0] = pixel.bytes[0];
            tar[tar_byte1] = pixel.bytes[1];
            tar[tar_byte2] = pixel.bytes[2];
            tar[tar_byte3] = pixel.bytes[3];
#endif
            for (i = tar_padding_start; i < tar_padding_end; ++i) {
                tar[i] = 0;
            }
            tar += tar_strides[dim];
            src += src_strides[dim];
            --counters[dim];
        }
        else {
            /* Enter loop for next index to the right
             */
            dim += 1;
            counters[dim] = shape[dim];
        }
    }

    /* Cleanup
     */
    pgBuffer_Release(&src_pg_view);
    pgBuffer_Release(&tar_pg_view);
    if (!pgSurface_Unlock(format_surf)) {
        return 0;
    }
    Py_RETURN_NONE;

fail:
    if (is_src_alloc) {
        pgBuffer_Release(&src_pg_view);
    }
    if (is_tar_alloc) {
        pgBuffer_Release(&tar_pg_view);
    }
    pgSurface_Unlock(format_surf);
    return 0;
}

static PyObject *
make_surface(PyObject *self, PyObject *arg)
{
    pg_buffer pg_view;
    Py_buffer *view_p = (Py_buffer *)&pg_view;
    pgSurfaceObject *surfobj;
    PyObject *args;
    PyObject *result;
    SDL_Surface *surf;
    int sizex, sizey, bitsperpixel;
    Uint32 rmask, gmask, bmask;

    if (pgObject_GetBuffer(arg, &pg_view, PyBUF_RECORDS_RO)) {
        return 0;
    }

    if (!(view_p->ndim == 2 || (view_p->ndim == 3 && view_p->shape[2] == 3))) {
        pgBuffer_Release(&pg_view);
        return RAISE(PyExc_ValueError, "must be a valid 2d or 3d array\n");
    }
    if (_validate_view_format(view_p->format)) {
        pgBuffer_Release(&pg_view);
        return NULL;
    }

    if (view_p->ndim == 2) {
        bitsperpixel = 8;
        rmask = 0;
        gmask = 0;
        bmask = 0;
    }
    else {
        bitsperpixel = 32;
        rmask = 0xFF << 16;
        gmask = 0xFF << 8;
        bmask = 0xFF;
    }
    sizex = (int)view_p->shape[0];
    sizey = (int)view_p->shape[1];

    surf = SDL_CreateRGBSurface(0, sizex, sizey, bitsperpixel, rmask, gmask,
                                bmask, 0);
    if (!surf) {
        pgBuffer_Release(&pg_view);
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    if (SDL_ISPIXELFORMAT_INDEXED(surf->format->format)) {
        /* Give the surface something other than an all white palette.
         *          */
        if (SDL_SetPaletteColors(surf->format->palette, default_palette_colors,
                                 0, default_palette_size - 1) != 0) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(surf);
            return 0;
        }
    }
    surfobj = pgSurface_New(surf);
    if (!surfobj) {
        pgBuffer_Release(&pg_view);
        SDL_FreeSurface(surf);
        return 0;
    }

    args = Py_BuildValue("(OO)", surfobj, arg);
    if (!args) {
        pgBuffer_Release(&pg_view);
        Py_DECREF(surfobj);
        return 0;
    }

    result = array_to_surface(self, args);
    pgBuffer_Release(&pg_view);
    Py_DECREF(args);

    if (!result) {
        Py_DECREF(surfobj);
        return 0;
    }
    Py_DECREF(result);
    return (PyObject *)surfobj;
}

static PyMethodDef _pixelcopy_methods[] = {
    {"array_to_surface", array_to_surface, METH_VARARGS,
     DOC_PYGAMEPIXELCOPYARRAYTOSURFACE},
    {"surface_to_array", (PyCFunction)surface_to_array,
     METH_VARARGS | METH_KEYWORDS, DOC_PYGAMEPIXELCOPYSURFACETOARRAY},
    {"map_array", map_array, METH_VARARGS, DOC_PYGAMEPIXELCOPYMAPARRAY},
    {"make_surface", make_surface, METH_O, DOC_PYGAMEPIXELCOPYMAKESURFACE},
    {0, 0, 0, 0}};

MODINIT_DEFINE(pixelcopy)
{
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "pixelcopy",
                                         DOC_PYGAMEPIXELCOPY,
                                         -1,
                                         _pixelcopy_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        return NULL;
    }
    import_pygame_surface();
    if (PyErr_Occurred()) {
        return NULL;
    }

    return PyModule_Create(&_module);
}
