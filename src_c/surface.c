/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners
  Copyright (C) 2007 Marcus von Appen

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

#define PYGAMEAPI_SURFACE_INTERNAL

#include "surface.h"

#include "palette.h"

#include "structmember.h"
#include "pgcompat.h"
#include "doc/surface_doc.h"

/* stdint.h is missing from some versions of MSVC. */
#ifdef _MSC_VER
#ifndef UINT32_MAX
#define UINT32_MAX 0xFFFFFFFF
#endif
#else
#include <stdint.h>
#endif /* _MSC_VER */

typedef enum {
    VIEWKIND_0D = 0,
    VIEWKIND_1D = 1,
    VIEWKIND_2D = 2,
    VIEWKIND_3D = 3,
    VIEWKIND_RED,
    VIEWKIND_GREEN,
    VIEWKIND_BLUE,
    VIEWKIND_ALPHA
} SurfViewKind;

/* To avoid problems with non-const Py_buffer format field */
#define FormatUint8 "B"
#define FormatUint16 "=H"
#define FormatUint24 "3x"
#define FormatUint32 "=I"

typedef struct pg_bufferinternal_s {
    PyObject *consumer_ref; /* A weak reference to a bufferproxy object   */
    Py_ssize_t mem[6];      /* Enough memory for dim 3 shape and strides  */
} pg_bufferinternal;

int
pgSurface_Blit(pgSurfaceObject *dstobj, pgSurfaceObject *srcobj,
               SDL_Rect *dstrect, SDL_Rect *srcrect, int blend_flags);

/* statics */
static pgSurfaceObject *
pgSurface_New2(SDL_Surface *info, int owner);
static PyObject *
surf_subtype_new(PyTypeObject *type, SDL_Surface *s, int owner);
static PyObject *
surface_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static intptr_t
surface_init(pgSurfaceObject *self, PyObject *args, PyObject *kwds);
static PyObject *
surface_str(PyObject *self);
static void
surface_dealloc(PyObject *self);
static void
surface_cleanup(pgSurfaceObject *self);

static PyObject *
surf_get_at(PyObject *self, PyObject *args);
static PyObject *
surf_set_at(PyObject *self, PyObject *const *args, Py_ssize_t nargs);
static PyObject *
surf_get_at_mapped(PyObject *self, PyObject *args);
static PyObject *
surf_map_rgb(PyObject *self, PyObject *args);
static PyObject *
surf_unmap_rgb(PyObject *self, PyObject *arg);
static PyObject *
surf_lock(PyObject *self, PyObject *args);
static PyObject *
surf_unlock(PyObject *self, PyObject *args);
static PyObject *
surf_mustlock(PyObject *self, PyObject *args);
static PyObject *
surf_get_locked(PyObject *self, PyObject *args);
static PyObject *
surf_get_locks(PyObject *self, PyObject *args);
static PyObject *
surf_get_palette(PyObject *self, PyObject *args);
static PyObject *
surf_get_palette_at(PyObject *self, PyObject *args);
static PyObject *
surf_set_palette(PyObject *self, PyObject *seq);
static PyObject *
surf_set_palette_at(PyObject *self, PyObject *args);
static PyObject *
surf_set_colorkey(pgSurfaceObject *self, PyObject *args);
static PyObject *
surf_get_colorkey(pgSurfaceObject *self, PyObject *args);
static PyObject *
surf_set_alpha(pgSurfaceObject *self, PyObject *args);
static PyObject *
surf_get_alpha(pgSurfaceObject *self, PyObject *args);
static PyObject *
surf_get_blendmode(PyObject *self, PyObject *args);
static PyObject *
surf_copy(pgSurfaceObject *self, PyObject *args);
static PyObject *
surf_convert(pgSurfaceObject *self, PyObject *args);
static PyObject *
surf_convert_alpha(pgSurfaceObject *self, PyObject *args);
static PyObject *
surf_set_clip(PyObject *self, PyObject *args);
static PyObject *
surf_get_clip(PyObject *self, PyObject *args);
static PyObject *
surf_blit(pgSurfaceObject *self, PyObject *args, PyObject *keywds);
static PyObject *
surf_blits(pgSurfaceObject *self, PyObject *args, PyObject *keywds);
static PyObject *
surf_fblits(pgSurfaceObject *self, PyObject *const *args, Py_ssize_t nargs);
static PyObject *
surf_fill(pgSurfaceObject *self, PyObject *args, PyObject *keywds);
static PyObject *
surf_scroll(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *
surf_get_abs_offset(PyObject *self, PyObject *args);
static PyObject *
surf_get_abs_parent(PyObject *self, PyObject *args);
static PyObject *
surf_get_bitsize(PyObject *self, PyObject *args);
static PyObject *
surf_get_bytesize(PyObject *self, PyObject *args);
static PyObject *
surf_get_flags(PyObject *self, PyObject *args);
static PyObject *
surf_get_height(PyObject *self, PyObject *args);
static PyObject *
surf_get_pitch(PyObject *self, PyObject *args);
static PyObject *
surf_get_rect(PyObject *self, PyObject *const *args, Py_ssize_t nargs,
              PyObject *kwnames);
static PyObject *
surf_get_frect(PyObject *self, PyObject *const *args, Py_ssize_t nargs,
               PyObject *kwnames);
static PyObject *
surf_get_width(PyObject *self, PyObject *args);
static PyObject *
surf_get_shifts(PyObject *self, PyObject *args);
static PyObject *
surf_set_shifts(PyObject *self, PyObject *args);
static PyObject *
surf_get_size(PyObject *self, PyObject *args);
static PyObject *
surf_get_losses(PyObject *self, PyObject *args);
static PyObject *
surf_get_masks(PyObject *self, PyObject *args);
static PyObject *
surf_set_masks(PyObject *self, PyObject *args);
static PyObject *
surf_get_offset(PyObject *self, PyObject *args);
static PyObject *
surf_get_parent(PyObject *self, PyObject *args);
static PyObject *
surf_subsurface(PyObject *self, PyObject *args);
static PyObject *
surf_get_view(PyObject *self, PyObject *args);
static PyObject *
surf_get_buffer(PyObject *self, PyObject *args);
static PyObject *
surf_get_bounding_rect(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *
surf_get_pixels_address(PyObject *self, PyObject *closure);
static PyObject *
surf_premul_alpha(pgSurfaceObject *self, PyObject *args);
static PyObject *
surf_premul_alpha_ip(pgSurfaceObject *self, PyObject *args);
static int
_view_kind(PyObject *obj, void *view_kind_vptr);
static int
_get_buffer_0D(PyObject *obj, Py_buffer *view_p, int flags);
static int
_get_buffer_1D(PyObject *obj, Py_buffer *view_p, int flags);
static int
_get_buffer_2D(PyObject *obj, Py_buffer *view_p, int flags);
static int
_get_buffer_3D(PyObject *obj, Py_buffer *view_p, int flags);
static int
_get_buffer_red(PyObject *obj, Py_buffer *view_p, int flags);
static int
_get_buffer_green(PyObject *obj, Py_buffer *view_p, int flags);
static int
_get_buffer_blue(PyObject *obj, Py_buffer *view_p, int flags);
static int
_get_buffer_alpha(PyObject *obj, Py_buffer *view_p, int flags);
static int
_get_buffer_colorplane(PyObject *obj, Py_buffer *view_p, int flags, char *name,
                       Uint32 mask);
static int
_init_buffer(PyObject *surf, Py_buffer *view_p, int flags);
static void
_release_buffer(Py_buffer *view_p);
static PyObject *
_raise_get_view_ndim_error(int bitsize, SurfViewKind kind);
static SDL_Surface *
pg_DisplayFormatAlpha(SDL_Surface *surface);
static SDL_Surface *
pg_DisplayFormat(SDL_Surface *surface);
static int
_PgSurface_SrcAlpha(SDL_Surface *surf);

static PyGetSetDef surface_getsets[] = {
    {"_pixels_address", (getter)surf_get_pixels_address, NULL,
     "pixel buffer address (readonly)", NULL},
    {"width", (getter)surf_get_width, NULL, DOC_SURFACE_WIDTH, NULL},
    {"height", (getter)surf_get_height, NULL, DOC_SURFACE_HEIGHT, NULL},
    {"size", (getter)surf_get_size, NULL, DOC_SURFACE_SIZE, NULL},
    {NULL, NULL, NULL, NULL, NULL}};

static struct PyMethodDef surface_methods[] = {
    {"get_at", surf_get_at, METH_O, DOC_SURFACE_GETAT},
    {"set_at", (PyCFunction)surf_set_at, METH_FASTCALL, DOC_SURFACE_SETAT},
    {"get_at_mapped", surf_get_at_mapped, METH_O, DOC_SURFACE_GETATMAPPED},
    {"map_rgb", surf_map_rgb, METH_VARARGS, DOC_SURFACE_MAPRGB},
    {"unmap_rgb", surf_unmap_rgb, METH_O, DOC_SURFACE_UNMAPRGB},

    {"get_palette", surf_get_palette, METH_NOARGS, DOC_SURFACE_GETPALETTE},
    {"get_palette_at", surf_get_palette_at, METH_VARARGS,
     DOC_SURFACE_GETPALETTEAT},
    {"set_palette", surf_set_palette, METH_O, DOC_SURFACE_SETPALETTE},
    {"set_palette_at", surf_set_palette_at, METH_VARARGS,
     DOC_SURFACE_SETPALETTEAT},

    {"lock", surf_lock, METH_NOARGS, DOC_SURFACE_LOCK},
    {"unlock", surf_unlock, METH_NOARGS, DOC_SURFACE_UNLOCK},
    {"mustlock", surf_mustlock, METH_NOARGS, DOC_SURFACE_MUSTLOCK},
    {"get_locked", surf_get_locked, METH_NOARGS, DOC_SURFACE_GETLOCKED},
    {"get_locks", surf_get_locks, METH_NOARGS, DOC_SURFACE_GETLOCKS},

    {"set_colorkey", (PyCFunction)surf_set_colorkey, METH_VARARGS,
     DOC_SURFACE_SETCOLORKEY},
    {"get_colorkey", (PyCFunction)surf_get_colorkey, METH_NOARGS,
     DOC_SURFACE_GETCOLORKEY},
    {"set_alpha", (PyCFunction)surf_set_alpha, METH_VARARGS,
     DOC_SURFACE_SETALPHA},
    {"get_alpha", (PyCFunction)surf_get_alpha, METH_NOARGS,
     DOC_SURFACE_GETALPHA},
    {"get_blendmode", surf_get_blendmode, METH_NOARGS,
     "Return the surface's SDL 2 blend mode"},

    {"copy", (PyCFunction)surf_copy, METH_NOARGS, DOC_SURFACE_COPY},
    {"__copy__", (PyCFunction)surf_copy, METH_NOARGS, DOC_SURFACE_COPY},
    {"__deepcopy__", (PyCFunction)surf_copy, METH_O, DOC_SURFACE_COPY},
    {"convert", (PyCFunction)surf_convert, METH_VARARGS, DOC_SURFACE_CONVERT},
    {"convert_alpha", (PyCFunction)surf_convert_alpha, METH_VARARGS,
     DOC_SURFACE_CONVERTALPHA},

    {"set_clip", surf_set_clip, METH_VARARGS, DOC_SURFACE_SETCLIP},
    {"get_clip", surf_get_clip, METH_NOARGS, DOC_SURFACE_GETCLIP},

    {"fill", (PyCFunction)surf_fill, METH_VARARGS | METH_KEYWORDS,
     DOC_SURFACE_FILL},
    {"blit", (PyCFunction)surf_blit, METH_VARARGS | METH_KEYWORDS,
     DOC_SURFACE_BLIT},
    {"blits", (PyCFunction)surf_blits, METH_VARARGS | METH_KEYWORDS,
     DOC_SURFACE_BLITS},
    {"fblits", (PyCFunction)surf_fblits, METH_FASTCALL, DOC_SURFACE_FBLITS},
    {"scroll", (PyCFunction)surf_scroll, METH_VARARGS | METH_KEYWORDS,
     DOC_SURFACE_SCROLL},

    {"get_flags", surf_get_flags, METH_NOARGS, DOC_SURFACE_GETFLAGS},
    {"get_size", surf_get_size, METH_NOARGS, DOC_SURFACE_GETSIZE},
    {"get_width", surf_get_width, METH_NOARGS, DOC_SURFACE_GETWIDTH},
    {"get_height", surf_get_height, METH_NOARGS, DOC_SURFACE_GETHEIGHT},
    {"get_rect", (PyCFunction)surf_get_rect, METH_FASTCALL | METH_KEYWORDS,
     DOC_SURFACE_GETRECT},
    {"get_frect", (PyCFunction)surf_get_frect, METH_FASTCALL | METH_KEYWORDS,
     DOC_SURFACE_GETFRECT},
    {"get_pitch", surf_get_pitch, METH_NOARGS, DOC_SURFACE_GETPITCH},
    {"get_bitsize", surf_get_bitsize, METH_NOARGS, DOC_SURFACE_GETBITSIZE},
    {"get_bytesize", surf_get_bytesize, METH_NOARGS, DOC_SURFACE_GETBYTESIZE},
    {"get_masks", surf_get_masks, METH_NOARGS, DOC_SURFACE_GETMASKS},
    {"get_shifts", surf_get_shifts, METH_NOARGS, DOC_SURFACE_GETSHIFTS},
    {"set_masks", surf_set_masks, METH_VARARGS, DOC_SURFACE_SETMASKS},
    {"set_shifts", surf_set_shifts, METH_VARARGS, DOC_SURFACE_SETSHIFTS},

    {"get_losses", surf_get_losses, METH_NOARGS, DOC_SURFACE_GETLOSSES},

    {"subsurface", surf_subsurface, METH_VARARGS, DOC_SURFACE_SUBSURFACE},
    {"get_offset", surf_get_offset, METH_NOARGS, DOC_SURFACE_GETOFFSET},
    {"get_abs_offset", surf_get_abs_offset, METH_NOARGS,
     DOC_SURFACE_GETABSOFFSET},
    {"get_parent", surf_get_parent, METH_NOARGS, DOC_SURFACE_GETPARENT},
    {"get_abs_parent", surf_get_abs_parent, METH_NOARGS,
     DOC_SURFACE_GETABSPARENT},
    {"get_bounding_rect", (PyCFunction)surf_get_bounding_rect,
     METH_VARARGS | METH_KEYWORDS, DOC_SURFACE_GETBOUNDINGRECT},
    {"get_view", surf_get_view, METH_VARARGS, DOC_SURFACE_GETVIEW},
    {"get_buffer", surf_get_buffer, METH_NOARGS, DOC_SURFACE_GETBUFFER},
    {"premul_alpha", (PyCFunction)surf_premul_alpha, METH_NOARGS,
     DOC_SURFACE_PREMULALPHA},
    {"premul_alpha_ip", (PyCFunction)surf_premul_alpha_ip, METH_NOARGS,
     DOC_SURFACE_PREMULALPHAIP},

    {NULL, NULL, 0, NULL}};

static PyTypeObject pgSurface_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pygame.surface.Surface",
    .tp_basicsize = sizeof(pgSurfaceObject),
    .tp_dealloc = surface_dealloc,
    .tp_repr = surface_str,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = DOC_SURFACE,
    .tp_weaklistoffset = offsetof(pgSurfaceObject, weakreflist),
    .tp_methods = surface_methods,
    .tp_getset = surface_getsets,
    .tp_init = (initproc)surface_init,
    .tp_new = surface_new,
};

#define pgSurface_Check(x) \
    (PyObject_IsInstance((x), (PyObject *)&pgSurface_Type))

static pgSurfaceObject *
pgSurface_New2(SDL_Surface *s, int owner)
{
    return (pgSurfaceObject *)surf_subtype_new(&pgSurface_Type, s, owner);
}

static int
pgSurface_SetSurface(pgSurfaceObject *self, SDL_Surface *s, int owner)
{
    if (!s) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return -1;
    }
    if (s == self->surf) {
        self->owner = owner;
        return 0;
    }

    surface_cleanup(self);
    self->surf = s;
    self->owner = owner;
    return 0;
}

static PyObject *
surf_subtype_new(PyTypeObject *type, SDL_Surface *s, int owner)
{
    pgSurfaceObject *self;

    if (!s) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    self = (pgSurfaceObject *)pgSurface_Type.tp_new(type, NULL, NULL);

    if (pgSurface_SetSurface(self, s, owner)) {
        return NULL;
    }

    return (PyObject *)self;
}

static PyObject *
surface_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    pgSurfaceObject *self;

    self = (pgSurfaceObject *)type->tp_alloc(type, 0);
    if (self) {
        self->surf = NULL;
        self->owner = 0;
        self->subsurface = NULL;
        self->weakreflist = NULL;
        self->dependency = NULL;
        self->locklist = NULL;
    }
    return (PyObject *)self;
}

/* surface object internals */
static void
surface_cleanup(pgSurfaceObject *self)
{
    if (self->surf && self->owner) {
        SDL_FreeSurface(self->surf);
        self->surf = NULL;
    }
    if (self->subsurface) {
        Py_XDECREF(self->subsurface->owner);
        PyMem_Free(self->subsurface);
        self->subsurface = NULL;
    }
    if (self->dependency) {
        Py_DECREF(self->dependency);
        self->dependency = NULL;
    }

    if (self->locklist) {
        Py_DECREF(self->locklist);
        self->locklist = NULL;
    }
    self->owner = 0;
}

static void
surface_dealloc(PyObject *self)
{
    if (((pgSurfaceObject *)self)->weakreflist) {
        PyObject_ClearWeakRefs(self);
    }
    surface_cleanup((pgSurfaceObject *)self);
    Py_TYPE(self)->tp_free(self);
}

static PyObject *
surface_str(PyObject *self)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);

    if (!surf) {
        return PyUnicode_FromString("<Surface(Dead Display)>");
    }

    PyObject *colorkey = surf_get_colorkey((pgSurfaceObject *)self, NULL);
    if (colorkey == NULL) {
        return NULL;
    }
    PyObject *global_alpha = surf_get_alpha((pgSurfaceObject *)self, NULL);
    if (global_alpha == NULL) {
        Py_DECREF(colorkey);
        return NULL;
    }

    // 49 is max len of format str, plus null terminator
    char format[50] = "<Surface(%dx%dx%d";
    if (PyObject_IsTrue(colorkey)) {
        strcat(format, ", colorkey=%S");
    }
    if (PyObject_IsTrue(global_alpha)) {
        strcat(format, ", global_alpha=%S");
    }
    strcat(format, ")>");

    // Because PyUnicode_FromFormat ignores extra args, if we have no colorkey,
    // but alpha, we can "move up" the global alpha to this spot No need to do
    // this vice-versa, as it ignores extra args
    PyObject *formatted_str = PyUnicode_FromFormat(
        format, surf->w, surf->h, PG_SURF_BitsPerPixel(surf),
        PyObject_IsTrue(colorkey) ? colorkey : global_alpha, global_alpha);

    Py_DECREF(colorkey);
    Py_DECREF(global_alpha);

    return formatted_str;
}

static intptr_t
surface_init(pgSurfaceObject *self, PyObject *args, PyObject *kwds)
{
    Uint32 flags = 0;
    int width, height;
    PyObject *depth = NULL, *masks = NULL, *size = NULL;
    int bpp;
    Uint32 Rmask, Gmask, Bmask, Amask;
    SDL_Surface *surface;

#if SDL_VERSION_ATLEAST(3, 0, 0)
    PG_PixelFormatEnum format = SDL_PIXELFORMAT_UNKNOWN;
#else
    SDL_PixelFormat default_format;
    default_format.palette = NULL;
#endif

    char *kwids[] = {"size", "flags", "depth", "masks", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iOO", kwids, &size, &flags,
                                     &depth, &masks)) {
        return -1;
    }

    if (PySequence_Check(size) && PySequence_Length(size) == 2) {
        if ((!pg_IntFromObjIndex(size, 0, &width)) ||
            (!pg_IntFromObjIndex(size, 1, &height))) {
            PyErr_SetString(PyExc_ValueError,
                            "size needs to be (number width, number height)");
            return -1;
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                        "size needs to be (number width, number height)");
        return -1;
    }

    if (width < 0 || height < 0) {
        PyErr_SetString(pgExc_SDLError, "Invalid resolution for Surface");
        return -1;
    }

    surface_cleanup(self);

#if SDL_VERSION_ATLEAST(3, 0, 0)
    if (depth && masks) { /* all info supplied, most errorchecking
                           * needed */
        if (pgSurface_Check(depth)) {
            PyErr_SetString(PyExc_ValueError,
                            "cannot pass surface for depth and color masks");
            return -1;
        }
        if (!pg_IntFromObj(depth, &bpp)) {
            PyErr_SetString(PyExc_ValueError,
                            "invalid bits per pixel depth argument");
            return -1;
        }
        if (!PySequence_Check(masks) || PySequence_Length(masks) != 4) {
            PyErr_SetString(PyExc_ValueError,
                            "masks argument must be sequence of four numbers");
            return -1;
        }
        if (!pg_UintFromObjIndex(masks, 0, &Rmask) ||
            !pg_UintFromObjIndex(masks, 1, &Gmask) ||
            !pg_UintFromObjIndex(masks, 2, &Bmask) ||
            !pg_UintFromObjIndex(masks, 3, &Amask)) {
            PyErr_SetString(PyExc_ValueError,
                            "invalid mask values in masks sequence");
            return -1;
        }

        format = SDL_MasksToPixelFormatEnum(bpp, Rmask, Gmask, Bmask, Amask);
    }
    else if (depth && PyNumber_Check(depth)) { /* use default masks */
        if (!pg_IntFromObj(depth, &bpp)) {
            PyErr_SetString(PyExc_ValueError,
                            "invalid bits per pixel depth argument");
            return -1;
        }
        if (flags & PGS_SRCALPHA) {
            switch (bpp) {
                case 16:
                    format = SDL_PIXELFORMAT_ARGB4444;
                    break;
                case 32:
                    format = SDL_PIXELFORMAT_ARGB8888;
                    break;
                default:
                    PyErr_SetString(
                        PyExc_ValueError,
                        "no standard masks exist for given bitdepth with "
                        "alpha");
                    return -1;
            }
        }
        else {
            switch (bpp) {
                case 8:
                    format = SDL_PIXELFORMAT_INDEX8;
                    break;
                case 12:
                    format = SDL_PIXELFORMAT_XRGB4444;
                    break;
                case 15:
                    format = SDL_PIXELFORMAT_XRGB1555;
                    break;
                case 16:
                    format = SDL_PIXELFORMAT_RGB565;
                    break;
                case 24:
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
                    format = SDL_PIXELFORMAT_RGB24;
#else
                    format = SDL_PIXELFORMAT_BGR24;
#endif
                    break;
                case 32:
                    format = SDL_PIXELFORMAT_XRGB8888;
                    break;
                default:
                    PyErr_SetString(PyExc_ValueError,
                                    "nonstandard bit depth given");
                    return -1;
            }
        }
    }
    else { /* no depth or surface */
        if (depth && pgSurface_Check(depth)) {
            format = PG_SURF_FORMATENUM(((pgSurfaceObject *)depth)->surf);
        }
        else if (pg_GetDefaultWindowSurface()) {
            format = PG_SURF_FORMATENUM(
                pgSurface_AsSurface(pg_GetDefaultWindowSurface()));
        }
        else {
            format = SDL_PIXELFORMAT_XRGB8888;
        }

        if (flags & PGS_SRCALPHA) {
            switch (SDL_BITSPERPIXEL(format)) {
                case 16:
                    format = SDL_PIXELFORMAT_ARGB4444;
                    break;
                case 24:
                    // we automatically step up to 32 if video is 24, fall
                    // through to case below
                case 32:
                    format = SDL_PIXELFORMAT_ARGB8888;
                    break;
                default:
                    PyErr_SetString(
                        PyExc_ValueError,
                        "no standard masks exist for given bitdepth with "
                        "alpha");
                    return -1;
            }
        }
    }

    if (format == SDL_PIXELFORMAT_UNKNOWN) {
        PyErr_SetString(PyExc_ValueError, "Invalid mask values");
        return -1;
    }

    surface = PG_CreateSurface(width, height, format);
    if (!surface) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return -1;
    }

    if (!(flags & PGS_SRCALPHA)) {
        /* We ignore the error if any. */
        SDL_SetSurfaceBlendMode(surface, SDL_BLENDMODE_NONE);

        /* When the display format has a full alpha channel (macOS right now),
         * Surfaces may be created with an unreqested alpha channel, which
         * could cause issues.
         * pygame Surfaces are supposed to be (0, 0, 0, 255) by default.
         * This is a simple fix to fill it with (0, 0, 0, 255) if necessary.
         * See Github issue:
         * https://github.com/pygame/pygame/issues/796
         */
        PG_PixelFormat *surf_format;
        SDL_Palette *surf_palette;
        if (!PG_GetSurfaceDetails(surface, &surf_format, &surf_palette)) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(surface);
            return -1;
        }

        if (surf_format->Amask != 0) {
            SDL_FillRect(surface, NULL,
                         PG_MapRGBA(surf_format, surf_palette, 0, 0, 0, 255));
        }
    }

    if (SDL_ISPIXELFORMAT_INDEXED(PG_SURF_FORMATENUM(surface))) {
        /* Give the surface something other than an all white palette.
         *          */
        SDL_Palette *surf_palette = SDL_CreateSurfacePalette(surface);
        if (!PG_SetPaletteColors(surf_palette, default_palette_colors, 0,
                                 default_palette_size - 1)) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(surface);
            return -1;
        }
    }
#else
    if (depth && masks) { /* all info supplied, most errorchecking
                           * needed */
        if (pgSurface_Check(depth)) {
            PyErr_SetString(PyExc_ValueError,
                            "cannot pass surface for depth and color masks");
            return -1;
        }
        if (!pg_IntFromObj(depth, &bpp)) {
            PyErr_SetString(PyExc_ValueError,
                            "invalid bits per pixel depth argument");
            return -1;
        }
        if (!PySequence_Check(masks) || PySequence_Length(masks) != 4) {
            PyErr_SetString(PyExc_ValueError,
                            "masks argument must be sequence of four numbers");
            return -1;
        }
        if (!pg_UintFromObjIndex(masks, 0, &Rmask) ||
            !pg_UintFromObjIndex(masks, 1, &Gmask) ||
            !pg_UintFromObjIndex(masks, 2, &Bmask) ||
            !pg_UintFromObjIndex(masks, 3, &Amask)) {
            PyErr_SetString(PyExc_ValueError,
                            "invalid mask values in masks sequence");
            return -1;
        }
    }
    else if (depth && PyNumber_Check(depth)) { /* use default masks */
        if (!pg_IntFromObj(depth, &bpp)) {
            PyErr_SetString(PyExc_ValueError,
                            "invalid bits per pixel depth argument");
            return -1;
        }
        if (flags & PGS_SRCALPHA) {
            switch (bpp) {
                case 16:
                    Rmask = 0xFu << 8;
                    Gmask = 0xFu << 4;
                    Bmask = 0xFu;
                    Amask = 0xFu << 12;
                    break;
                case 32:
                    Rmask = 0xFFu << 16;
                    Gmask = 0xFFu << 8;
                    Bmask = 0xFFu;
                    Amask = 0xFFu << 24;
                    break;
                default:
                    PyErr_SetString(
                        PyExc_ValueError,
                        "no standard masks exist for given bitdepth with "
                        "alpha");
                    return -1;
            }
        }
        else {
            Amask = 0;
            switch (bpp) {
                case 8:
                    Rmask = 0;
                    Gmask = 0;
                    Bmask = 0;
                    break;
                case 12:
                    Rmask = 0xFF >> 4 << 8;
                    Gmask = 0xFF >> 4 << 4;
                    Bmask = 0xFF >> 4;
                    break;
                case 15:
                    Rmask = 0xFF >> 3 << 10;
                    Gmask = 0xFF >> 3 << 5;
                    Bmask = 0xFF >> 3;
                    break;
                case 16:
                    Rmask = 0xFF >> 3 << 11;
                    Gmask = 0xFF >> 2 << 5;
                    Bmask = 0xFF >> 3;
                    break;
                case 24:
                case 32:
                    Rmask = 0xFF << 16;
                    Gmask = 0xFF << 8;
                    Bmask = 0xFF;
                    break;
                default:
                    PyErr_SetString(PyExc_ValueError,
                                    "nonstandard bit depth given");
                    return -1;
            }
        }
    }
    else { /* no depth or surface */
        SDL_PixelFormat *pix;
        if (depth && pgSurface_Check(depth)) {
            pix = ((pgSurfaceObject *)depth)->surf->format;
        }
        else if (pg_GetDefaultWindowSurface()) {
            pix = pgSurface_AsSurface(pg_GetDefaultWindowSurface())->format;
        }
        else {
            pix = &default_format;
            pix->BitsPerPixel = 32;
            pix->Amask = 0;
            pix->Rmask = 0xFF0000;
            pix->Gmask = 0xFF00;
            pix->Bmask = 0xFF;
        }
        bpp = PG_FORMAT_BitsPerPixel(pix);

        if (flags & PGS_SRCALPHA) {
            switch (bpp) {
                case 16:
                    Rmask = 0xFu << 8;
                    Gmask = 0xFu << 4;
                    Bmask = 0xFu;
                    Amask = 0xFu << 12;
                    break;
                case 24:
                    bpp = 32;
                    // we automatically step up to 32 if video is 24, fall
                    // through to case below
                case 32:
                    Rmask = 0xFFu << 16;
                    Gmask = 0xFFu << 8;
                    Bmask = 0xFFu;
                    Amask = 0xFFu << 24;
                    break;
                default:
                    PyErr_SetString(
                        PyExc_ValueError,
                        "no standard masks exist for given bitdepth with "
                        "alpha");
                    return -1;
            }
        }
        else {
            Rmask = pix->Rmask;
            Gmask = pix->Gmask;
            Bmask = pix->Bmask;
            Amask = pix->Amask;
        }
    }

    Uint32 pxformat =
        SDL_MasksToPixelFormatEnum(bpp, Rmask, Gmask, Bmask, Amask);
    if (pxformat == SDL_PIXELFORMAT_UNKNOWN) {
        PyErr_SetString(PyExc_ValueError, "Invalid mask values");
        return -1;
    }

    surface = PG_CreateSurface(width, height, pxformat);
    if (!surface) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return -1;
    }

    if (!(flags & PGS_SRCALPHA)) {
        /* We ignore the error if any. */
        SDL_SetSurfaceBlendMode(surface, SDL_BLENDMODE_NONE);

        /* When the display format has a full alpha channel (macOS right now),
         * Surfaces may be created with an unreqested alpha channel, which
         * could cause issues.
         * pygame Surfaces are supposed to be (0, 0, 0, 255) by default.
         * This is a simple fix to fill it with (0, 0, 0, 255) if necessary.
         * See Github issue:
         * https://github.com/pygame/pygame/issues/796
         */
        if (Amask != 0) {
            SDL_FillRect(surface, NULL,
                         SDL_MapRGBA(surface->format, 0, 0, 0, 255));
        }
    }

    if (SDL_ISPIXELFORMAT_INDEXED(PG_SURF_FORMATENUM(surface))) {
        /* Give the surface something other than an all white palette.
         *          */
        if (!PG_SetPaletteColors(surface->format->palette,
                                 default_palette_colors, 0,
                                 default_palette_size - 1)) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(surface);
            return -1;
        }
    }
#endif

    if (surface) {
        self->surf = surface;
        self->owner = 1;
        self->subsurface = NULL;
    }

    return 0;
}

/* surface object methods */
static PyObject *
surf_get_at(PyObject *self, PyObject *position)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    Uint8 *pixels = NULL;
    int x, y;
    Uint32 color;
    Uint8 *pix;
    Uint8 rgba[4] = {0, 0, 0, 255};

    SURF_INIT_CHECK(surf)

    if (!pg_TwoIntsFromObj(position, &x, &y)) {
        return RAISE(PyExc_TypeError,
                     "position must be a sequence of two numbers");
    }

    if (x < 0 || x >= surf->w || y < 0 || y >= surf->h) {
        return RAISE(PyExc_IndexError, "pixel index out of range");
    }

    PG_PixelFormat *format;
    SDL_Palette *palette;
    if (!PG_GetSurfaceDetails(surf, &format, &palette)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    int bpp = PG_FORMAT_BytesPerPixel(format);

    if (bpp < 1 || bpp > 4) {
        return RAISE(PyExc_RuntimeError, "invalid color depth for surface");
    }

    if (!pgSurface_Lock((pgSurfaceObject *)self)) {
        return NULL;
    }

    pixels = (Uint8 *)surf->pixels;

    switch (bpp) {
        case 1:
            color = (Uint32) * ((Uint8 *)pixels + y * surf->pitch + x);
            PG_GetRGB(color, format, palette, rgba, rgba + 1, rgba + 2);
            break;
        case 2:
            color = (Uint32) * ((Uint16 *)(pixels + y * surf->pitch) + x);
            PG_GetRGBA(color, format, palette, rgba, rgba + 1, rgba + 2,
                       rgba + 3);
            break;
        case 3:
            pix = ((Uint8 *)(pixels + y * surf->pitch) + x * 3);
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
            color = (pix[0]) + (pix[1] << 8) + (pix[2] << 16);
#else
            color = (pix[2]) + (pix[1] << 8) + (pix[0] << 16);
#endif
            PG_GetRGB(color, format, palette, rgba, rgba + 1, rgba + 2);
            break;
        default: /* case 4: */
            assert(PG_FORMAT_BytesPerPixel(format) == 4);
            color = *((Uint32 *)(pixels + y * surf->pitch) + x);
            PG_GetRGBA(color, format, palette, rgba, rgba + 1, rgba + 2,
                       rgba + 3);
            break;
    }
    if (!pgSurface_Unlock((pgSurfaceObject *)self)) {
        return NULL;
    }

    return pgColor_New(rgba);
}

static PyObject *
surf_set_at(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SURF_INIT_CHECK(surf)
    Uint8 *pixels;
    int x, y;
    Uint32 color;
    PyObject *rgba_obj;
    Uint8 *byte_buf;

    if (nargs != 2) {
        return PyErr_Format(PyExc_TypeError,
                            "set_at takes exactly 2 arguments (%zd given)",
                            nargs);
    }

    if (!pg_TwoIntsFromObj(args[0], &x, &y)) {
        return RAISE(PyExc_TypeError,
                     "position must be a sequence of two numbers");
    }

    rgba_obj = args[1];

    PG_PixelFormat *format = PG_GetSurfaceFormat(surf);
    if (format == NULL) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    int bpp = PG_FORMAT_BytesPerPixel(format);

    if (bpp < 1 || bpp > 4) {
        return RAISE(PyExc_RuntimeError, "invalid color depth for surface");
    }

    SDL_Rect clip_rect;
    if (!PG_GetSurfaceClipRect(surf, &clip_rect)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    if (x < clip_rect.x || x >= clip_rect.x + clip_rect.w || y < clip_rect.y ||
        y >= clip_rect.y + clip_rect.h) {
        /* out of clip area */
        Py_RETURN_NONE;
    }

    if (!pg_MappedColorFromObj(rgba_obj, surf, &color, PG_COLOR_HANDLE_ALL)) {
        return NULL;
    }

    if (!pgSurface_Lock((pgSurfaceObject *)self)) {
        return NULL;
    }
    pixels = (Uint8 *)surf->pixels;

    switch (bpp) {
        case 1:
            *((Uint8 *)pixels + y * surf->pitch + x) = (Uint8)color;
            break;
        case 2:
            *((Uint16 *)(pixels + y * surf->pitch) + x) = (Uint16)color;
            break;
        case 3:
            byte_buf = (Uint8 *)(pixels + y * surf->pitch) + x * 3;
            // Shouldn't this be able to happen without awareness of shifts?
            // mapped color -> pixel and all.
#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
            *(byte_buf + (format->Rshift >> 3)) =
                (Uint8)(color >> format->Rshift);
            *(byte_buf + (format->Gshift >> 3)) =
                (Uint8)(color >> format->Gshift);
            *(byte_buf + (format->Bshift >> 3)) =
                (Uint8)(color >> format->Bshift);
#else
            *(byte_buf + 2 - (format->Rshift >> 3)) =
                (Uint8)(color >> format->Rshift);
            *(byte_buf + 2 - (format->Gshift >> 3)) =
                (Uint8)(color >> format->Gshift);
            *(byte_buf + 2 - (format->Bshift >> 3)) =
                (Uint8)(color >> format->Bshift);
#endif
            break;
        default: /* case 4: */
            *((Uint32 *)(pixels + y * surf->pitch) + x) = color;
            break;
    }

    if (!pgSurface_Unlock((pgSurfaceObject *)self)) {
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *
surf_get_at_mapped(PyObject *self, PyObject *position)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    Uint8 *pixels = NULL;
    int x, y;
    Sint32 color;
    Uint8 *pix;

    SURF_INIT_CHECK(surf)

    if (!pg_TwoIntsFromObj(position, &x, &y)) {
        return RAISE(PyExc_TypeError,
                     "position must be a sequence of two numbers");
    }

    if (x < 0 || x >= surf->w || y < 0 || y >= surf->h) {
        return RAISE(PyExc_IndexError, "pixel index out of range");
    }

    int bpp = PG_SURF_BytesPerPixel(surf);

    if (bpp < 1 || bpp > 4) {
        return RAISE(PyExc_RuntimeError, "invalid color depth for surface");
    }

    if (!pgSurface_Lock((pgSurfaceObject *)self)) {
        return NULL;
    }

    pixels = (Uint8 *)surf->pixels;

    switch (bpp) {
        case 1:
            color = (Uint32) * ((Uint8 *)pixels + y * surf->pitch + x);
            break;
        case 2:
            color = (Uint32) * ((Uint16 *)(pixels + y * surf->pitch) + x);
            break;
        case 3:
            pix = ((Uint8 *)(pixels + y * surf->pitch) + x * 3);
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
            color = (pix[0]) + (pix[1] << 8) + (pix[2] << 16);
#else
            color = (pix[2]) + (pix[1] << 8) + (pix[0] << 16);
#endif
            break;
        default: /* case 4: */
            color = *((Uint32 *)(pixels + y * surf->pitch) + x);
            break;
    }
    if (!pgSurface_Unlock((pgSurfaceObject *)self)) {
        return NULL;
    }

    return PyLong_FromLong((long)color);
}

static PyObject *
surf_map_rgb(PyObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    Uint8 rgba[4];
    int color;

    if (!pg_RGBAFromObjEx(args, rgba, PG_COLOR_HANDLE_ALL)) {
        return NULL; /* Exception already set for us */
    }

    SURF_INIT_CHECK(surf)

    PG_PixelFormat *format;
    SDL_Palette *palette;
    if (!PG_GetSurfaceDetails(surf, &format, &palette)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    color = PG_MapRGBA(format, palette, rgba[0], rgba[1], rgba[2], rgba[3]);
    return PyLong_FromLong(color);
}

static PyObject *
surf_unmap_rgb(PyObject *self, PyObject *arg)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    Uint32 col;
    Uint8 rgba[4];

    col = (Uint32)PyLong_AsLong(arg);
    if (col == (Uint32)-1 && PyErr_Occurred()) {
        PyErr_Clear();
        return RAISE(PyExc_TypeError, "unmap_rgb expects 1 number argument");
    }
    SURF_INIT_CHECK(surf)

    PG_PixelFormat *format;
    SDL_Palette *palette;
    if (!PG_GetSurfaceDetails(surf, &format, &palette)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    if (SDL_ISPIXELFORMAT_ALPHA(format->format)) {
        PG_GetRGBA(col, format, palette, rgba, rgba + 1, rgba + 2, rgba + 3);
    }
    else {
        PG_GetRGB(col, format, palette, rgba, rgba + 1, rgba + 2);
        rgba[3] = 255;
    }

    return pgColor_New(rgba);
}

static PyObject *
surf_lock(PyObject *self, PyObject *_null)
{
    SURF_INIT_CHECK(pgSurface_AsSurface(self))
    if (!pgSurface_Lock((pgSurfaceObject *)self)) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
surf_unlock(PyObject *self, PyObject *_null)
{
    SURF_INIT_CHECK(pgSurface_AsSurface(self))
    pgSurface_Unlock((pgSurfaceObject *)self);
    Py_RETURN_NONE;
}

static PyObject *
surf_mustlock(PyObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SURF_INIT_CHECK(surf)
    return PyBool_FromLong(SDL_MUSTLOCK(surf) ||
                           ((pgSurfaceObject *)self)->subsurface);
}

static PyObject *
surf_get_locked(PyObject *self, PyObject *_null)
{
    pgSurfaceObject *surf = (pgSurfaceObject *)self;

    SURF_INIT_CHECK(surf)

    if (surf->locklist && PyList_Size(surf->locklist) > 0) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *
surf_get_locks(PyObject *self, PyObject *_null)
{
    pgSurfaceObject *surf = (pgSurfaceObject *)self;
    Py_ssize_t len, i = 0;
    int weakref_getref_result;
    PyObject *tuple, *tmp;
    SURF_INIT_CHECK(pgSurface_AsSurface(self))
    if (!surf->locklist) {
        return PyTuple_New(0);
    }

    len = PyList_Size(surf->locklist);
    tuple = PyTuple_New(len);
    if (!tuple) {
        return NULL;
    }

    for (i = 0; i < len; i++) {
        weakref_getref_result =
            PyWeakref_GetRef(PyList_GetItem(surf->locklist, i), &tmp);
        if (weakref_getref_result == -1) {  // exception already set
            Py_DECREF(tuple);
            return NULL;
        }
        if (weakref_getref_result == 0) {
            tmp = Py_None;
            Py_INCREF(tmp);
        }
        PyTuple_SetItem(tuple, i, tmp);
    }
    return tuple;
}

static PyObject *
surf_get_palette(PyObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SDL_Palette *pal = NULL;
    PyObject *list;
    int i;
    PyObject *color;
    SDL_Color *c;
    Uint8 rgba[4] = {0, 0, 0, 255};

    SURF_INIT_CHECK(surf)

    pal = PG_GetSurfacePalette(surf);

    if (!pal) {
        return RAISE(pgExc_SDLError, "Surface has no palette to get\n");
    }

    list = PyTuple_New(pal->ncolors);
    if (!list) {
        return NULL;
    }

    for (i = 0; i < pal->ncolors; i++) {
        c = &pal->colors[i];
        rgba[0] = c->r;
        rgba[1] = c->g;
        rgba[2] = c->b;
        color = pgColor_NewLength(rgba, 3);

        if (!color) {
            Py_DECREF(list);
            return NULL;
        }
        PyTuple_SET_ITEM(list, i, color);
    }

    return list;
}

static PyObject *
surf_get_palette_at(PyObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SDL_Palette *pal = NULL;
    SDL_Color *c;
    int _index;
    Uint8 rgba[4];

    if (!PyArg_ParseTuple(args, "i", &_index)) {
        return NULL;
    }
    SURF_INIT_CHECK(surf)

    pal = PG_GetSurfacePalette(surf);

    if (!pal) {
        return RAISE(pgExc_SDLError, "Surface has no palette to set\n");
    }
    if (_index >= pal->ncolors || _index < 0) {
        return RAISE(PyExc_IndexError, "index out of bounds");
    }

    c = &pal->colors[_index];
    rgba[0] = c->r;
    rgba[1] = c->g;
    rgba[2] = c->b;
    rgba[3] = 255;

    return pgColor_NewLength(rgba, 3);
}

static PyObject *
surf_set_palette(PyObject *self, PyObject *seq)
{
    /* This method works differently from the SDL 1.2 equivalent.
     * It replaces colors in the surface's existing palette. So, if the
     * color list is shorter than the existing palette, only the first
     * part of the palette is replaced. For the SDL 1.2 Pygame version,
     * the actual colors array is replaced, making it shorter.
     */
    const SDL_Color *old_colors;
    SDL_Color colors[256];
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SDL_Palette *pal = NULL;
    PyObject *item;
    int i, len;
    Uint8 rgba[4];
    int ecode;

    SURF_INIT_CHECK(surf)

    if (!PySequence_Check(seq)) {
        return RAISE(PyExc_ValueError, "Argument must be a sequence type");
    }

    pal = PG_GetSurfacePalette(surf);

    if (!SDL_ISPIXELFORMAT_INDEXED(PG_SURF_FORMATENUM(surf))) {
        return RAISE(pgExc_SDLError, "Surface colors are not indexed\n");
    }

    if (!pal) {
        return RAISE(pgExc_SDLError, "Surface is not palettized\n");
    }
    old_colors = pal->colors;

    len = (int)MIN(pal->ncolors, PySequence_Length(seq));

    for (i = 0; i < len; i++) {
        item = PySequence_GetItem(seq, i);

        ecode = pg_RGBAFromObjEx(item, rgba, PG_COLOR_HANDLE_SIMPLE);
        Py_DECREF(item);
        if (!ecode) {
            return RAISE(PyExc_ValueError,
                         "takes a sequence of integers of RGB");
        }
        if (rgba[3] != 255) {
            return RAISE(PyExc_ValueError, "takes an alpha value of 255");
        }
        colors[i].r = (unsigned char)rgba[0];
        colors[i].g = (unsigned char)rgba[1];
        colors[i].b = (unsigned char)rgba[2];
        /* Preserve palette alphas. Normally, a palette entry has alpha 255.
         * If, however, colorkey is set, the corresponding palette entry has
         * 0 alpha.
         */
        colors[i].a = (unsigned char)old_colors[i].a;
    }

    if (!PG_SetPaletteColors(pal, colors, 0, len)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    Py_RETURN_NONE;
}

static PyObject *
surf_set_palette_at(PyObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SDL_Palette *pal = NULL;
    SDL_Color color;
    int _index;
    PyObject *color_obj;
    Uint8 rgba[4];

    if (!PyArg_ParseTuple(args, "iO", &_index, &color_obj)) {
        return NULL;
    }
    SURF_INIT_CHECK(surf)

    if (!pg_RGBAFromObjEx(color_obj, rgba, PG_COLOR_HANDLE_SIMPLE)) {
        return NULL;
    }

    if (!SDL_ISPIXELFORMAT_INDEXED(PG_SURF_FORMATENUM(surf))) {
        return RAISE(pgExc_SDLError, "Surface colors are not indexed\n");
    }

    pal = PG_GetSurfacePalette(surf);

    if (!pal) {
        return RAISE(pgExc_SDLError, "Surface is not palettized\n");
    }

    if (_index >= pal->ncolors || _index < 0) {
        return RAISE(PyExc_IndexError, "index out of bounds");
    }

    color.r = rgba[0];
    color.g = rgba[1];
    color.b = rgba[2];
    color.a = pal->colors[_index].a; /* May be a colorkey color. */

    if (!PG_SetPaletteColors(pal, &color, _index, 1)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    Py_RETURN_NONE;
}

static PyObject *
surf_set_colorkey(pgSurfaceObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    Uint32 flags = 0, color = 0;
    PyObject *rgba_obj = NULL;
    int hascolor = SDL_FALSE;

    if (!PyArg_ParseTuple(args, "|Oi", &rgba_obj, &flags)) {
        return NULL;
    }

    SURF_INIT_CHECK(surf)

    if (rgba_obj && rgba_obj != Py_None) {
        if (!pg_MappedColorFromObj(rgba_obj, surf, &color,
                                   PG_COLOR_HANDLE_ALL)) {
            return NULL;
        }

        hascolor = SDL_TRUE;
    }

    pgSurface_Prep(self);
    bool success = true;
    if (hascolor && PG_SURF_BytesPerPixel(surf) == 1) {
        /* For an indexed surface, remove the previous colorkey first.
         */
        success = PG_SetSurfaceColorKey(surf, SDL_FALSE, color);
    }
    if (success && hascolor) {
        success = PG_SetSurfaceRLE(
            surf, (flags & PGS_RLEACCEL) ? SDL_TRUE : SDL_FALSE);
    }
    if (success) {
        success = PG_SetSurfaceColorKey(surf, hascolor, color);
    }
    pgSurface_Unprep(self);

    if (!success) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    Py_RETURN_NONE;
}

static PyObject *
surf_get_colorkey(pgSurfaceObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    Uint32 mapped_color;
    Uint8 r, g, b, a = 255;

    SURF_INIT_CHECK(surf)

    if (!SDL_HasColorKey(surf)) {
        Py_RETURN_NONE;
    }

    SDL_GetColorKey(surf, &mapped_color);

    PG_PixelFormat *format;
    SDL_Palette *palette;
    if (!PG_GetSurfaceDetails(surf, &format, &palette)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    if (SDL_ISPIXELFORMAT_ALPHA(format->format)) {
        PG_GetRGBA(mapped_color, format, palette, &r, &g, &b, &a);
    }
    else {
        PG_GetRGB(mapped_color, format, palette, &r, &g, &b);
    }

    return Py_BuildValue("(bbbb)", r, g, b, a);
}

static PyObject *
surf_set_alpha(pgSurfaceObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    Uint32 flags = 0;
    PyObject *alpha_obj = NULL, *intobj = NULL;
    Uint8 alpha;
    int alphaval = 255;

    if (!PyArg_ParseTuple(args, "|Oi", &alpha_obj, &flags)) {
        return NULL;
    }
    SURF_INIT_CHECK(surf)

    if (alpha_obj && alpha_obj != Py_None) {
        if (PyNumber_Check(alpha_obj) && (intobj = PyNumber_Long(alpha_obj))) {
            alphaval = (int)PyLong_AsLong(intobj);
            Py_DECREF(intobj);
            if (alphaval == -1 && PyErr_Occurred()) {
                return RAISE(PyExc_TypeError, "invalid alpha argument");
            }
        }
        else {
            return RAISE(PyExc_TypeError, "invalid alpha argument");
        }

        if (!PG_SetSurfaceBlendMode(surf, SDL_BLENDMODE_BLEND)) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
    }
    else {
        if (!PG_SetSurfaceBlendMode(surf, SDL_BLENDMODE_NONE)) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
    }

    if (alphaval > 255) {
        alpha = 255;
    }
    else if (alphaval < 0) {
        alpha = 0;
    }
    else {
        alpha = (Uint8)alphaval;
    }

    if (alpha == 255 && (PG_SURF_BytesPerPixel(surf) == 1)) {
        /* Can't blend with a surface alpha of 255 and 8bit surfaces */
        if (!PG_SetSurfaceBlendMode(surf, SDL_BLENDMODE_NONE)) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
    }
    pgSurface_Prep(self);
    bool success =
        PG_SetSurfaceRLE(surf, (flags & PGS_RLEACCEL) ? SDL_TRUE : SDL_FALSE);
    /* HACK HACK HACK */
    if (SDL_MUSTLOCK(surf) && (!(flags & PGS_RLEACCEL))) {
        /* hack to strip SDL_RLEACCEL flag off surface immediately when
           it is not requested */
        SDL_Rect sdlrect;
        sdlrect.x = 0;
        sdlrect.y = 0;
        sdlrect.h = 1;
        sdlrect.w = 1;

        SDL_Surface *surface =
            PG_CreateSurface(1, 1, PG_SURF_FORMATENUM(surf));

        SDL_LowerBlit(surf, &sdlrect, surface, &sdlrect);
        SDL_FreeSurface(surface);
    }
    /* HACK HACK HACK */
    if (success) {
        success = PG_SetSurfaceAlphaMod(surf, alpha);
    }
    pgSurface_Unprep(self);

    if (!success) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    Py_RETURN_NONE;
}

static PyObject *
surf_get_alpha(pgSurfaceObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SDL_BlendMode mode;
    Uint8 alpha;

    SURF_INIT_CHECK(surf)

    if (!PG_GetSurfaceBlendMode(surf, &mode)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    if (mode != SDL_BLENDMODE_BLEND) {
        Py_RETURN_NONE;
    }

    if (!PG_GetSurfaceAlphaMod(surf, &alpha)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    return PyLong_FromLong(alpha);
}

static PyObject *
surf_get_blendmode(PyObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SDL_BlendMode mode;

    SURF_INIT_CHECK(surf)

    if (!PG_GetSurfaceBlendMode(surf, &mode)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    return PyLong_FromLong((long)mode);
}

static PyObject *
surf_copy(pgSurfaceObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    PyObject *final;
    SDL_Surface *newsurf;

    SURF_INIT_CHECK(surf)

    pgSurface_Prep(self);
    newsurf = PG_ConvertSurface(surf, surf->format);
    pgSurface_Unprep(self);

    final = surf_subtype_new(Py_TYPE(self), newsurf, 1);
    if (!final) {
        SDL_FreeSurface(newsurf);
    }
    return final;
}

static PyObject *
surf_convert(pgSurfaceObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    PyObject *final;
    PyObject *argobject = NULL;
    SDL_Surface *src;
    SDL_Surface *newsurf;
    Uint32 flags = UINT32_MAX;

    Uint32 colorkey;
    Uint8 key_r, key_g, key_b, key_a = 255;
    int has_colorkey = SDL_FALSE;

    if (!PyArg_ParseTuple(args, "|Oi", &argobject, &flags)) {
        return NULL;
    }

    if (!argobject && !SDL_WasInit(SDL_INIT_VIDEO)) {
        return RAISE(pgExc_SDLError,
                     "cannot convert without format "
                     "when pygame.display is not initialized");
    }

    SURF_INIT_CHECK(surf)

    pgSurface_Prep(self);

#if SDL_VERSION_ATLEAST(3, 0, 0)
    if ((has_colorkey = SDL_HasColorKey(surf))) {
        PG_PixelFormat *surf_format;
        SDL_Palette *surf_palette;
        if (!PG_GetSurfaceDetails(surf, &surf_format, &surf_palette)) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }

        SDL_GetColorKey(surf, &colorkey);
        if (SDL_ISPIXELFORMAT_ALPHA(PG_SURF_FORMATENUM(surf))) {
            PG_GetRGBA(colorkey, surf_format, surf_palette, &key_r, &key_g,
                       &key_b, &key_a);
        }
        else {
            PG_GetRGB(colorkey, surf_format, surf_palette, &key_r, &key_g,
                      &key_b);
        }
    }

    if (argobject) {
        if (pgSurface_Check(argobject)) {
            src = pgSurface_AsSurface(argobject);
            newsurf = PG_ConvertSurface(surf, src->format);
        }
        else {
            /* will be updated later, initialize to make static analyzer happy
             */
            int bpp = 0;
            SDL_Palette *palette = NULL;
            PG_PixelFormatEnum format_enum = SDL_PIXELFORMAT_UNKNOWN;

            // PATH 1 = from bpp
            if (pg_IntFromObj(argobject, &bpp)) {
                if (flags != UINT32_MAX && flags & PGS_SRCALPHA) {
                    switch (bpp) {
                        case 16:
                            format_enum = SDL_PIXELFORMAT_ARGB4444;
                            break;
                        case 32:
                            format_enum = SDL_PIXELFORMAT_ARGB8888;
                            break;
                        default:
                            return RAISE(PyExc_ValueError,
                                         "no standard masks exist for given "
                                         "bitdepth with alpha");
                    }
                }
                else {
                    switch (bpp) {
                        case 8:
                            format_enum = SDL_PIXELFORMAT_INDEX8;
                            break;
                        case 12:
                            format_enum = SDL_PIXELFORMAT_XRGB4444;
                            break;
                        case 15:
                            format_enum = SDL_PIXELFORMAT_XRGB1555;
                            break;
                        case 16:
                            format_enum = SDL_PIXELFORMAT_RGB565;
                            break;
                        case 24:
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
                            format_enum = SDL_PIXELFORMAT_RGB24;
#else
                            format_enum = SDL_PIXELFORMAT_BGR24;
#endif
                            break;
                        case 32:
                            format_enum = SDL_PIXELFORMAT_XRGB8888;
                            break;
                        default:
                            return RAISE(PyExc_ValueError,
                                         "nonstandard bit depth given");
                    }
                }
            }
            // PATH 2 = from masks only
            else if (PySequence_Check(argobject) &&
                     PySequence_Size(argobject) == 4) {
                Uint32 Rmask, Gmask, Bmask, Amask;

                if (!pg_UintFromObjIndex(argobject, 0, &Rmask) ||
                    !pg_UintFromObjIndex(argobject, 1, &Gmask) ||
                    !pg_UintFromObjIndex(argobject, 2, &Bmask) ||
                    !pg_UintFromObjIndex(argobject, 3, &Amask)) {
                    pgSurface_Unprep(self);
                    return RAISE(PyExc_ValueError,
                                 "invalid color masks given");
                }
                Uint32 mask = Rmask | Gmask | Bmask | Amask;

                // This code shocked me. -Starbuck, Mar. 2025
                // Like what if you have a hole in the mask?
                // Like a blank alpha mask first-- it would just terminate
                // the whole loop right?
                // I think this whole code path should be deprecated.
                for (bpp = 0; bpp < 32; ++bpp) {
                    if (!(mask >> bpp)) {
                        break;
                    }
                }

                format_enum = SDL_MasksToPixelFormatEnum(bpp, Rmask, Gmask,
                                                         Bmask, Amask);
            }
            else {
                pgSurface_Unprep(self);
                return RAISE(
                    PyExc_ValueError,
                    "invalid argument specifying new format to convert to");
            }

            // If the destination format is indexed, provide a new palette or
            // copy over existing palette.
            if (SDL_ISPIXELFORMAT_INDEXED(format_enum)) {
                if (SDL_ISPIXELFORMAT_INDEXED(PG_SURF_FORMATENUM(surf))) {
                    palette = PG_GetSurfacePalette(surf);
                }
                else {
                    /* Give the surface something other than an all white
                     * palette.
                     */
                    palette = SDL_AllocPalette(default_palette_size);
                    SDL_SetPaletteColors(palette, default_palette_colors, 0,
                                         default_palette_size);
                }
            }

            newsurf = SDL_ConvertSurfaceAndColorspace(
                surf, format_enum, palette, SDL_GetSurfaceColorspace(surf), 0);

            // In this scenario, we allocated the palette, so we also need
            // to remove our reference to it.
            if (SDL_ISPIXELFORMAT_INDEXED(format_enum) &&
                !SDL_ISPIXELFORMAT_INDEXED(PG_SURF_FORMATENUM(surf))) {
                SDL_FreePalette(palette);
            }

            SDL_SetSurfaceBlendMode(newsurf, SDL_BLENDMODE_NONE);
        }
    }
    else {
        newsurf = pg_DisplayFormat(surf);
        if (newsurf) {
            SDL_SetSurfaceBlendMode(newsurf, SDL_BLENDMODE_NONE);
        }
    }

    if (newsurf == NULL) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    if (has_colorkey) {
        colorkey = SDL_MapSurfaceRGBA(newsurf, key_r, key_g, key_b, key_a);
        if (!PG_SetSurfaceColorKey(newsurf, SDL_TRUE, colorkey)) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(newsurf);
            return NULL;
        }
    }
#else
    if ((has_colorkey = SDL_HasColorKey(surf))) {
        SDL_GetColorKey(surf, &colorkey);
        if (SDL_ISPIXELFORMAT_ALPHA(PG_SURF_FORMATENUM(surf))) {
            SDL_GetRGBA(colorkey, surf->format, &key_r, &key_g, &key_b,
                        &key_a);
        }
        else {
            SDL_GetRGB(colorkey, surf->format, &key_r, &key_g, &key_b);
        }
    }

    if (argobject) {
        if (pgSurface_Check(argobject)) {
            src = pgSurface_AsSurface(argobject);
            newsurf = PG_ConvertSurface(surf, src->format);
        }
        else {
            /* will be updated later, initialize to make static analyzer happy
             */
            int bpp = 0;
            SDL_Palette *palette = SDL_AllocPalette(default_palette_size);
            Uint32 format_enum = 0;

            if (pg_IntFromObj(argobject, &bpp)) {
                Uint32 Rmask, Gmask, Bmask, Amask;

                if (flags != UINT32_MAX && flags & PGS_SRCALPHA) {
                    switch (bpp) {
                        case 16:
                            Rmask = 0xF << 8;
                            Gmask = 0xF << 4;
                            Bmask = 0xF;
                            Amask = 0xF << 12;
                            break;
                        case 32:
                            Rmask = 0xFF << 16;
                            Gmask = 0xFF << 8;
                            Bmask = 0xFF;
                            Amask = 0xFF << 24;
                            break;
                        default:
                            return RAISE(PyExc_ValueError,
                                         "no standard masks exist for given "
                                         "bitdepth with alpha");
                    }
                }
                else {
                    Amask = 0;
                    switch (bpp) {
                        case 8:
                            Rmask = 0;
                            Gmask = 0;
                            Bmask = 0;
                            break;
                        case 12:
                            Rmask = 0xFF >> 4 << 8;
                            Gmask = 0xFF >> 4 << 4;
                            Bmask = 0xFF >> 4;
                            break;
                        case 15:
                            Rmask = 0xFF >> 3 << 10;
                            Gmask = 0xFF >> 3 << 5;
                            Bmask = 0xFF >> 3;
                            break;
                        case 16:
                            Rmask = 0xFF >> 3 << 11;
                            Gmask = 0xFF >> 2 << 5;
                            Bmask = 0xFF >> 3;
                            break;
                        case 24:
                        case 32:
                            Rmask = 0xFF << 16;
                            Gmask = 0xFF << 8;
                            Bmask = 0xFF;
                            break;
                        default:
                            return RAISE(PyExc_ValueError,
                                         "nonstandard bit depth given");
                    }
                }
                format_enum = SDL_MasksToPixelFormatEnum(bpp, Rmask, Gmask,
                                                         Bmask, Amask);
            }
            else if (PySequence_Check(argobject) &&
                     PySequence_Size(argobject) == 4) {
                Uint32 Rmask, Gmask, Bmask, Amask;

                if (!pg_UintFromObjIndex(argobject, 0, &Rmask) ||
                    !pg_UintFromObjIndex(argobject, 1, &Gmask) ||
                    !pg_UintFromObjIndex(argobject, 2, &Bmask) ||
                    !pg_UintFromObjIndex(argobject, 3, &Amask)) {
                    pgSurface_Unprep(self);
                    return RAISE(PyExc_ValueError,
                                 "invalid color masks given");
                }

                const Uint32 mask = Rmask | Gmask | Bmask | Amask;
                for (bpp = 0; bpp < 32; ++bpp) {
                    if (!(mask >> bpp)) {
                        break;
                    }
                }
                format_enum = SDL_MasksToPixelFormatEnum(bpp, Rmask, Gmask,
                                                         Bmask, Amask);
            }
            else {
                pgSurface_Unprep(self);
                return RAISE(
                    PyExc_ValueError,
                    "invalid argument specifying new format to convert to");
            }
            SDL_PixelFormat *format = SDL_AllocFormat(format_enum);
            if (!format) {
                SDL_FreePalette(palette);
                pgSurface_Unprep(self);
                return RAISE(pgExc_SDLError, SDL_GetError());
            }

            if (SDL_ISPIXELFORMAT_INDEXED(format_enum)) {
                if (SDL_ISPIXELFORMAT_INDEXED(PG_SURF_FORMATENUM(surf))) {
                    SDL_SetPixelFormatPalette(format, surf->format->palette);
                }
                else {
                    /* Give the surface something other than an all white
                     * palette.
                     */
                    SDL_SetPaletteColors(palette, default_palette_colors, 0,
                                         default_palette_size);
                    SDL_SetPixelFormatPalette(format, palette);
                }
            }
            newsurf = PG_ConvertSurface(surf, format);
            SDL_SetSurfaceBlendMode(newsurf, SDL_BLENDMODE_NONE);
            SDL_FreeFormat(format);
            SDL_FreePalette(palette);
        }
    }
    else {
        newsurf = pg_DisplayFormat(surf);
        if (newsurf) {
            SDL_SetSurfaceBlendMode(newsurf, SDL_BLENDMODE_NONE);
        }
    }

    if (newsurf == NULL) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    if (has_colorkey) {
        colorkey = SDL_MapRGBA(newsurf->format, key_r, key_g, key_b, key_a);
        if (!PG_SetSurfaceColorKey(newsurf, SDL_TRUE, colorkey)) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(newsurf);
            return NULL;
        }
    }
#endif

    pgSurface_Unprep(self);

    final = surf_subtype_new(Py_TYPE(self), newsurf, 1);
    if (!final) {
        SDL_FreeSurface(newsurf);
    }
    return final;
}

static SDL_Surface *
pg_DisplayFormat(SDL_Surface *surface)
{
    PG_PixelFormatEnum default_format = pg_GetDefaultConvertFormat();
    if (!default_format) {
        SDL_SetError(
            "No convert format has been set, try display.set_mode()"
            " or Window.get_surface().");
        return NULL;
    }
    return PG_ConvertSurfaceFormat(surface, default_format);
}

static SDL_Surface *
pg_DisplayFormatAlpha(SDL_Surface *surface)
{
    PG_PixelFormatEnum pfe = SDL_PIXELFORMAT_ARGB8888;
    PG_PixelFormatEnum dformat = pg_GetDefaultConvertFormat();
    if (!dformat) {
        SDL_SetError(
            "No convert format has been set, try display.set_mode()"
            " or Window.get_surface().");
        return NULL;
    }

    switch (dformat) {
#if SDL_VERSION_ATLEAST(3, 0, 0)
        case SDL_PIXELFORMAT_XBGR1555:
#else
        case SDL_PIXELFORMAT_BGR555:
#endif
        case SDL_PIXELFORMAT_ABGR1555:
        case SDL_PIXELFORMAT_BGR565:
        case SDL_PIXELFORMAT_XBGR8888:
        case SDL_PIXELFORMAT_ABGR8888:
            pfe = SDL_PIXELFORMAT_ABGR8888;
            break;

        case SDL_PIXELFORMAT_BGRX8888:
        case SDL_PIXELFORMAT_BGRA8888:
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
        case SDL_PIXELFORMAT_BGR24:
#else
        case SDL_PIXELFORMAT_RGB24:
#endif
            pfe = SDL_PIXELFORMAT_BGRA8888;
            break;

        default:
            break;
    }
    return PG_ConvertSurfaceFormat(surface, pfe);
}

static PyObject *
surf_convert_alpha(pgSurfaceObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    PyObject *final;
    pgSurfaceObject *srcsurf = NULL;
    SDL_Surface *newsurf;

    SURF_INIT_CHECK(surf)

    if (!SDL_WasInit(SDL_INIT_VIDEO)) {
        return RAISE(pgExc_SDLError,
                     "cannot convert without pygame.display initialized");
    }

    if (!PyArg_ParseTuple(args, "|O!", &pgSurface_Type, &srcsurf)) {
        return NULL;
    }

    if (srcsurf != NULL) {
        if (PyErr_WarnEx(PyExc_DeprecationWarning,
                         "depth argument deprecated since version 2.4.0",
                         1) == -1) {
            return NULL;
        }
    }

    newsurf = pg_DisplayFormatAlpha(surf);
    if (newsurf) {
        SDL_SetSurfaceBlendMode(newsurf, SDL_BLENDMODE_BLEND);
    }
    final = surf_subtype_new(Py_TYPE(self), newsurf, 1);

    if (!final) {
        SDL_FreeSurface(newsurf);
    }
    return final;
}

static PyObject *
surf_set_clip(PyObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    PyObject *item;
    SDL_Rect *rect = NULL, temp;
    SDL_Rect sdlrect;
    int result;

    SURF_INIT_CHECK(surf)
    if (PyTuple_Size(args)) {
        item = PyTuple_GET_ITEM(args, 0);
        if (item == Py_None && PyTuple_Size(args) == 1) {
            result = SDL_SetClipRect(surf, NULL);
        }
        else {
            rect = pgRect_FromObject(args, &temp);
            if (!rect) {
                return RAISE(PyExc_ValueError, "invalid rectstyle object");
            }
            sdlrect.x = rect->x;
            sdlrect.y = rect->y;
            sdlrect.h = rect->h;
            sdlrect.w = rect->w;
            result = SDL_SetClipRect(surf, &sdlrect);
        }
    }
    else {
        result = SDL_SetClipRect(surf, NULL);
    }

    if (result == -1) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    Py_RETURN_NONE;
}

static PyObject *
surf_get_clip(PyObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);

    SURF_INIT_CHECK(surf)

    SDL_Rect clip_rect;
    if (!PG_GetSurfaceClipRect(surf, &clip_rect)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    return pgRect_New(&clip_rect);
}

static PyObject *
surf_fill(pgSurfaceObject *self, PyObject *args, PyObject *keywds)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SDL_Rect *rect, temp;
    PyObject *r = NULL;
    Uint32 color;
    int result;
    PyObject *rgba_obj;
    SDL_Rect sdlrect;
    int blendargs = 0;

    static char *kwids[] = {"color", "rect", "special_flags", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|Oi", kwids, &rgba_obj,
                                     &r, &blendargs)) {
        return NULL;
    }
    SURF_INIT_CHECK(surf)

    if (!pg_MappedColorFromObj(rgba_obj, surf, &color, PG_COLOR_HANDLE_ALL)) {
        return NULL;
    }

    if (!r || r == Py_None) {
        rect = &temp;
        temp.x = temp.y = 0;
        temp.w = surf->w;
        temp.h = surf->h;
    }
    else if (!(rect = pgRect_FromObject(r, &temp))) {
        return RAISE(PyExc_ValueError, "invalid rectstyle object");
    }

    /* we need a fresh copy so our Rect values don't get munged */
    if (rect != &temp) {
        memcpy(&temp, rect, sizeof(temp));
        rect = &temp;
    }

    // In SDL3, SDL_IntersectRect is renamed to SDL_GetRectIntersection
    SDL_Rect surfrect = {0, 0, surf->w, surf->h};
    if (!SDL_IntersectRect(rect, &surfrect, &sdlrect)) {
        sdlrect.x = 0;
        sdlrect.y = 0;
        sdlrect.w = 0;
        sdlrect.h = 0;
    }

    if (sdlrect.w <= 0 || sdlrect.h <= 0) {
        return pgRect_New(&sdlrect);
    }

    if (blendargs != 0) {
        result = surface_fill_blend(surf, &sdlrect, color, blendargs);
    }
    else {
        pgSurface_Prep(self);
        pgSurface_Lock((pgSurfaceObject *)self);
        result = PG_FillSurfaceRect(surf, &sdlrect, color) - 1;
        pgSurface_Unlock((pgSurfaceObject *)self);
        pgSurface_Unprep(self);
    }
    if (result == -1) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    return pgRect_New(&sdlrect);
}

static PyObject *
surf_blit(pgSurfaceObject *self, PyObject *args, PyObject *keywds)
{
    SDL_Surface *src, *dest = pgSurface_AsSurface(self);
    SDL_Rect *src_rect, temp;
    PyObject *argpos = NULL, *argrect = NULL;
    pgSurfaceObject *srcobject;
    int dx, dy, result;
    SDL_Rect dest_rect;
    int sx, sy;
    int blend_flags = 0;

    static char *kwids[] = {"source", "dest", "area", "special_flags", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!|OOi", kwids,
                                     &pgSurface_Type, &srcobject, &argpos,
                                     &argrect, &blend_flags)) {
        return NULL;
    }

    src = pgSurface_AsSurface(srcobject);
    SURF_INIT_CHECK(src)
    SURF_INIT_CHECK(dest)

    if (argpos == NULL) { /* dest argument is absent  */
        dx = 0;
        dy = 0;
    }
    else if ((src_rect = pgRect_FromObject(argpos, &temp))) {
        dx = src_rect->x;
        dy = src_rect->y;
    }
    else if (pg_TwoIntsFromObj(argpos, &sx, &sy)) {
        dx = sx;
        dy = sy;
    }
    else {
        return RAISE(PyExc_TypeError, "invalid destination position for blit");
    }

    if (argrect && argrect != Py_None) {
        if (!(src_rect = pgRect_FromObject(argrect, &temp))) {
            return RAISE(PyExc_TypeError, "Invalid rectstyle argument");
        }
    }
    else {
        temp.x = temp.y = 0;
        temp.w = src->w;
        temp.h = src->h;
        src_rect = &temp;
    }

    dest_rect.x = dx;
    dest_rect.y = dy;
    dest_rect.w = src_rect->w;
    dest_rect.h = src_rect->h;

    if (!blend_flags) {
        blend_flags = 0;
    }

    result =
        pgSurface_Blit(self, srcobject, &dest_rect, src_rect, blend_flags);

    if (result != 0) {
        return NULL;
    }

    return pgRect_New(&dest_rect);
}

#define BLITS_ERR_SEQUENCE_REQUIRED 1
#define BLITS_ERR_DISPLAY_SURF_QUIT 2
#define BLITS_ERR_SEQUENCE_SURF 3
#define BLITS_ERR_NO_OPENGL_SURF 4
#define BLITS_ERR_INVALID_DESTINATION 5
#define BLITS_ERR_INVALID_RECT_STYLE 6
#define BLITS_ERR_MUST_ASSIGN_NUMERIC 7
#define BLITS_ERR_BLIT_FAIL 8
#define BLITS_ERR_PY_EXCEPTION_RAISED 9
#define BLITS_ERR_SOURCE_NOT_SURFACE 10

static PyObject *
surf_blits(pgSurfaceObject *self, PyObject *args, PyObject *keywds)
{
    SDL_Surface *src, *dest = pgSurface_AsSurface(self);
    SDL_Rect *src_rect, temp;
    PyObject *srcobject = NULL, *argpos = NULL, *argrect = NULL;
    int dx, dy, result;
    SDL_Rect dest_rect;
    int sx, sy;
    int blend_flags = 0;

    PyObject *blitsequence;
    PyObject *iterator = NULL;
    PyObject *item = NULL;
    PyObject *special_flags = NULL;
    PyObject *ret = NULL;
    PyObject *retrect = NULL;
    Py_ssize_t itemlength, sequencelength, curriter = 0;
    int doreturn = 1;
    int bliterrornum = 0;
    int issequence = 0;

    static char *kwids[] = {"blit_sequence", "doreturn", NULL};

    SURF_INIT_CHECK(dest)
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|i", kwids, &blitsequence,
                                     &doreturn)) {
        return NULL;
    }

    if (!PyIter_Check(blitsequence) &&
        !(issequence = PySequence_Check(blitsequence))) {
        bliterrornum = BLITS_ERR_SEQUENCE_REQUIRED;
        goto bliterror;
    }

    if (doreturn) {
        /* If the sequence is countable, meaning not a generator, we can get
         * faster rect appending to the list by pre allocating it
         * to later call the more efficient SET_ITEM*/
        if (issequence) {
            sequencelength = PySequence_Size(blitsequence);
            if (sequencelength == -1) {
                bliterrornum = BLITS_ERR_PY_EXCEPTION_RAISED;
                goto bliterror;
            }

            ret = PyList_New(sequencelength);
        }
        else {
            ret = PyList_New(0);
        }
        if (!ret) {
            return NULL;
        }
    }

    iterator = PyObject_GetIter(blitsequence);
    if (!iterator) {
        Py_XDECREF(ret);
        return NULL;
    }

    while ((item = PyIter_Next(iterator))) {
        if (PySequence_Check(item)) {
            itemlength = PySequence_Length(item);
            if (itemlength > 4 || itemlength < 2) {
                bliterrornum = BLITS_ERR_SEQUENCE_REQUIRED;
                goto bliterror;
            }
        }
        else {
            bliterrornum = BLITS_ERR_SEQUENCE_REQUIRED;
            goto bliterror;
        }
        argrect = NULL;
        special_flags = NULL;
        blend_flags = 0;

        /* We know that there will be at least two items due to the
           conditional at the start of the loop */
        assert(itemlength >= 2);

        /* (Surface, dest) */
        srcobject = PySequence_ITEM(item, 0);
        argpos = PySequence_ITEM(item, 1);

        if (itemlength >= 3) {
            /* (Surface, dest, area) */
            argrect = PySequence_ITEM(item, 2);
        }
        if (itemlength == 4) {
            /* (Surface, dest, area, special_flags) */
            special_flags = PySequence_ITEM(item, 3);
        }
        Py_DECREF(item);
        /* Clear item to avoid double deref on errors */
        item = NULL;

        if (!pgSurface_Check(srcobject)) {
            bliterrornum = BLITS_ERR_SOURCE_NOT_SURFACE;
            goto bliterror;
        }

        src = pgSurface_AsSurface(srcobject);
        if (!dest) {
            bliterrornum = BLITS_ERR_DISPLAY_SURF_QUIT;
            goto bliterror;
        }
        if (!src) {
            bliterrornum = BLITS_ERR_SEQUENCE_SURF;
            goto bliterror;
        }

        if ((src_rect = pgRect_FromObject(argpos, &temp))) {
            dx = src_rect->x;
            dy = src_rect->y;
        }
        else if (pg_TwoIntsFromObj(argpos, &sx, &sy)) {
            dx = sx;
            dy = sy;
        }
        else {
            bliterrornum = BLITS_ERR_INVALID_DESTINATION;
            goto bliterror;
        }
        if (argrect && argrect != Py_None) {
            if (!(src_rect = pgRect_FromObject(argrect, &temp))) {
                bliterrornum = BLITS_ERR_INVALID_RECT_STYLE;
                goto bliterror;
            }
        }
        else {
            temp.x = temp.y = 0;
            temp.w = src->w;
            temp.h = src->h;
            src_rect = &temp;
        }

        dest_rect.x = dx;
        dest_rect.y = dy;
        dest_rect.w = src_rect->w;
        dest_rect.h = src_rect->h;

        if (special_flags) {
            if (!pg_IntFromObj(special_flags, &blend_flags)) {
                bliterrornum = BLITS_ERR_MUST_ASSIGN_NUMERIC;
                goto bliterror;
            }
        }

        result = pgSurface_Blit(self, (pgSurfaceObject *)srcobject, &dest_rect,
                                src_rect, blend_flags);

        if (result != 0) {
            bliterrornum = BLITS_ERR_BLIT_FAIL;
            goto bliterror;
        }

        if (doreturn) {
            retrect = NULL;
            retrect = pgRect_New(&dest_rect);

            /* If the sequence is countable, we already pre allocated a list
             * of matching size. Now we can use the efficient PyList_SET_ITEM
             * to add elements to the list */
            if (issequence) {
                PyList_SET_ITEM(ret, curriter++, retrect);
            }
            else if (PyList_Append(ret, retrect) != -1) {
                Py_DECREF(retrect);
            }
            else {
                Py_DECREF(retrect);
                retrect = NULL;
                bliterrornum = BLITS_ERR_PY_EXCEPTION_RAISED;
                goto bliterror;
            }
            retrect = NULL; /* Clear to avoid double deref on errors */
        }
        Py_DECREF(srcobject);
        Py_DECREF(argpos);
        Py_XDECREF(argrect);
        Py_XDECREF(special_flags);
        /* Clear to avoid double deref on errors */
        srcobject = NULL;
        argpos = NULL;
        argrect = NULL;
        special_flags = NULL;
    }

    Py_DECREF(iterator);
    if (PyErr_Occurred()) {
        Py_XDECREF(ret);
        return NULL;
    }

    if (doreturn) {
        return ret;
    }
    else {
        Py_RETURN_NONE;
    }

bliterror:
    Py_XDECREF(srcobject);
    Py_XDECREF(argpos);
    Py_XDECREF(argrect);
    Py_XDECREF(retrect);
    Py_XDECREF(special_flags);
    Py_XDECREF(iterator);
    Py_XDECREF(item);
    Py_XDECREF(ret);

    switch (bliterrornum) {
        case BLITS_ERR_SEQUENCE_REQUIRED:
            return RAISE(
                PyExc_ValueError,
                "blit_sequence should be iterator of (Surface, dest)");
        case BLITS_ERR_DISPLAY_SURF_QUIT:
            return RAISE(pgExc_SDLError, "display Surface quit");
        case BLITS_ERR_SEQUENCE_SURF:
            return RAISE(PyExc_TypeError,
                         "First element of blit_list needs to be Surface.");
        case BLITS_ERR_INVALID_DESTINATION:
            return RAISE(PyExc_TypeError,
                         "invalid destination position for blit");
        case BLITS_ERR_INVALID_RECT_STYLE:
            return RAISE(PyExc_TypeError, "Invalid rectstyle argument");
        case BLITS_ERR_MUST_ASSIGN_NUMERIC:
            return RAISE(PyExc_TypeError, "Must assign numeric values");
        case BLITS_ERR_BLIT_FAIL:
            return RAISE(PyExc_TypeError, "Blit failed");
        case BLITS_ERR_PY_EXCEPTION_RAISED:
            return NULL; /* Raising a previously set exception */
        case BLITS_ERR_SOURCE_NOT_SURFACE:
            return RAISE(PyExc_TypeError, "Source objects must be a surface");
    }
    return RAISE(PyExc_TypeError, "Unknown error");
}

#define FBLITS_ERR_TUPLE_REQUIRED 11
#define FBLITS_ERR_INCORRECT_ARGS_NUM 12
#define FBLITS_ERR_FLAG_NOT_NUMERIC 13

static int PG_FORCEINLINE
_surf_fblits_item_check_and_blit(pgSurfaceObject *self, PyObject *item,
                                 int blend_flags)
{
    PyObject *src_surf, *blit_pos;
    SDL_Surface *src;
    SDL_Rect *src_rect, temp, dest_rect;

    /* Check that the item is a tuple of length 2 */
    if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) != 2) {
        return FBLITS_ERR_TUPLE_REQUIRED;
    }

    /* Extract the Surface and destination objects from the
     * (Surface, dest) tuple */
    src_surf = PyTuple_GET_ITEM(item, 0);
    blit_pos = PyTuple_GET_ITEM(item, 1);

    /* Check that the source is a Surface */
    if (!pgSurface_Check(src_surf)) {
        return BLITS_ERR_SOURCE_NOT_SURFACE;
    }
    if (!(src = pgSurface_AsSurface(src_surf))) {
        return BLITS_ERR_SEQUENCE_SURF;
    }

    /* Try to extract a valid blit position */
    if (pg_TwoIntsFromObj(blit_pos, &dest_rect.x, &dest_rect.y)) {
    }
    else if ((src_rect = pgRect_FromObject(blit_pos, &temp))) {
        dest_rect.x = src_rect->x;
        dest_rect.y = src_rect->y;
    }
    else {
        return BLITS_ERR_INVALID_DESTINATION;
    }

    dest_rect.w = src->w;
    dest_rect.h = src->h;

    /* Perform the blit */
    if (pgSurface_Blit(self, (pgSurfaceObject *)src_surf, &dest_rect, NULL,
                       blend_flags)) {
        return BLITS_ERR_BLIT_FAIL;
    }

    return 0;
}

static PyObject *
surf_fblits(pgSurfaceObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    SDL_Surface *dest = pgSurface_AsSurface(self);
    SURF_INIT_CHECK(dest)

    PyObject *blit_sequence, *item;
    int blend_flags = 0; /* Default flag is 0, opaque */
    int error = 0;
    int is_generator = 0;

    if (nargs == 0 || nargs > 2) {
        error = FBLITS_ERR_INCORRECT_ARGS_NUM;
        goto on_error;
    }
    /* Get the blend flags if they are passed */
    else if (nargs == 2) {
        if (!PyLong_Check(args[1])) {
            error = FBLITS_ERR_FLAG_NOT_NUMERIC;
            goto on_error;
        }
        blend_flags = PyLong_AsLong(args[1]);
        if (PyErr_Occurred()) {
            return NULL;
        }
    }

    blit_sequence = args[0];

    /* Fast path for Lists or Tuples */
    if (pgSequenceFast_Check(blit_sequence)) {
        Py_ssize_t i;
        PyObject **sequence_items = PySequence_Fast_ITEMS(blit_sequence);
        for (i = 0; i < PySequence_Fast_GET_SIZE(blit_sequence); i++) {
            item = sequence_items[i];
            error = _surf_fblits_item_check_and_blit(self, item, blend_flags);
            if (error) {
                goto on_error;
            }
        }
    }
    /* Generator path */
    else if (PyIter_Check(blit_sequence)) {
        is_generator = 1;
        while ((item = PyIter_Next(blit_sequence))) {
            error = _surf_fblits_item_check_and_blit(self, item, blend_flags);
            if (error) {
                goto on_error;
            }
            Py_DECREF(item);
        }

        /* If the generator raises an exception */
        if (PyErr_Occurred()) {
            return NULL;
        }
    }
    else {
        error = BLITS_ERR_SEQUENCE_REQUIRED;
        goto on_error;
    }

    Py_RETURN_NONE;

on_error:
    if (is_generator) {
        Py_XDECREF(item);
    }
    switch (error) {
        case BLITS_ERR_SEQUENCE_REQUIRED:
            return RAISE(
                PyExc_ValueError,
                "blit_sequence should be iterator of (Surface, dest)");
        case BLITS_ERR_SEQUENCE_SURF:
            return RAISE(
                PyExc_TypeError,
                "First element of pairs (Surface, dest) in blit_sequence "
                "must be a Surface.");
        case BLITS_ERR_INVALID_DESTINATION:
            return RAISE(PyExc_TypeError,
                         "invalid destination position for blit");
        case BLITS_ERR_INVALID_RECT_STYLE:
            return RAISE(PyExc_TypeError, "Invalid rectstyle argument");
        case BLITS_ERR_MUST_ASSIGN_NUMERIC:
            return RAISE(PyExc_TypeError, "Must assign numeric values");
        case BLITS_ERR_BLIT_FAIL:
            return RAISE(
                PyExc_TypeError,
                "Blit failed (probably the flag used does not exist)");
        case BLITS_ERR_PY_EXCEPTION_RAISED:
            return NULL; /* Raising a previously set exception */
        case BLITS_ERR_SOURCE_NOT_SURFACE:
            return RAISE(PyExc_TypeError, "Source objects must be a Surface");
        case FBLITS_ERR_TUPLE_REQUIRED:
            return RAISE(
                PyExc_ValueError,
                "Blit_sequence item should be a tuple of (Surface, dest)");
        case FBLITS_ERR_INCORRECT_ARGS_NUM:
            return RAISE(PyExc_ValueError,
                         "Incorrect number of parameters passed: need at "
                         "least one, 2 at max");
        case FBLITS_ERR_FLAG_NOT_NUMERIC:
            return RAISE(PyExc_TypeError,
                         "The special_flags parameter must be an int");
    }
    return RAISE(PyExc_TypeError, "Unknown error");
}

static int
scroll_repeat(int h, int dx, int dy, int pitch, int span, int xoffset,
              Uint8 *startsrc, Uint8 *endsrc, Uint8 *linesrc)
{
    if (dy != 0) {
        int yincrease = dy > 0 ? -pitch : pitch;
        int spanincrease = dy > 0 ? -span : span;
        int templen = (dy > 0 ? dy * span : -dy * span);
        int tempheight = (dy > 0 ? dy : -dy);
        /* Create a temporary buffer to store the pixels that
           are disappearing from the surface */
        Uint8 *tempbuf = (Uint8 *)malloc(templen);
        if (tempbuf == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        memset(tempbuf, 0, templen);
        Uint8 *templine = tempbuf;
        Uint8 *tempend =
            templine + (dy > 0 ? (dy - 1) * span : -(dy + 1) * span);
        if (dy > 0) {
            templine = tempend;
        }
        int looph = h;
        while (looph--) {
            // If the current line should disappear copy it to the
            // temporary buffer
            if ((templine <= tempend && dy < 0) ||
                (templine >= tempbuf && dy > 0)) {
                if (dx > 0) {
                    memcpy(templine, linesrc + span - xoffset, xoffset);
                    memcpy(templine + xoffset, linesrc, span - xoffset);
                }
                else if (dx < 0) {
                    memcpy(templine + span + xoffset, linesrc, -xoffset);
                    memcpy(templine, linesrc - xoffset, span + xoffset);
                }
                else {
                    memcpy(templine, linesrc, span);
                }
                memset(linesrc, 0, span);
                templine += spanincrease;
            }
            else {
                Uint8 *pastesrc = linesrc + pitch * dy;
                if ((dy < 0 && pastesrc >= startsrc) ||
                    (dy > 0 && pastesrc <= endsrc)) {
                    if (dx > 0) {
                        memcpy(pastesrc, linesrc + span - xoffset, xoffset);
                        memcpy(pastesrc + xoffset, linesrc, span - xoffset);
                    }
                    else if (dx < 0) {
                        memcpy(pastesrc + span + xoffset, linesrc, -xoffset);
                        memcpy(pastesrc, linesrc - xoffset, span + xoffset);
                    }
                    else {
                        memcpy(pastesrc, linesrc, span);
                    }
                }
            }
            linesrc += yincrease;
        }
        // Copy the data of the buffer back to the original pixels to
        // repeat
        templine = tempbuf;
        if (dy < 0) {
            linesrc = startsrc + pitch * (h - tempheight);
        }
        else {
            linesrc = startsrc;
        }
        while (tempheight--) {
            memcpy(linesrc, templine, span);
            linesrc += pitch;
            templine += span;
        }
        free(tempbuf);
    }
    else {
        // No y-shifting, the temporary buffer should only store the x loss
        Uint8 *tempbuf = (Uint8 *)malloc((dx > 0 ? xoffset : -xoffset));
        if (tempbuf == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        while (h--) {
            if (dx > 0) {
                memcpy(tempbuf, linesrc + span - xoffset, xoffset);
                memmove(linesrc + xoffset, linesrc, span - xoffset);
                memcpy(linesrc, tempbuf, xoffset);
            }
            else if (dx < 0) {
                memcpy(tempbuf, linesrc, -xoffset);
                memmove(linesrc, linesrc - xoffset, span + xoffset);
                memcpy(linesrc + span + xoffset, tempbuf, -xoffset);
            }
            linesrc += pitch;
        }
        free(tempbuf);
    }
    return 0;
}

static int
scroll_default(int h, int dx, int dy, int pitch, int span, int xoffset,
               Uint8 *startsrc, Uint8 *endsrc, Uint8 *linesrc, int erase)
{
    if (dy != 0) {
        /* Copy the current line to a before or after position if it's
           valid with consideration of x offset and memset to avoid
           artifacts */
        int yincrease = dy > 0 ? -pitch : pitch;
        while (h--) {
            Uint8 *pastesrc = linesrc + pitch * dy;
            if ((dy < 0 && pastesrc >= startsrc) ||
                (dy > 0 && pastesrc <= endsrc)) {
                if (dx > 0) {
                    memcpy(pastesrc + xoffset, linesrc, span - xoffset);
                }
                else if (dx < 0) {
                    memcpy(pastesrc, linesrc - xoffset, span + xoffset);
                }
                else {
                    memcpy(pastesrc, linesrc, span);
                }
                if (erase) {
                    memset(linesrc, 0, span);
                    // Fix the missing pixel bug
                    if (dx < 0) {
                        memset(pastesrc + span + xoffset, 0, -xoffset);
                    }
                    else if (dx > 0) {
                        memset(pastesrc, 0, xoffset);
                    }
                }
            }
            linesrc += yincrease;
        }
    }
    else {
        // No y-shifting, we only need to move pixels on the same line
        while (h--) {
            if (dx > 0) {
                memmove(linesrc + xoffset, linesrc, span - xoffset);
                if (erase) {
                    memset(linesrc, 0, xoffset);
                }
            }
            else if (dx < 0) {
                memmove(linesrc, linesrc - xoffset, span + xoffset);
                if (erase) {
                    memset(linesrc + span + xoffset, 0, -xoffset);
                }
            }
            linesrc += pitch;
        }
    }
    return 0;
}

static int
scroll(SDL_Surface *surf, int dx, int dy, int x, int y, int w, int h,
       int repeat, int erase)
{
    int bpp = PG_SURF_BytesPerPixel(surf);
    int pitch = surf->pitch;
    int span = w * bpp;
    Uint8 *linesrc = (Uint8 *)surf->pixels + pitch * y + bpp * x;
    Uint8 *startsrc = linesrc;
    int xoffset = dx * bpp;
    Uint8 *endsrc = linesrc;
    if (dy > 0) {
        endsrc = linesrc + pitch * (h - 1);
        linesrc = endsrc;
    }

    if (repeat) {
        return scroll_repeat(h, dx, dy, pitch, span, xoffset, startsrc, endsrc,
                             linesrc);
    }
    else {
        return scroll_default(h, dx, dy, pitch, span, xoffset, startsrc,
                              endsrc, linesrc, erase);
    }
}

static PyObject *
surf_scroll(PyObject *self, PyObject *args, PyObject *keywds)
{
    int dx = 0, dy = 0, scroll_flag = PGS_SCROLL_DEFAULT;
    int erase = 0, repeat = 0;
    SDL_Surface *surf;
    SDL_Rect work_rect;
    int w = 0, h = 0, x = 0, y = 0;

    static char *kwids[] = {"dx", "dy", "scroll_flag", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|iii", kwids, &dx, &dy,
                                     &scroll_flag)) {
        return NULL;
    }

    surf = pgSurface_AsSurface(self);
    SURF_INIT_CHECK(surf)

    if (dx == 0 && dy == 0) {
        Py_RETURN_NONE;
    }

    switch (scroll_flag) {
        case PGS_SCROLL_REPEAT: {
            repeat = 1;
            break;
        }
        case PGS_SCROLL_ERASE: {
            erase = 1;
            break;
        }
        default: {
            if (scroll_flag != PGS_SCROLL_DEFAULT) {
                return RAISE(PyExc_ValueError, "Invalid scroll flag");
            }
        }
    }

    SDL_Rect clip_rect;
    if (!PG_GetSurfaceClipRect(surf, &clip_rect)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    SDL_Rect surf_rect = {0, 0, surf->w, surf->h};

    // In SDL3, SDL_IntersectRect is renamed to SDL_GetRectIntersection
    if (!SDL_IntersectRect(&clip_rect, &surf_rect, &work_rect)) {
        Py_RETURN_NONE;
    }

    w = work_rect.w;
    h = work_rect.h;
    x = work_rect.x;
    y = work_rect.y;

    /* If the clip rect is outside the surface fill and return
      for scrolls without repeat. Only fill when erase is true */
    if (!repeat) {
        if (dx >= w || dx <= -w || dy >= h || dy <= -h) {
            if (erase) {
                if (!PG_FillSurfaceRect(surf, NULL, 0)) {
                    PyErr_SetString(pgExc_SDLError, SDL_GetError());
                    return NULL;
                }
            }
            Py_RETURN_NONE;
        }
    }
    // Repeated scrolls are periodic so we can delete the exceeding value
    dx = dx % w;
    dy = dy % h;

    if (!pgSurface_Lock((pgSurfaceObject *)self)) {
        return NULL;
    }

    if (scroll(surf, dx, dy, x, y, w, h, repeat, erase) < 0) {
        pgSurface_Unlock((pgSurfaceObject *)self);
        return NULL;
    }

    if (!pgSurface_Unlock((pgSurfaceObject *)self)) {
        return NULL;
    }

    Py_RETURN_NONE;
}

static int
_PgSurface_SrcAlpha(SDL_Surface *surf)
{
    SDL_BlendMode mode;
    if (!PG_GetSurfaceBlendMode(surf, &mode)) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return -1;
    }
    return (mode != SDL_BLENDMODE_NONE);
}

static PyObject *
surf_get_flags(PyObject *self, PyObject *_null)
{
    Uint32 sdl_flags = 0;
#if SDL_VERSION_ATLEAST(3, 0, 0)
    SDL_WindowFlags window_flags = 0;
#else
    Uint32 window_flags = 0;
#endif
    Uint32 flags = 0;
    int is_alpha;
    int is_window_surf = 0;
    SDL_Window *win = pg_GetDefaultWindow();

    SDL_Surface *surf = pgSurface_AsSurface(self);

    SURF_INIT_CHECK(surf)

    if (win && pg_GetDefaultWindowSurface()) {
        if (surf == pgSurface_AsSurface(pg_GetDefaultWindowSurface())) {
            is_window_surf = 1;
            window_flags = SDL_GetWindowFlags(win);
        }
    }
    sdl_flags = surf->flags;
    if ((is_alpha = _PgSurface_SrcAlpha(surf)) == -1) {
        return NULL;
    }
    if (is_alpha) {
        flags |= PGS_SRCALPHA;
    }
    if (SDL_HasColorKey(surf)) {
        flags |= PGS_SRCCOLORKEY;
    }
    if (sdl_flags & SDL_PREALLOC) {
        flags |= PGS_PREALLOC;
    }
    /* This checks if RLE was requested on the surface */
    if (PG_SurfaceHasRLE(surf)) {
        flags |= PGS_RLEACCELOK;
    }
    /* this checks if the surface actually has RLE.
     * On SDL2: SDL_MUSTLOCK is (flags & SDL_RLEACCEL) */
    if (SDL_MUSTLOCK(surf)) {
        flags |= PGS_RLEACCEL;
    }
    if (is_window_surf) {
        if (window_flags & PG_WINDOW_FULLSCREEN_INCLUSIVE) {
            flags |= PGS_FULLSCREEN;
        }
        if (window_flags & SDL_WINDOW_OPENGL) {
            flags |= PGS_OPENGL;
        }
        if (window_flags & SDL_WINDOW_RESIZABLE) {
            flags |= PGS_RESIZABLE;
        }
        if (window_flags & SDL_WINDOW_BORDERLESS) {
            flags |= PGS_NOFRAME;
        }
    }

    return PyLong_FromLong((long)flags);
}

static PyObject *
surf_get_pitch(PyObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);

    SURF_INIT_CHECK(surf)
    return PyLong_FromLong(surf->pitch);
}

static PyObject *
surf_get_size(PyObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);

    SURF_INIT_CHECK(surf)
    return pg_tuple_couple_from_values_int(surf->w, surf->h);
}

static PyObject *
surf_get_width(PyObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);

    SURF_INIT_CHECK(surf)
    return PyLong_FromLong(surf->w);
}

static PyObject *
surf_get_height(PyObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);

    SURF_INIT_CHECK(surf)
    return PyLong_FromLong(surf->h);
}

static PyObject *
surf_get_rect(PyObject *self, PyObject *const *args, Py_ssize_t nargs,
              PyObject *kwnames)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SURF_INIT_CHECK(surf)

    PyObject *rect = pgRect_New4(0, 0, surf->w, surf->h);

    return pgObject_getRectHelper(rect, args, nargs, kwnames, "rect");
}

static PyObject *
surf_get_frect(PyObject *self, PyObject *const *args, Py_ssize_t nargs,
               PyObject *kwnames)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SURF_INIT_CHECK(surf)

    PyObject *rect = pgFRect_New4(0.f, 0.f, (float)surf->w, (float)surf->h);

    return pgObject_getRectHelper(rect, args, nargs, kwnames, "frect");
}

static PyObject *
surf_get_bitsize(PyObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SURF_INIT_CHECK(surf)

    return PyLong_FromLong(PG_SURF_BitsPerPixel(surf));
}

static PyObject *
surf_get_bytesize(PyObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SURF_INIT_CHECK(surf)

    return PyLong_FromLong(PG_SURF_BytesPerPixel(surf));
}

static PyObject *
surf_get_masks(PyObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);

    SURF_INIT_CHECK(surf)

    PG_PixelFormat *format = PG_GetSurfaceFormat(surf);
    if (format == NULL) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    return Py_BuildValue("(IIII)", format->Rmask, format->Gmask, format->Bmask,
                         format->Amask);
}

static PyObject *
surf_set_masks(PyObject *self, PyObject *args)
{
    return RAISE(PyExc_TypeError, "The surface masks are read-only in SDL2");
}

static PyObject *
surf_get_shifts(PyObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);

    SURF_INIT_CHECK(surf)

    PG_PixelFormat *format = PG_GetSurfaceFormat(surf);
    if (format == NULL) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    return Py_BuildValue("(iiii)", format->Rshift, format->Gshift,
                         format->Bshift, format->Ashift);
}

static PyObject *
surf_set_shifts(PyObject *self, PyObject *args)
{
    return RAISE(PyExc_TypeError, "The surface shifts are read-only in SDL2");
}

static PyObject *
surf_get_losses(PyObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);

    SURF_INIT_CHECK(surf)

    PG_PixelFormat *format = PG_GetSurfaceFormat(surf);
    if (format == NULL) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    return Py_BuildValue("(iiii)", PG_FORMAT_R_LOSS(format),
                         PG_FORMAT_G_LOSS(format), PG_FORMAT_B_LOSS(format),
                         PG_FORMAT_A_LOSS(format));
}

static PyObject *
surf_subsurface(PyObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SDL_Rect *rect, temp;
    SDL_Surface *sub;
    PyObject *subobj;
    struct pgSubSurface_Data *data;
    Uint8 alpha;
    Uint32 colorkey;

    SURF_INIT_CHECK(surf)

    PG_PixelFormat *format;
    SDL_Palette *palette;
    if (!PG_GetSurfaceDetails(surf, &format, &palette)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    if (!(rect = pgRect_FromObject(args, &temp))) {
        return RAISE(PyExc_ValueError, "invalid rectstyle argument");
    }
    if (rect->x < 0 || rect->y < 0 || rect->x + rect->w > surf->w ||
        rect->y + rect->h > surf->h) {
        return RAISE(PyExc_ValueError,
                     "subsurface rectangle outside surface area");
    }

    pgSurface_Lock((pgSurfaceObject *)self);

    char *startpixel = ((char *)surf->pixels) +
                       rect->x * PG_FORMAT_BytesPerPixel(format) +
                       rect->y * surf->pitch;

    sub = PG_CreateSurfaceFrom(rect->w, rect->h, format->format, startpixel,
                               surf->pitch);

    pgSurface_Unlock((pgSurfaceObject *)self);

    if (!sub) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    /* copy the colormap if we need it */
    if (SDL_ISPIXELFORMAT_INDEXED(format->format) && palette) {
        SDL_Color *colors = palette->colors;
        int ncolors = palette->ncolors;
        SDL_Palette *pal = SDL_AllocPalette(ncolors);

        if (!pal) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(sub);
            return NULL;
        }
        if (!PG_SetPaletteColors(pal, colors, 0, ncolors)) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreePalette(pal);
            SDL_FreeSurface(sub);
            return NULL;
        }
        if (!PG_SetSurfacePalette(sub, pal)) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreePalette(pal);
            SDL_FreeSurface(sub);
            return NULL;
        }
        SDL_FreePalette(pal);
    }
    if (!PG_GetSurfaceAlphaMod(surf, &alpha)) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        SDL_FreeSurface(sub);
        return NULL;
    }
    if (alpha != 255) {
        if (!PG_SetSurfaceAlphaMod(sub, alpha)) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(sub);
            return NULL;
        }
    }
    if (SDL_HasColorKey(surf)) {
        SDL_GetColorKey(surf, &colorkey);
        if (!PG_SetSurfaceColorKey(sub, SDL_TRUE, colorkey)) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(sub);
            return NULL;
        }
    }

    data = PyMem_New(struct pgSubSurface_Data, 1);
    if (!data) {
        return NULL;
    }

    subobj = surf_subtype_new(Py_TYPE(self), sub, 1);
    if (!subobj) {
        PyMem_Free(data);
        return NULL;
    }
    Py_INCREF(self);
    data->owner = self;
    data->offsetx = rect->x;
    data->offsety = rect->y;
    ((pgSurfaceObject *)subobj)->subsurface = data;

    return subobj;
}

static PyObject *
surf_get_offset(PyObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    struct pgSubSurface_Data *subdata;

    SURF_INIT_CHECK(surf)

    subdata = ((pgSurfaceObject *)self)->subsurface;
    if (!subdata) {
        return pg_tuple_couple_from_values_int(0, 0);
    }
    return pg_tuple_couple_from_values_int(subdata->offsetx, subdata->offsety);
}

static PyObject *
surf_get_abs_offset(PyObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    struct pgSubSurface_Data *subdata;
    PyObject *owner;
    int offsetx, offsety;

    SURF_INIT_CHECK(surf)

    subdata = ((pgSurfaceObject *)self)->subsurface;
    if (!subdata) {
        return pg_tuple_couple_from_values_int(0, 0);
    }

    subdata = ((pgSurfaceObject *)self)->subsurface;
    owner = subdata->owner;
    offsetx = subdata->offsetx;
    offsety = subdata->offsety;

    while (((pgSurfaceObject *)owner)->subsurface) {
        subdata = ((pgSurfaceObject *)owner)->subsurface;
        owner = subdata->owner;
        offsetx += subdata->offsetx;
        offsety += subdata->offsety;
    }
    return pg_tuple_couple_from_values_int(offsetx, offsety);
}

static PyObject *
surf_get_parent(PyObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    struct pgSubSurface_Data *subdata;

    SURF_INIT_CHECK(surf)

    subdata = ((pgSurfaceObject *)self)->subsurface;
    if (!subdata) {
        Py_RETURN_NONE;
    }

    Py_INCREF(subdata->owner);
    return subdata->owner;
}

static PyObject *
surf_get_abs_parent(PyObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    struct pgSubSurface_Data *subdata;
    PyObject *owner;

    SURF_INIT_CHECK(surf)

    subdata = ((pgSurfaceObject *)self)->subsurface;
    if (!subdata) {
        Py_INCREF(self);
        return self;
    }

    subdata = ((pgSurfaceObject *)self)->subsurface;
    owner = subdata->owner;

    while (((pgSurfaceObject *)owner)->subsurface) {
        subdata = ((pgSurfaceObject *)owner)->subsurface;
        owner = subdata->owner;
    }

    Py_INCREF(owner);
    return owner;
}

static PyObject *
surf_get_bounding_rect(PyObject *self, PyObject *args, PyObject *kwargs)
{
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
    const int BYTE0 = 0;
    const int BYTE1 = 1;
    const int BYTE2 = 2;
#else
    const int BYTE0 = 2;
    const int BYTE1 = 1;
    const int BYTE2 = 0;
#endif
    PyObject *rect;
    SDL_Surface *surf = pgSurface_AsSurface(self);
    Uint8 *pixels = NULL;
    Uint8 *pixel;
    int x, y;
    int min_x, min_y, max_x, max_y;
    int min_alpha = 1;
    int found_alpha = 0;
    Uint32 value;
    Uint8 r, g, b, a;
    int has_colorkey = 0;
    Uint32 colorkey;
    Uint8 keyr, keyg, keyb;

    char *kwids[] = {"min_alpha", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", kwids, &min_alpha)) {
        return RAISE(PyExc_ValueError,
                     "get_bounding_rect only accepts a single optional "
                     "min_alpha argument");
    }

    SURF_INIT_CHECK(surf)

    if (!pgSurface_Lock((pgSurfaceObject *)self)) {
        return RAISE(pgExc_SDLError, "could not lock surface");
    }

    PG_PixelFormat *format;
    SDL_Palette *palette;
    if (!PG_GetSurfaceDetails(surf, &format, &palette)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    int bpp = PG_FORMAT_BytesPerPixel(format);

    if ((has_colorkey = SDL_HasColorKey(surf))) {
        SDL_GetColorKey(surf, &colorkey);
        PG_GetRGBA(colorkey, format, palette, &keyr, &keyg, &keyb, &a);
    }

    pixels = (Uint8 *)surf->pixels;
    min_y = 0;
    min_x = 0;
    max_x = surf->w;
    max_y = surf->h;

    found_alpha = 0;
    for (y = max_y - 1; y >= min_y; --y) {
        for (x = min_x; x < max_x; ++x) {
            pixel = (pixels + y * surf->pitch) + x * bpp;
            switch (bpp) {
                case 1:
                    value = *pixel;
                    break;
                case 2:
                    value = *(Uint16 *)pixel;
                    break;
                case 3:
                    value = pixel[BYTE0];
                    value |= pixel[BYTE1] << 8;
                    value |= pixel[BYTE2] << 16;
                    break;
                default:
                    assert(bpp == 4);
                    value = *(Uint32 *)pixel;
            }
            PG_GetRGBA(value, format, palette, &r, &g, &b, &a);
            if ((a >= min_alpha && has_colorkey == 0) ||
                (has_colorkey != 0 && (r != keyr || g != keyg || b != keyb))) {
                found_alpha = 1;
                break;
            }
        }
        if (found_alpha == 1) {
            break;
        }
        max_y = y;
    }
    found_alpha = 0;
    for (x = max_x - 1; x >= min_x; --x) {
        for (y = min_y; y < max_y; ++y) {
            pixel = (pixels + y * surf->pitch) + x * bpp;
            switch (bpp) {
                case 1:
                    value = *pixel;
                    break;
                case 2:
                    value = *(Uint16 *)pixel;
                    break;
                case 3:
                    value = pixel[BYTE0];
                    value |= pixel[BYTE1] << 8;
                    value |= pixel[BYTE2] << 16;
                    break;
                default:
                    assert(bpp == 4);
                    value = *(Uint32 *)pixel;
            }
            PG_GetRGBA(value, format, palette, &r, &g, &b, &a);
            if ((a >= min_alpha && has_colorkey == 0) ||
                (has_colorkey != 0 && (r != keyr || g != keyg || b != keyb))) {
                found_alpha = 1;
                break;
            }
        }
        if (found_alpha == 1) {
            break;
        }
        max_x = x;
    }
    found_alpha = 0;
    for (y = min_y; y < max_y; ++y) {
        min_y = y;
        for (x = min_x; x < max_x; ++x) {
            pixel = (pixels + y * surf->pitch) + x * bpp;
            switch (bpp) {
                case 1:
                    value = *pixel;
                    break;
                case 2:
                    value = *(Uint16 *)pixel;
                    break;
                case 3:
                    value = pixel[BYTE0];
                    value |= pixel[BYTE1] << 8;
                    value |= pixel[BYTE2] << 16;
                    break;
                default:
                    assert(bpp == 4);
                    value = *(Uint32 *)pixel;
            }
            PG_GetRGBA(value, format, palette, &r, &g, &b, &a);
            if ((a >= min_alpha && has_colorkey == 0) ||
                (has_colorkey != 0 && (r != keyr || g != keyg || b != keyb))) {
                found_alpha = 1;
                break;
            }
        }
        if (found_alpha == 1) {
            break;
        }
    }
    found_alpha = 0;
    for (x = min_x; x < max_x; ++x) {
        min_x = x;
        for (y = min_y; y < max_y; ++y) {
            pixel = (pixels + y * surf->pitch) + x * bpp;
            switch (bpp) {
                case 1:
                    value = *pixel;
                    break;
                case 2:
                    value = *(Uint16 *)pixel;
                    break;
                case 3:
                    value = pixel[BYTE0];
                    value |= pixel[BYTE1] << 8;
                    value |= pixel[BYTE2] << 16;
                    break;
                default:
                    assert(bpp == 4);
                    value = *(Uint32 *)pixel;
            }
            PG_GetRGBA(value, format, palette, &r, &g, &b, &a);
            if ((a >= min_alpha && has_colorkey == 0) ||
                (has_colorkey != 0 && (r != keyr || g != keyg || b != keyb))) {
                found_alpha = 1;
                break;
            }
        }
        if (found_alpha == 1) {
            break;
        }
    }
    if (!pgSurface_Unlock((pgSurfaceObject *)self)) {
        return RAISE(pgExc_SDLError, "could not unlock surface");
    }

    rect = pgRect_New4(min_x, min_y, max_x - min_x, max_y - min_y);
    return rect;
}

static PyObject *
_raise_get_view_ndim_error(int bitsize, SurfViewKind kind)
{
    const char *name = "<unknown>"; /* guard against a segfault */

    /* Put a human readable name to a surface view kind */
    switch (kind) {
            /* This switch statement is exhaustive over the SurfViewKind enum
             */

        case VIEWKIND_0D:
            name = "contiguous bytes";
            break;
        case VIEWKIND_1D:
            name = "contiguous pixels";
            break;
        case VIEWKIND_2D:
            name = "2D";
            break;
        case VIEWKIND_3D:
            name = "3D";
            break;
        case VIEWKIND_RED:
            name = "red";
            break;
        case VIEWKIND_GREEN:
            name = "green";
            break;
        case VIEWKIND_BLUE:
            name = "blue";
            break;
        case VIEWKIND_ALPHA:
            name = "alpha";
            break;

#ifndef NDEBUG
            /* Assert this switch statement is exhaustive */
        default:
            /* Should not be here */
            PyErr_Format(PyExc_SystemError,
                         "pygame bug in _raise_get_view_ndim_error:"
                         " unknown view kind %d",
                         (int)kind);
            return 0;
#endif
    }
    PyErr_Format(PyExc_ValueError,
                 "unsupported bit depth %d for %s reference array", bitsize,
                 name);
    return 0;
}

static PyObject *
surf_get_view(PyObject *self, PyObject *args)
{
    SDL_Surface *surface = pgSurface_AsSurface(self);
    Uint32 mask = 0;
    SurfViewKind view_kind = VIEWKIND_2D;
    getbufferproc get_buffer = 0;

    if (!PyArg_ParseTuple(args, "|O&", _view_kind, &view_kind)) {
        return 0;
    }

    SURF_INIT_CHECK(surface)

    PG_PixelFormat *format;
    SDL_Palette *palette;
    if (!PG_GetSurfaceDetails(surface, &format, &palette)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    int bpp = PG_FORMAT_BytesPerPixel(format);

    switch (view_kind) {
            /* This switch statement is exhaustive over the SurfViewKind enum
             */

        case VIEWKIND_0D:
            if (surface->pitch != bpp * surface->w) {
                PyErr_SetString(PyExc_ValueError,
                                "Surface data is not contiguous");
                return 0;
            }
            get_buffer = _get_buffer_0D;
            break;
        case VIEWKIND_1D:
            if (surface->pitch != bpp * surface->w) {
                PyErr_SetString(PyExc_ValueError,
                                "Surface data is not contiguous");
                return 0;
            }
            get_buffer = _get_buffer_1D;
            break;
        case VIEWKIND_2D:
            get_buffer = _get_buffer_2D;
            break;
        case VIEWKIND_3D:
            if (bpp < 3) {
                return _raise_get_view_ndim_error(bpp * 8, view_kind);
            }
            if (format->Gmask != 0x00ff00 &&
                (bpp != 4 || format->Gmask != 0xff0000)) {
                return RAISE(PyExc_ValueError,
                             "unsupported colormasks for 3D reference array");
            }
            get_buffer = _get_buffer_3D;
            break;
        case VIEWKIND_RED:
            mask = format->Rmask;
            if (mask != 0x000000ffU && mask != 0x0000ff00U &&
                mask != 0x00ff0000U && mask != 0xff000000U) {
                return RAISE(PyExc_ValueError,
                             "unsupported colormasks for red reference array");
            }
            get_buffer = _get_buffer_red;
            break;
        case VIEWKIND_GREEN:
            mask = format->Gmask;
            if (mask != 0x000000ffU && mask != 0x0000ff00U &&
                mask != 0x00ff0000U && mask != 0xff000000U) {
                return RAISE(
                    PyExc_ValueError,
                    "unsupported colormasks for green reference array");
            }
            get_buffer = _get_buffer_green;
            break;
        case VIEWKIND_BLUE:
            mask = format->Bmask;
            if (mask != 0x000000ffU && mask != 0x0000ff00U &&
                mask != 0x00ff0000U && mask != 0xff000000U) {
                return RAISE(
                    PyExc_ValueError,
                    "unsupported colormasks for blue reference array");
            }
            get_buffer = _get_buffer_blue;
            break;
        case VIEWKIND_ALPHA:
            mask = format->Amask;
            if (mask != 0x000000ffU && mask != 0x0000ff00U &&
                mask != 0x00ff0000U && mask != 0xff000000U) {
                return RAISE(
                    PyExc_ValueError,
                    "unsupported colormasks for alpha reference array");
            }
            get_buffer = _get_buffer_alpha;
            break;

#ifndef NDEBUG
            /* Assert this switch statement is exhaustive */
        default:
            /* Should not be here */
            PyErr_Format(PyExc_SystemError,
                         "pygame bug in surf_get_view:"
                         " unrecognized view kind %d",
                         (int)view_kind);
            return 0;
#endif
    }
    assert(get_buffer);
    return pgBufferProxy_New(self, get_buffer);
}

static PyObject *
surf_get_buffer(PyObject *self, PyObject *_null)
{
    SDL_Surface *surface = pgSurface_AsSurface(self);
    PyObject *proxy_obj;

    SURF_INIT_CHECK(surface)

    proxy_obj = pgBufferProxy_New(self, _get_buffer_0D);
    if (proxy_obj) {
        if (pgBufferProxy_Trip(proxy_obj)) {
            Py_DECREF(proxy_obj);
            proxy_obj = 0;
        }
    }
    return proxy_obj;
}

static PyObject *
surf_premul_alpha(pgSurfaceObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    PyObject *final;
    SDL_Surface *newsurf;

    SURF_INIT_CHECK(surf)

    pgSurface_Prep(self);
    // Make a copy of the surface first
    newsurf = PG_ConvertSurface(surf, surf->format);

    if ((surf->w > 0 && surf->h > 0)) {
        // If the surface has no pixels we don't need to premul
        // just return the copy.
        int result = premul_surf_color_by_alpha(surf, newsurf);
        if (result == -1) {
            return RAISE(PyExc_ValueError,
                         "source surface to be alpha pre-multiplied must have "
                         "alpha channel");
        }
        else if (result == -2) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
    }
    pgSurface_Unprep(self);

    final = surf_subtype_new(Py_TYPE(self), newsurf, 1);
    if (!final) {
        SDL_FreeSurface(newsurf);
    }
    return final;
}

static PyObject *
surf_premul_alpha_ip(pgSurfaceObject *self, PyObject *_null)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SURF_INIT_CHECK(surf)

    if (!surf->w || !surf->h) {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    pgSurface_Prep(self);

    int result = premul_surf_color_by_alpha(surf, surf);
    if (result == -1) {
        return RAISE(PyExc_ValueError,
                     "source surface to be alpha pre-multiplied must have "
                     "alpha channel");
    }
    else if (result == -2) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    pgSurface_Unprep(self);

    Py_INCREF(self);
    return (PyObject *)self;
}

static int
_get_buffer_0D(PyObject *obj, Py_buffer *view_p, int flags)
{
    SDL_Surface *surface = pgSurface_AsSurface(obj);

    view_p->obj = 0;
    if (_init_buffer(obj, view_p, flags)) {
        return -1;
    }
    view_p->buf = surface->pixels;
    view_p->itemsize = 1;
    view_p->len = (Py_ssize_t)surface->pitch * surface->h;
    view_p->readonly = 0;
    if (PyBUF_HAS_FLAG(flags, PyBUF_FORMAT)) {
        view_p->format = FormatUint8;
    }
    if (PyBUF_HAS_FLAG(flags, PyBUF_ND)) {
        view_p->ndim = 1;
        view_p->shape[0] = view_p->len;
        if (PyBUF_HAS_FLAG(flags, PyBUF_STRIDES)) {
            view_p->strides[0] = view_p->itemsize;
        }
    }
    Py_INCREF(obj);
    view_p->obj = obj;
    return 0;
}

static int
_get_buffer_1D(PyObject *obj, Py_buffer *view_p, int flags)
{
    SDL_Surface *surface = pgSurface_AsSurface(obj);
    Py_ssize_t itemsize = PG_SURF_BytesPerPixel(surface);

    view_p->obj = 0;
    if (itemsize == 1) {
        return _get_buffer_0D(obj, view_p, flags);
    }
    if (_init_buffer(obj, view_p, flags)) {
        return -1;
    }
    if (PyBUF_HAS_FLAG(flags, PyBUF_FORMAT)) {
        switch (itemsize) {
                /* This switch statement is exhaustive over all remaining
                   possible itemsize values, the valid pixel byte sizes of non
                   color-mapped images */

            case 2:
                view_p->format = FormatUint16;
                break;
            case 3:
                view_p->format = FormatUint24;
                break;
            case 4:
                view_p->format = FormatUint32;
                break;

#ifndef NDEBUG
                /* Assert this switch statement is exhaustive */
            default:
                /* Should not be here */
                PyErr_Format(PyExc_SystemError,
                             "Pygame bug caught at line %i in file %s: "
                             "unknown pixel size %zd. Please report",
                             (int)__LINE__, __FILE__, itemsize);
                return -1;
#endif
        }
    }
    view_p->buf = surface->pixels;
    view_p->itemsize = itemsize;
    view_p->readonly = 0;
    view_p->len = (Py_ssize_t)surface->pitch * surface->h;
    if (PyBUF_HAS_FLAG(flags, PyBUF_ND)) {
        view_p->ndim = 1;
        view_p->shape[0] = (Py_ssize_t)surface->w * surface->h;
        if (PyBUF_HAS_FLAG(flags, PyBUF_STRIDES)) {
            view_p->strides[0] = itemsize;
        }
    }
    view_p->suboffsets = 0;
    Py_INCREF(obj);
    view_p->obj = obj;
    return 0;
}

static int
_get_buffer_2D(PyObject *obj, Py_buffer *view_p, int flags)
{
    SDL_Surface *surface = pgSurface_AsSurface(obj);
    int itemsize = PG_SURF_BytesPerPixel(surface);

    view_p->obj = 0;
    if (!PyBUF_HAS_FLAG(flags, PyBUF_ND)) {
        if (surface->pitch != surface->w * itemsize) {
            PyErr_SetString(pgExc_BufferError,
                            "A 2D surface view is not C contiguous");
            return -1;
        }
        return _get_buffer_1D(obj, view_p, flags);
    }
    if (!PyBUF_HAS_FLAG(flags, PyBUF_STRIDES)) {
        PyErr_SetString(pgExc_BufferError,
                        "A 2D surface view is not C contiguous: "
                        "need strides");
        return -1;
    }
    if (PyBUF_HAS_FLAG(flags, PyBUF_C_CONTIGUOUS)) {
        PyErr_SetString(pgExc_BufferError,
                        "A 2D surface view is not C contiguous");
        return -1;
    }
    if (PyBUF_HAS_FLAG(flags, PyBUF_F_CONTIGUOUS) &&
        surface->pitch != surface->w * itemsize) {
        PyErr_SetString(pgExc_BufferError,
                        "This 2D surface view is not F contiguous");
        return -1;
    }
    if (PyBUF_HAS_FLAG(flags, PyBUF_ANY_CONTIGUOUS) &&
        surface->pitch != surface->w * itemsize) {
        PyErr_SetString(pgExc_BufferError,
                        "This 2D surface view is not contiguous");
        return -1;
    }
    if (_init_buffer(obj, view_p, flags)) {
        return -1;
    }
    if (PyBUF_HAS_FLAG(flags, PyBUF_FORMAT)) {
        switch (itemsize) {
                /* This switch statement is exhaustive over all possible
                   itemsize values, valid pixel byte sizes */

            case 1:
                view_p->format = FormatUint8;
                break;
            case 2:
                view_p->format = FormatUint16;
                break;
            case 3:
                view_p->format = FormatUint24;
                break;
            case 4:
                view_p->format = FormatUint32;
                break;

#ifndef NDEBUG
                /* Assert this switch statement is exhaustive */
            default:
                /* Should not be here */
                PyErr_Format(PyExc_SystemError,
                             "Pygame bug caught at line %i in file %s: "
                             "unknown pixel size %i. Please report",
                             (int)__LINE__, __FILE__, itemsize);
                return -1;
#endif
        }
    }
    view_p->buf = surface->pixels;
    view_p->itemsize = itemsize;
    view_p->ndim = 2;
    view_p->readonly = 0;
    view_p->len = (Py_ssize_t)surface->w * surface->h * itemsize;
    view_p->shape[0] = surface->w;
    view_p->shape[1] = surface->h;
    view_p->strides[0] = itemsize;
    view_p->strides[1] = surface->pitch;
    view_p->suboffsets = 0;
    Py_INCREF(obj);
    view_p->obj = obj;
    return 0;
}

static int
_get_buffer_3D(PyObject *obj, Py_buffer *view_p, int flags)
{
    SDL_Surface *surface = pgSurface_AsSurface(obj);
    int pixelsize = PG_SURF_BytesPerPixel(surface);
    char *startpixel = (char *)surface->pixels;

    view_p->obj = 0;
    if (!PyBUF_HAS_FLAG(flags, PyBUF_STRIDES)) {
        PyErr_SetString(pgExc_BufferError,
                        "A 3D surface view is not contiguous: needs strides");
        return -1;
    }
    if (PyBUF_HAS_FLAG(flags, PyBUF_C_CONTIGUOUS) ||
        PyBUF_HAS_FLAG(flags, PyBUF_F_CONTIGUOUS) ||
        PyBUF_HAS_FLAG(flags, PyBUF_ANY_CONTIGUOUS)) {
        PyErr_SetString(pgExc_BufferError,
                        "A 3D surface view is not contiguous");
        return -1;
    }
    PG_PixelFormat *surface_format = PG_GetSurfaceFormat(surface);
    if (surface_format == NULL) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return -1;
    }

    if (_init_buffer(obj, view_p, flags)) {
        return -1;
    }
    if (PyBUF_HAS_FLAG(flags, PyBUF_FORMAT)) {
        view_p->format = FormatUint8;
    }
    view_p->itemsize = 1;
    view_p->ndim = 3;
    view_p->readonly = 0;
    view_p->len = (Py_ssize_t)surface->w * surface->h * 3;
    view_p->shape[0] = surface->w;
    view_p->shape[1] = surface->h;
    view_p->shape[2] = 3;
    view_p->strides[0] = pixelsize;
    view_p->strides[1] = surface->pitch;
    switch (surface_format->Rmask) {
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
        case 0xffU:
            view_p->strides[2] = 1;
            break;
        case 0xff00U:
            assert(pixelsize == 4);
            view_p->strides[2] = 1;
            startpixel += 1;
            break;
        case 0xff0000U:
            view_p->strides[2] = -1;
            startpixel += 2;
            break;
        default: /* 0xff000000U */
            assert(pixelsize == 4);
            view_p->strides[2] = -1;
            startpixel += 3;
#else  /* SDL_BYTEORDER != SDL_LIL_ENDIAN */
        case 0xffU:
            view_p->strides[2] = -1;
            startpixel += pixelsize - 1;
            break;
        case 0xff00U:
            assert(pixelsize == 4);
            view_p->strides[2] = -1;
            startpixel += pixelsize - 2;
            break;
        case 0xff0000U:
            view_p->strides[2] = 1;
            startpixel += pixelsize - 3;
            break;
        default: /* 0xff000000U */
            assert(pixelsize == 4);
            view_p->strides[2] = 1;
#endif /* SDL_BYTEORDER != SDL_LIL_ENDIAN */
    }
    view_p->buf = startpixel;
    Py_INCREF(obj);
    view_p->obj = obj;
    return 0;
}

static int
_get_buffer_red(PyObject *obj, Py_buffer *view_p, int flags)
{
    PG_PixelFormat *surface_format =
        PG_GetSurfaceFormat(pgSurface_AsSurface(obj));
    if (surface_format == NULL) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return -1;
    }

    return _get_buffer_colorplane(obj, view_p, flags, "red",
                                  surface_format->Rmask);
}

static int
_get_buffer_green(PyObject *obj, Py_buffer *view_p, int flags)
{
    PG_PixelFormat *surface_format =
        PG_GetSurfaceFormat(pgSurface_AsSurface(obj));
    if (surface_format == NULL) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return -1;
    }

    return _get_buffer_colorplane(obj, view_p, flags, "green",
                                  surface_format->Gmask);
}

static int
_get_buffer_blue(PyObject *obj, Py_buffer *view_p, int flags)
{
    PG_PixelFormat *surface_format =
        PG_GetSurfaceFormat(pgSurface_AsSurface(obj));
    if (surface_format == NULL) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return -1;
    }

    return _get_buffer_colorplane(obj, view_p, flags, "blue",
                                  surface_format->Bmask);
}

static int
_get_buffer_alpha(PyObject *obj, Py_buffer *view_p, int flags)
{
    PG_PixelFormat *surface_format =
        PG_GetSurfaceFormat(pgSurface_AsSurface(obj));
    if (surface_format == NULL) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return -1;
    }

    return _get_buffer_colorplane(obj, view_p, flags, "alpha",
                                  surface_format->Amask);
}

static int
_get_buffer_colorplane(PyObject *obj, Py_buffer *view_p, int flags, char *name,
                       Uint32 mask)
{
    SDL_Surface *surface = pgSurface_AsSurface(obj);
    int pixelsize = PG_SURF_BytesPerPixel(surface);
    char *startpixel = (char *)surface->pixels;

    view_p->obj = 0;
    if (!PyBUF_HAS_FLAG(flags, PyBUF_STRIDES)) {
        PyErr_SetString(pgExc_BufferError,
                        "A surface color plane view is not contiguous: "
                        "need strides");
        return -1;
    }
    if (PyBUF_HAS_FLAG(flags, PyBUF_C_CONTIGUOUS) ||
        PyBUF_HAS_FLAG(flags, PyBUF_F_CONTIGUOUS) ||
        PyBUF_HAS_FLAG(flags, PyBUF_ANY_CONTIGUOUS)) {
        PyErr_SetString(pgExc_BufferError,
                        "A surface color plane view is not contiguous");
        return -1;
    }
    switch (mask) {
        /* This switch statement is exhaustive over possible mask value,
           the allowable masks for 24 bit and 32 bit surfaces */

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
        case 0x000000ffU:
            break;
        case 0x0000ff00U:
            startpixel += 1;
            break;
        case 0x00ff0000U:
            startpixel += 2;
            break;
        case 0xff000000U:
            startpixel += 3;
            break;
#else  /* SDL_BYTEORDER != SDL_LIL_ENDIAN */
        case 0x000000ffU:
            startpixel += pixelsize - 1;
            break;
        case 0x0000ff00U:
            startpixel += pixelsize - 2;
            break;
        case 0x00ff0000U:
            startpixel += pixelsize - 3;
            break;
        case 0xff000000U:
            break;
#endif /* SDL_BYTEORDER != SDL_LIL_ENDIAN */

#ifndef NDEBUG
            /* Assert this switch statement is exhaustive */
        default:
            /* Should not be here */
            PyErr_Format(PyExc_SystemError,
                         "Pygame bug caught at line %i in file %s: "
                         "unknown mask value %X. Please report",
                         (int)__LINE__, __FILE__, mask);
            return -1;
#endif
    }
    if (_init_buffer(obj, view_p, flags)) {
        return -1;
    }
    view_p->buf = startpixel;
    if (PyBUF_HAS_FLAG(flags, PyBUF_FORMAT)) {
        view_p->format = FormatUint8;
    }
    view_p->itemsize = 1;
    view_p->ndim = 2;
    view_p->readonly = 0;
    view_p->len = (Py_ssize_t)surface->w * surface->h;
    view_p->shape[0] = surface->w;
    view_p->shape[1] = surface->h;
    view_p->strides[0] = pixelsize;
    view_p->strides[1] = surface->pitch;
    Py_INCREF(obj);
    view_p->obj = obj;
    return 0;
}

static int
_init_buffer(PyObject *surf, Py_buffer *view_p, int flags)
{
    PyObject *consumer;
    pg_bufferinternal *internal;

    assert(surf);
    assert(view_p);
    assert(pgSurface_Check(surf));
    assert(PyBUF_HAS_FLAG(flags, PyBUF_PYGAME));
    consumer = ((pg_buffer *)view_p)->consumer;
    assert(consumer);
    internal = PyMem_New(pg_bufferinternal, 1);
    if (!internal) {
        PyErr_NoMemory();
        return -1;
    }
    internal->consumer_ref = PyWeakref_NewRef(consumer, 0);
    if (!internal->consumer_ref) {
        PyMem_Free(internal);
        return -1;
    }
    if (!pgSurface_LockBy((pgSurfaceObject *)surf, consumer)) {
        PyErr_Format(pgExc_BufferError,
                     "Unable to lock <%s at %p> by <%s at %p>",
                     Py_TYPE(surf)->tp_name, (void *)surf,
                     Py_TYPE(consumer)->tp_name, (void *)consumer);
        Py_DECREF(internal->consumer_ref);
        PyMem_Free(internal);
        return -1;
    }
    if (PyBUF_HAS_FLAG(flags, PyBUF_ND)) {
        view_p->shape = internal->mem;
        if (PyBUF_HAS_FLAG(flags, PyBUF_STRIDES)) {
            view_p->strides = internal->mem + 3;
        }
        else {
            view_p->strides = 0;
        }
    }
    else {
        view_p->shape = 0;
        view_p->strides = 0;
    }
    view_p->ndim = 0;
    view_p->format = 0;
    view_p->suboffsets = 0;
    view_p->internal = internal;
    ((pg_buffer *)view_p)->release_buffer = _release_buffer;
    return 0;
}

static void
_release_buffer(Py_buffer *view_p)
{
    pg_bufferinternal *internal;
    PyObject *consumer_ref;
    PyObject *consumer = NULL;

    assert(view_p && view_p->obj && view_p->internal);
    internal = (pg_bufferinternal *)view_p->internal;
    consumer_ref = internal->consumer_ref;
    assert(consumer_ref && PyWeakref_CheckRef(consumer_ref));

    if (PyWeakref_GetRef(consumer_ref, &consumer) != 1) {
        PyErr_Clear();  // ignore any errors here
    }

    if (!pgSurface_UnlockBy((pgSurfaceObject *)view_p->obj, consumer)) {
        PyErr_Clear();
    }
    Py_XDECREF(consumer);

    Py_DECREF(consumer_ref);
    PyMem_Free(internal);
    Py_DECREF(view_p->obj);
    view_p->obj = 0;
}

static int
_view_kind(PyObject *obj, void *view_kind_vptr)
{
    unsigned long ch;
    SurfViewKind *view_kind_ptr = (SurfViewKind *)view_kind_vptr;

    if (PyUnicode_Check(obj)) {
        if (PyUnicode_GET_LENGTH(obj) != 1) {
            PyErr_SetString(PyExc_TypeError,
                            "expected a length 1 string for argument 1");
            return 0;
        }
        ch = PyUnicode_READ_CHAR(obj, 0);
    }
    else if (PyBytes_Check(obj)) {
        if (PyBytes_GET_SIZE(obj) != 1) {
            PyErr_SetString(PyExc_TypeError,
                            "expected a length 1 string for argument 1");
            return 0;
        }
        ch = *PyBytes_AS_STRING(obj);
    }
    else {
        PyErr_Format(PyExc_TypeError,
                     "expected a length one string for argument 1: got '%s'",
                     Py_TYPE(obj)->tp_name);
        return 0;
    }
    switch (ch) {
        case '0':
            *view_kind_ptr = VIEWKIND_0D;
            break;
        case '1':
            *view_kind_ptr = VIEWKIND_1D;
            break;
        case '2':
            *view_kind_ptr = VIEWKIND_2D;
            break;
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
        case '3':
            *view_kind_ptr = VIEWKIND_3D;
            break;
        default:
            PyErr_Format(PyExc_TypeError,
                         "unrecognized view kind '%c' for argument 1",
                         (int)ch);
            return 0;
    }
    return 1;
}

static PyObject *
surf_get_pixels_address(PyObject *self, PyObject *closure)
{
    SDL_Surface *surface = pgSurface_AsSurface(self);
    void *address;

    if (!surface) {
        Py_RETURN_NONE;
    }
    if (!surface->pixels) {
        return PyLong_FromLong(0L);
    }
    address = surface->pixels;
#if SIZEOF_VOID_P > SIZEOF_LONG
    return PyLong_FromUnsignedLongLong((unsigned PY_LONG_LONG)address);
#else
    return PyLong_FromUnsignedLong((unsigned long)address);
#endif
}

static int
surface_do_overlap(SDL_Surface *src, SDL_Rect *srcrect, SDL_Surface *dst,
                   SDL_Rect *dstrect, SDL_Rect *clip)
{
    Uint8 *srcpixels;
    Uint8 *dstpixels;
    int srcx = srcrect->x, srcy = srcrect->y;
    int dstx = dstrect->x, dsty = dstrect->y;
    int x, y;
    int w = srcrect->w, h = srcrect->h;
    int maxw, maxh;
    int span;
    int dstoffset;

    /* clip the source rectangle to the source surface */
    if (srcx < 0) {
        w += srcx;
        dstx -= srcx;
        srcx = 0;
    }
    maxw = src->w - srcx;
    if (maxw < w) {
        w = maxw;
    }

    if (srcy < 0) {
        h += srcy;
        dsty -= srcy;
        srcy = 0;
    }
    maxh = src->h - srcy;
    if (maxh < h) {
        h = maxh;
    }

    /* clip the destination rectangle against the clip rectangle */
    x = clip->x - dstx;
    if (x > 0) {
        w -= x;
        dstx += x;
        srcx += x;
    }
    x = dstx + w - clip->x - clip->w;
    if (x > 0) {
        w -= x;
    }
    y = clip->y - dsty;
    if (y > 0) {
        h -= y;
        dsty += y;
        srcy += y;
    }
    y = dsty + h - clip->y - clip->h;
    if (y > 0) {
        h -= y;
    }

    if (w <= 0 || h <= 0) {
        return 0;
    }

    srcpixels = ((Uint8 *)src->pixels + srcy * src->pitch +
                 srcx * PG_SURF_BytesPerPixel(src));
    dstpixels = ((Uint8 *)dst->pixels + dsty * dst->pitch +
                 dstx * PG_SURF_BytesPerPixel(dst));

    if (dstpixels <= srcpixels) {
        return 0;
    }

    span = w * PG_SURF_BytesPerPixel(src);

    if (dstpixels >= srcpixels + (h - 1) * src->pitch + span) {
        return 0;
    }

    dstoffset = (dstpixels - srcpixels) % src->pitch;

    return dstoffset < span || dstoffset > src->pitch - span;
}

int
PG_BlitSurface(SDL_Surface *src, const SDL_Rect *srcrect, SDL_Surface *dst,
               SDL_Rect *dstrect)
{
#if SDL_VERSION_ATLEAST(3, 0, 0)
    /* SDL3 doesn't modify dstrect, so compat for that.
     * Below logic taken from SDL2 source with slight modifications */
    SDL_Rect r_src, r_dst;

    r_src.x = 0;
    r_src.y = 0;
    r_src.w = src->w;
    r_src.h = src->h;

    if (dstrect) {
        r_dst.x = dstrect->x;
        r_dst.y = dstrect->y;
    }
    else {
        r_dst.x = 0;
        r_dst.y = 0;
    }

    /* clip the source rectangle to the source surface */
    if (srcrect) {
        SDL_Rect tmp;
        if (SDL_IntersectRect(srcrect, &r_src, &tmp) == SDL_FALSE) {
            goto end;
        }

        /* Shift dstrect, if srcrect origin has changed */
        r_dst.x += tmp.x - srcrect->x;
        r_dst.y += tmp.y - srcrect->y;

        /* Update srcrect */
        r_src = tmp;
    }

    /* There're no dstrect.w/h parameters. It's the same as srcrect */
    r_dst.w = r_src.w;
    r_dst.h = r_src.h;

    /* clip the destination rectangle against the clip rectangle */
    {
        SDL_Rect tmp, clip_rect;
        SDL_GetSurfaceClipRect(dst, &clip_rect);
        if (SDL_IntersectRect(&r_dst, &clip_rect, &tmp) == SDL_FALSE) {
            goto end;
        }

        /* Update dstrect */
        r_dst = tmp;
    }

    if (r_dst.w > 0 && r_dst.h > 0) {
        if (dstrect) { /* update output parameter */
            *dstrect = r_dst;
        }
        return SDL_BlitSurface(src, srcrect, dst, dstrect) ? 0 : -1;
    }
end:
    if (dstrect) {
        dstrect->w = dstrect->h = 0;
    }
    return 0;
#else
    return SDL_BlitSurface(src, srcrect, dst, dstrect);
#endif
}

/*this internal blit function is accessible through the C api*/
int
pgSurface_Blit(pgSurfaceObject *dstobj, pgSurfaceObject *srcobj,
               SDL_Rect *dstrect, SDL_Rect *srcrect, int blend_flags)
{
    SDL_Surface *src = pgSurface_AsSurface(srcobj);
    SDL_Surface *dst = pgSurface_AsSurface(dstobj);
    SDL_Surface *subsurface = NULL;
    int result, suboffsetx = 0, suboffsety = 0;
    SDL_Rect orig_clip, sub_clip, dstclip;
    Uint8 alpha;

    if (!PG_GetSurfaceClipRect(dst, &dstclip)) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return 1;
    }

    /* passthrough blits to the real surface */
    if (((pgSurfaceObject *)dstobj)->subsurface) {
        PyObject *owner;
        struct pgSubSurface_Data *subdata;

        subdata = ((pgSurfaceObject *)dstobj)->subsurface;
        owner = subdata->owner;
        subsurface = pgSurface_AsSurface(owner);
        suboffsetx = subdata->offsetx;
        suboffsety = subdata->offsety;

        while (((pgSurfaceObject *)owner)->subsurface) {
            subdata = ((pgSurfaceObject *)owner)->subsurface;
            owner = subdata->owner;
            subsurface = pgSurface_AsSurface(owner);
            suboffsetx += subdata->offsetx;
            suboffsety += subdata->offsety;
        }

        SDL_GetClipRect(subsurface, &orig_clip);
        SDL_GetClipRect(dst, &sub_clip);
        sub_clip.x += suboffsetx;
        sub_clip.y += suboffsety;
        SDL_SetClipRect(subsurface, &sub_clip);
        dstrect->x += suboffsetx;
        dstrect->y += suboffsety;
        dst = subsurface;
    }
    else {
        pgSurface_Prep(dstobj);
        subsurface = NULL;
    }

    pgSurface_Prep(srcobj);

    if ((blend_flags != 0 && blend_flags != PYGAME_BLEND_ALPHA_SDL2) ||
        ((SDL_HasColorKey(src) || _PgSurface_SrcAlpha(src) == 1) &&
         /* This simplification is possible because a source subsurface
            is converted to its owner with a clip rect and a dst
            subsurface cannot be blitted to its owner because the
            owner is locked.
            */
         dst->pixels == src->pixels && srcrect != NULL &&
         surface_do_overlap(src, srcrect, dst, dstrect, &dstclip))) {
        /* Py_BEGIN_ALLOW_THREADS */
        result = pygame_Blit(src, srcrect, dst, dstrect, blend_flags);
        /* Py_END_ALLOW_THREADS */
    }
    /* can't blit alpha to 8bit, crashes SDL */
    else if (PG_SURF_BytesPerPixel(dst) == 1 &&
             (SDL_ISPIXELFORMAT_ALPHA(PG_SURF_FORMATENUM(src)) ||
              ((PG_GetSurfaceAlphaMod(src, &alpha) && alpha != 255)))) {
        /* Py_BEGIN_ALLOW_THREADS */
        if (PG_SURF_BytesPerPixel(src) == 1) {
            result = pygame_Blit(src, srcrect, dst, dstrect, 0);
        }
        else {
#if SDL_VERSION_ATLEAST(3, 0, 0)
            const SDL_PixelFormatDetails *fmt =
                SDL_GetPixelFormatDetails(src->format);
            src = fmt ? SDL_ConvertSurface(src,
                                           SDL_GetPixelFormatForMasks(
                                               fmt->bits_per_pixel, fmt->Rmask,
                                               fmt->Gmask, fmt->Bmask, 0))
                      : NULL;
#else
            SDL_PixelFormat *fmt = src->format;
            SDL_PixelFormat *newfmt =
                SDL_AllocFormat(SDL_MasksToPixelFormatEnum(
                    fmt->BitsPerPixel, fmt->Rmask, fmt->Gmask, fmt->Bmask, 0));
            if (!newfmt) {
                src = NULL;
            }
            else {
                src = SDL_ConvertSurface(src, newfmt, 0);
                SDL_FreeFormat(newfmt);
            }
#endif
            if (src) {
                result = PG_BlitSurface(src, srcrect, dst, dstrect);
                SDL_FreeSurface(src);
            }
            else {
                result = -1;
            }
        }
        /* Py_END_ALLOW_THREADS */
    }
    else if (blend_flags != PYGAME_BLEND_ALPHA_SDL2 &&
             !(pg_EnvShouldBlendAlphaSDL2()) && !SDL_HasColorKey(src) &&
             (PG_SURF_BytesPerPixel(dst) == 4 ||
              PG_SURF_BytesPerPixel(dst) == 2) &&
             _PgSurface_SrcAlpha(src) &&
             (SDL_ISPIXELFORMAT_ALPHA(PG_SURF_FORMATENUM(src))) &&
             !PG_SurfaceHasRLE(src) && !PG_SurfaceHasRLE(dst) &&
             !SDL_MUSTLOCK(src) && !SDL_MUSTLOCK(dst)) {
        /* If we have a 32bit source surface with per pixel alpha
           and no RLE we'll use pygame_Blit so we can mimic how SDL1
            behaved */
        result = pygame_Blit(src, srcrect, dst, dstrect, blend_flags);
    }
    else {
        /* Py_BEGIN_ALLOW_THREADS */
        result = PG_BlitSurface(src, srcrect, dst, dstrect);
        /* Py_END_ALLOW_THREADS */
    }

    if (subsurface) {
        SDL_SetClipRect(subsurface, &orig_clip);
        dstrect->x -= suboffsetx;
        dstrect->y -= suboffsety;
    }
    else {
        pgSurface_Unprep(dstobj);
    }
    pgSurface_Unprep(srcobj);

    if (result == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
    }
    if (result == -2) {
        PyErr_SetString(pgExc_SDLError, "Surface was lost");
    }

    return result != 0;
}

static PyMethodDef _surface_methods[] = {{NULL, NULL, 0, NULL}};

int
exec_surface(PyObject *module)
{
    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        return -1;
    }
    import_pygame_color();
    if (PyErr_Occurred()) {
        return -1;
    }
    import_pygame_rect();
    if (PyErr_Occurred()) {
        return -1;
    }
    import_pygame_bufferproxy();
    if (PyErr_Occurred()) {
        return -1;
    }
    _IMPORT_PYGAME_MODULE(surflock);
    if (PyErr_Occurred()) {
        return -1;
    }

    /* type preparation */
    if (PyType_Ready(&pgSurface_Type) < 0) {
        return -1;
    }

    PyObject *apiobj;
    static void *c_api[PYGAMEAPI_SURFACE_NUMSLOTS];
#ifndef BUILD_STATIC
    if (pg_warn_simd_at_runtime_but_uncompiled() < 0) {
        return -1;
    }
#endif
    if (PyModule_AddObjectRef(module, "SurfaceType",
                              (PyObject *)&pgSurface_Type)) {
        return -1;
    }

    if (PyModule_AddObjectRef(module, "Surface",
                              (PyObject *)&pgSurface_Type)) {
        return -1;
    }

    /* export the c api */
    c_api[0] = &pgSurface_Type;
    c_api[1] = pgSurface_New2;
    c_api[2] = pgSurface_Blit;
    c_api[3] = pgSurface_SetSurface;
    apiobj = encapsulate_api(c_api, "surface");
    if (PyModule_Add(module, PYGAMEAPI_LOCAL_ENTRY, apiobj) < 0) {
        return -1;
    }

    if (PyModule_AddObjectRef(module, "_dict", pgSurface_Type.tp_dict)) {
        return -1;
    }

    return 0;
}

MODINIT_DEFINE(surface)
{
#ifndef BUILD_STATIC
    static PyModuleDef_Slot surf_slots[] = {
        {Py_mod_exec, &exec_surface},
#if PY_VERSION_HEX >= 0x030c0000
        {Py_mod_multiple_interpreters,
         Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},  // TODO: see if this can
                                                       // be supported later
#endif
#if PY_VERSION_HEX >= 0x030d0000
        {Py_mod_gil, Py_MOD_GIL_USED},  // TODO: support this later
#endif
        {0, NULL}};
#endif
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "surface",
                                         DOC_SURFACE,
                                         0,
                                         _surface_methods,
#ifndef BUILD_STATIC
                                         surf_slots,
#else
                                         NULL,
#endif
                                         NULL,
                                         NULL,
                                         NULL};
#ifndef BUILD_STATIC
    return PyModuleDef_Init(&_module);
#else
    // in static mode with want surface module to be ready before python types
    // are evaluated eg pygame.surface.Surface in sprite.py
    PyObject *module = PyModule_Create(&_module);
    if (module) {
        if (exec_surface(module) != 0) {
            Py_DECREF(module);
            return NULL;
        }
    }
    return module;
#endif
}
