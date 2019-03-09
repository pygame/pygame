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

#if IS_SDLv2
#include "palette.h"
#endif /* IS_SDLv2 */

#include "doc/surface_doc.h"

#include "structmember.h"

#include "pgcompat.h"

#include "pgbufferproxy.h"

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
static char FormatUint8[] = "B";
static char FormatUint16[] = "=H";
static char FormatUint24[] = "3x";
static char FormatUint32[] = "=I";

typedef struct pg_bufferinternal_s {
    PyObject *consumer_ref; /* A weak reference to a bufferproxy object   */
    Py_ssize_t mem[6];      /* Enough memory for dim 3 shape and strides  */
} pg_bufferinternal;

int
pgSurface_Blit(PyObject *dstobj, PyObject *srcobj, SDL_Rect *dstrect,
               SDL_Rect *srcrect, int the_args);

/* statics */
#if IS_SDLv1
static PyObject *
pgSurface_New(SDL_Surface *info);
static PyObject *
surf_subtype_new(PyTypeObject *type, SDL_Surface *s);
#else  /* IS_SDLv2 */
static PyObject *
pgSurface_New(SDL_Surface *info, int owner);
static PyObject *
surf_subtype_new(PyTypeObject *type, SDL_Surface *s, int owner);
#endif /* IS_SDLv2 */
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
static void
surface_move(Uint8 *src, Uint8 *dst, int h, int span, int srcpitch,
             int dstpitch);

static PyObject *
surf_get_at(PyObject *self, PyObject *args);
static PyObject *
surf_set_at(PyObject *self, PyObject *args);
static PyObject *
surf_get_at_mapped(PyObject *self, PyObject *args);
static PyObject *
surf_map_rgb(PyObject *self, PyObject *args);
static PyObject *
surf_unmap_rgb(PyObject *self, PyObject *arg);
static PyObject *
surf_lock(PyObject *self);
static PyObject *
surf_unlock(PyObject *self);
static PyObject *
surf_mustlock(PyObject *self);
static PyObject *
surf_get_locked(PyObject *self);
static PyObject *
surf_get_locks(PyObject *self);
static PyObject *
surf_get_palette(PyObject *self);
static PyObject *
surf_get_palette_at(PyObject *self, PyObject *args);
static PyObject *
surf_set_palette(PyObject *self, PyObject *args);
static PyObject *
surf_set_palette_at(PyObject *self, PyObject *args);
static PyObject *
surf_set_colorkey(PyObject *self, PyObject *args);
static PyObject *
surf_get_colorkey(PyObject *self);
static PyObject *
surf_set_alpha(PyObject *self, PyObject *args);
static PyObject *
surf_get_alpha(PyObject *self);
#if IS_SDLv2
static PyObject *
surf_get_blendmode(PyObject *self);
#endif /* IS_SDLv2 */
static PyObject *
surf_copy(PyObject *self);
static PyObject *
surf_convert(PyObject *self, PyObject *args);
static PyObject *
surf_convert_alpha(PyObject *self, PyObject *args);
static PyObject *
surf_set_clip(PyObject *self, PyObject *args);
static PyObject *
surf_get_clip(PyObject *self);
static PyObject *
surf_blit(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *
surf_blits(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *
surf_fill(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *
surf_scroll(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *
surf_get_abs_offset(PyObject *self);
static PyObject *
surf_get_abs_parent(PyObject *self);
static PyObject *
surf_get_bitsize(PyObject *self);
static PyObject *
surf_get_bytesize(PyObject *self);
static PyObject *
surf_get_flags(PyObject *self);
static PyObject *
surf_get_height(PyObject *self);
static PyObject *
surf_get_pitch(PyObject *self);
static PyObject *
surf_get_rect(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *
surf_get_width(PyObject *self);
static PyObject *
surf_get_shifts(PyObject *self);
static PyObject *
surf_set_shifts(PyObject *self, PyObject *args);
static PyObject *
surf_get_size(PyObject *self);
static PyObject *
surf_get_losses(PyObject *self);
static PyObject *
surf_get_masks(PyObject *self);
static PyObject *
surf_set_masks(PyObject *self, PyObject *args);
static PyObject *
surf_get_offset(PyObject *self);
static PyObject *
surf_get_parent(PyObject *self);
static PyObject *
surf_subsurface(PyObject *self, PyObject *args);
static PyObject *
surf_get_view(PyObject *self, PyObject *args);
static PyObject *
surf_get_buffer(PyObject *self);
static PyObject *
surf_get_bounding_rect(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *
surf_get_pixels_address(PyObject *self, PyObject *closure);
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
#if IS_SDLv2
static PyObject *
_raise_create_surface_error(void);
static SDL_Surface *
pg_DisplayFormatAlpha(SDL_Surface *surface);
static SDL_Surface *
pg_DisplayFormat(SDL_Surface *surface);
#endif /* IS_SDLv2 */

#if IS_SDLv2 && !SDL_VERSION_ATLEAST(2, 0, 10)
static Uint32
pg_map_rgb(SDL_Surface *surf, Uint8 r, Uint8 g, Uint8 b)
{
    /* SDL_MapRGB() returns wrong values for color keys
       for indexed formats since since alpha = 0 */
    Uint32 key;
    if (!surf->format->palette)
        return SDL_MapRGB(surf->format, r, g, b);
    if (!SDL_GetColorKey(surf, &key)) {
        Uint8 keyr, keyg, keyb;
        SDL_GetRGB(key, surf->format, &keyr, &keyg, &keyb);
        if (r == keyr && g == keyg && b == keyb)
            return key;
    } else
        SDL_ClearError();
    return SDL_MapRGBA(surf->format, r, g, b, SDL_ALPHA_OPAQUE);
}

static Uint32
pg_map_rgba(SDL_Surface *surf, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    if (!surf->format->palette)
        return SDL_MapRGBA(surf->format, r, g, b, a);
    return pg_map_rgb(surf, r, g, b);
}
#else /* IS_SDLv1 || SDL_VERSION_ATLEAST(2, 0, 10) */
#define pg_map_rgb(surf, r, g, b) SDL_MapRGB((surf)->format, (r), (g), (b))
#define pg_map_rgba(surf, r, g, b, a) SDL_MapRGBA((surf)->format, (r), (g), (b), (a))
#endif /* IS_SDLv1 || SDL_VERSION_ATLEAST(2, 0, 10) */

static PyGetSetDef surface_getsets[] = {
    {"_pixels_address", (getter)surf_get_pixels_address, NULL,
     "pixel buffer address (readonly)", NULL},
    {NULL, NULL, NULL, NULL, NULL}};

static struct PyMethodDef surface_methods[] = {
    {"get_at", surf_get_at, METH_VARARGS, DOC_SURFACEGETAT},
    {"set_at", surf_set_at, METH_VARARGS, DOC_SURFACESETAT},
    {"get_at_mapped", surf_get_at_mapped, METH_VARARGS,
     DOC_SURFACEGETATMAPPED},
    {"map_rgb", surf_map_rgb, METH_VARARGS, DOC_SURFACEMAPRGB},
    {"unmap_rgb", surf_unmap_rgb, METH_O, DOC_SURFACEUNMAPRGB},

    {"get_palette", (PyCFunction)surf_get_palette, METH_NOARGS,
     DOC_SURFACEGETPALETTE},
    {"get_palette_at", surf_get_palette_at, METH_VARARGS,
     DOC_SURFACEGETPALETTEAT},
    {"set_palette", surf_set_palette, METH_VARARGS, DOC_SURFACESETPALETTE},
    {"set_palette_at", surf_set_palette_at, METH_VARARGS,
     DOC_SURFACESETPALETTEAT},

    {"lock", (PyCFunction)surf_lock, METH_NOARGS, DOC_SURFACELOCK},
    {"unlock", (PyCFunction)surf_unlock, METH_NOARGS, DOC_SURFACEUNLOCK},
    {"mustlock", (PyCFunction)surf_mustlock, METH_NOARGS, DOC_SURFACEMUSTLOCK},
    {"get_locked", (PyCFunction)surf_get_locked, METH_NOARGS,
     DOC_SURFACEGETLOCKED},
    {"get_locks", (PyCFunction)surf_get_locks, METH_NOARGS,
     DOC_SURFACEGETLOCKS},

    {"set_colorkey", surf_set_colorkey, METH_VARARGS, DOC_SURFACESETCOLORKEY},
    {"get_colorkey", (PyCFunction)surf_get_colorkey, METH_NOARGS,
     DOC_SURFACEGETCOLORKEY},
    {"set_alpha", surf_set_alpha, METH_VARARGS, DOC_SURFACESETALPHA},
    {"get_alpha", (PyCFunction)surf_get_alpha, METH_NOARGS,
     DOC_SURFACEGETALPHA},
#if IS_SDLv2
    {"get_blendmode", (PyCFunction)surf_get_blendmode, METH_NOARGS,
     "Return the surface's SDL 2 blend mode"},
#endif /* IS_SDLv2 */

    {"copy", (PyCFunction)surf_copy, METH_NOARGS, DOC_SURFACECOPY},
    {"__copy__", (PyCFunction)surf_copy, METH_NOARGS, DOC_SURFACECOPY},
    {"convert", surf_convert, METH_VARARGS, DOC_SURFACECONVERT},
    {"convert_alpha", surf_convert_alpha, METH_VARARGS,
     DOC_SURFACECONVERTALPHA},

    {"set_clip", surf_set_clip, METH_VARARGS, DOC_SURFACESETCLIP},
    {"get_clip", (PyCFunction)surf_get_clip, METH_NOARGS, DOC_SURFACEGETCLIP},

    {"fill", (PyCFunction)surf_fill, METH_VARARGS | METH_KEYWORDS,
     DOC_SURFACEFILL},
    {"blit", (PyCFunction)surf_blit, METH_VARARGS | METH_KEYWORDS,
     DOC_SURFACEBLIT},
    {"blits", (PyCFunction)surf_blits, METH_VARARGS | METH_KEYWORDS,
     DOC_SURFACEBLITS},

    {"scroll", (PyCFunction)surf_scroll, METH_VARARGS | METH_KEYWORDS,
     DOC_SURFACESCROLL},

    {"get_flags", (PyCFunction)surf_get_flags, METH_NOARGS,
     DOC_SURFACEGETFLAGS},
    {"get_size", (PyCFunction)surf_get_size, METH_NOARGS, DOC_SURFACEGETSIZE},
    {"get_width", (PyCFunction)surf_get_width, METH_NOARGS,
     DOC_SURFACEGETWIDTH},
    {"get_height", (PyCFunction)surf_get_height, METH_NOARGS,
     DOC_SURFACEGETHEIGHT},
    {"get_rect", (PyCFunction)surf_get_rect, METH_VARARGS | METH_KEYWORDS,
     DOC_SURFACEGETRECT},
    {"get_pitch", (PyCFunction)surf_get_pitch, METH_NOARGS,
     DOC_SURFACEGETPITCH},
    {"get_bitsize", (PyCFunction)surf_get_bitsize, METH_NOARGS,
     DOC_SURFACEGETBITSIZE},
    {"get_bytesize", (PyCFunction)surf_get_bytesize, METH_NOARGS,
     DOC_SURFACEGETBYTESIZE},
    {"get_masks", (PyCFunction)surf_get_masks, METH_NOARGS,
     DOC_SURFACEGETMASKS},
    {"get_shifts", (PyCFunction)surf_get_shifts, METH_NOARGS,
     DOC_SURFACEGETSHIFTS},
    {"set_masks", (PyCFunction)surf_set_masks, METH_VARARGS,
     DOC_SURFACESETMASKS},
    {"set_shifts", (PyCFunction)surf_set_shifts, METH_VARARGS,
     DOC_SURFACESETSHIFTS},

    {"get_losses", (PyCFunction)surf_get_losses, METH_NOARGS,
     DOC_SURFACEGETLOSSES},

    {"subsurface", surf_subsurface, METH_VARARGS, DOC_SURFACESUBSURFACE},
    {"get_offset", (PyCFunction)surf_get_offset, METH_NOARGS,
     DOC_SURFACEGETOFFSET},
    {"get_abs_offset", (PyCFunction)surf_get_abs_offset, METH_NOARGS,
     DOC_SURFACEGETABSOFFSET},
    {"get_parent", (PyCFunction)surf_get_parent, METH_NOARGS,
     DOC_SURFACEGETPARENT},
    {"get_abs_parent", (PyCFunction)surf_get_abs_parent, METH_NOARGS,
     DOC_SURFACEGETABSPARENT},
    {"get_bounding_rect", (PyCFunction)surf_get_bounding_rect,
     METH_VARARGS | METH_KEYWORDS, DOC_SURFACEGETBOUNDINGRECT},
    {"get_view", (PyCFunction)surf_get_view, METH_VARARGS, DOC_SURFACEGETVIEW},
    {"get_buffer", (PyCFunction)surf_get_buffer, METH_NOARGS,
     DOC_SURFACEGETBUFFER},

    {NULL, NULL, 0, NULL}};

static PyTypeObject pgSurface_Type = {
    TYPE_HEAD(NULL, 0) "pygame.Surface", /* name */
    sizeof(pgSurfaceObject),             /* basic size */
    0,                                   /* itemsize */
    surface_dealloc,                     /* dealloc */
    0,                                   /* print */
    NULL,                                /* getattr */
    NULL,                                /* setattr */
    NULL,                                /* compare */
    surface_str,                         /* repr */
    NULL,                                /* as_number */
    NULL,                                /* as_sequence */
    NULL,                                /* as_mapping */
    (hashfunc)NULL,                      /* hash */
    (ternaryfunc)NULL,                   /* call */
    (reprfunc)NULL,                      /* str */
    0,
    0L,
    0L,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    DOC_PYGAMESURFACE,                        /* Documentation string */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    offsetof(pgSurfaceObject, weakreflist),   /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    surface_methods,                          /* tp_methods */
    0,                                        /* tp_members */
    surface_getsets,                          /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)surface_init,                   /* tp_init */
    0,                                        /* tp_alloc */
    surface_new,                              /* tp_new */
};

#define pgSurface_Check(x) \
    (PyObject_IsInstance((x), (PyObject *)&pgSurface_Type))

#if IS_SDLv1

static PyObject *
pgSurface_New(SDL_Surface *s)
{
    return surf_subtype_new(&pgSurface_Type, s);
}
#else  /* IS_SDLv2 */

static PyObject *
pgSurface_New(SDL_Surface *s, int owner)
{
    return surf_subtype_new(&pgSurface_Type, s, owner);
}
#endif /* IS_SDLv2 */

#if IS_SDLv1

static PyObject *
surf_subtype_new(PyTypeObject *type, SDL_Surface *s)
#else  /* IS_SDLv2 */

static PyObject *
surf_subtype_new(PyTypeObject *type, SDL_Surface *s, int owner)
#endif /* IS_SDLv2 */
{
    pgSurfaceObject *self;

    if (!s)
        return RAISE(pgExc_SDLError, SDL_GetError());

    self = (pgSurfaceObject *)pgSurface_Type.tp_new(type, NULL, NULL);

    if (self)
        self->surf = s;
#if IS_SDLv2
    if (owner)
        self->owner = 1;
#endif /* IS_SDLv2 */

    return (PyObject *)self;
}

static PyObject *
surface_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    pgSurfaceObject *self;

    self = (pgSurfaceObject *)type->tp_alloc(type, 0);
    if (self) {
        self->surf = NULL;
#if IS_SDLv2
        self->owner = 0;
#endif /* IS_SDLv2 */
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
#if IS_SDLv1
    if (self->surf) {
        if (!(self->surf->flags & SDL_HWSURFACE) ||
            SDL_WasInit(SDL_INIT_VIDEO)) {
            /* unsafe to free hardware surfaces without video init */
            /* i question SDL's ability to free a locked hardware surface */
            SDL_FreeSurface(self->surf);
        }
        self->surf = NULL;
    }
#else  /* IS_SDLv2 */
    if (self->surf && self->owner) {
        SDL_FreeSurface(self->surf);
        self->surf = NULL;
    }
#endif /* IS_SDLv2 */
    if (self->subsurface) {
        Py_XDECREF(self->subsurface->owner);
        PyMem_Del(self->subsurface);
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
#if IS_SDLv2
    self->owner = 0;
#endif /* IS_SDLv2 */
}

static void
surface_dealloc(PyObject *self)
{
    if (((pgSurfaceObject *)self)->weakreflist)
        PyObject_ClearWeakRefs(self);
    surface_cleanup((pgSurfaceObject *)self);
    self->ob_type->tp_free(self);
}

static PyObject *
surface_str(PyObject *self)
{
    char str[1024];
    SDL_Surface *surf = pgSurface_AsSurface(self);
#if IS_SDLv1
    const char *type;
#endif /* IS_SDLv1 */

    if (surf) {
#if IS_SDLv1
        type = (surf->flags & SDL_HWSURFACE) ? "HW" : "SW";
        sprintf(str, "<Surface(%dx%dx%d %s)>", surf->w, surf->h,
                surf->format->BitsPerPixel, type);
#else  /* IS_SDLv2 */
        sprintf(str, "<Surface(%dx%dx%d SW)>", surf->w, surf->h,
                surf->format->BitsPerPixel);
#endif /* IS_SDLv2 */
    }
    else {
        strcpy(str, "<Surface(Dead Display)>");
    }

    return Text_FromUTF8(str);
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
    SDL_PixelFormat default_format;
    int length;

    char *kwids[] = {"size", "flags", "depth", "masks", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iOO", kwids, &size, &flags,
                                     &depth, &masks))
        return -1;

    if (PySequence_Check(size) && (length = PySequence_Length(size)) == 2) {
        if ((!pg_IntFromObjIndex(size, 0, &width)) ||
            (!pg_IntFromObjIndex(size, 1, &height))) {
            RAISE(PyExc_ValueError,
                  "size needs to be (int width, int height)");
            return -1;
        }
    }
    else {
        RAISE(PyExc_ValueError, "size needs to be (int width, int height)");
        return -1;
    }

    if (width < 0 || height < 0) {
        RAISE(pgExc_SDLError, "Invalid resolution for Surface");
        return -1;
    }

#if IS_SDLv2
    default_format.palette = NULL;
#endif /* IS_SDLv2 */

    surface_cleanup(self);

    if (depth && masks) { /* all info supplied, most errorchecking
                           * needed */
        if (pgSurface_Check(depth)) {
            RAISE(PyExc_ValueError,
                  "cannot pass surface for depth and color masks");
            return -1;
        }
        if (!pg_IntFromObj(depth, &bpp)) {
            RAISE(PyExc_ValueError, "invalid bits per pixel depth argument");
            return -1;
        }
        if (!PySequence_Check(masks) || PySequence_Length(masks) != 4) {
            RAISE(PyExc_ValueError,
                  "masks argument must be sequence of four numbers");
            return -1;
        }
        if (!pg_UintFromObjIndex(masks, 0, &Rmask) ||
            !pg_UintFromObjIndex(masks, 1, &Gmask) ||
            !pg_UintFromObjIndex(masks, 2, &Bmask) ||
            !pg_UintFromObjIndex(masks, 3, &Amask)) {
            RAISE(PyExc_ValueError, "invalid mask values in masks sequence");
            return -1;
        }
    }
    else if (depth && PyNumber_Check(depth)) { /* use default masks */
        if (!pg_IntFromObj(depth, &bpp)) {
            RAISE(PyExc_ValueError, "invalid bits per pixel depth argument");
            return -1;
        }
#if IS_SDLv1
        if (flags & SDL_SRCALPHA) {
#else  /* IS_SDLv2 */
        if (flags & PGS_SRCALPHA) {
#endif /* IS_SDLv2 */
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
                    RAISE(PyExc_ValueError,
                          "no standard masks exist for given bitdepth with "
                          "alpha");
                    return -1;
            }
        }
        else {
            Amask = 0;
            switch (bpp) {
                case 8:
#if IS_SDLv1
                    Rmask = 0xFF >> 6 << 5;
                    Gmask = 0xFF >> 5 << 2;
                    Bmask = 0xFF >> 6;
#else  /* IS_SDLv2 */
                    Rmask = 0;
                    Gmask = 0;
                    Bmask = 0;
#endif /* IS_SDLv2 */
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
                    RAISE(PyExc_ValueError, "nonstandard bit depth given");
                    return -1;
            }
        }
    }
    else { /* no depth or surface */
        SDL_PixelFormat *pix;
        if (depth && pgSurface_Check(depth))
            pix = ((pgSurfaceObject *)depth)->surf->format;
#if IS_SDLv1
        else if (SDL_GetVideoSurface())
            pix = SDL_GetVideoSurface()->format;
        else if (SDL_WasInit(SDL_INIT_VIDEO))
            pix = SDL_GetVideoInfo()->vfmt;
#else  /* IS_SDLv2 */
        else if (pg_GetDefaultWindowSurface())
            pix = pgSurface_AsSurface(pg_GetDefaultWindowSurface())->format;
#endif /* IS_SDLv2 */
        else {
            pix = &default_format;
            pix->BitsPerPixel = 32;
            pix->Amask = 0;
            pix->Rmask = 0xFF0000;
            pix->Gmask = 0xFF00;
            pix->Bmask = 0xFF;
        }
        bpp = pix->BitsPerPixel;

#if IS_SDLv1
        if (flags & SDL_SRCALPHA) {
#else  /* IS_SDLv2 */
        if (flags & PGS_SRCALPHA) {
#endif /* IS_SDLv2 */
            switch (bpp) {
                case 16:
                    Rmask = 0xF << 8;
                    Gmask = 0xF << 4;
                    Bmask = 0xF;
                    Amask = 0xF << 12;
                    break;
                case 24:
                    bpp = 32;
                    // we automatically step up to 32 if video is 24, fall
                    // through to case below
                case 32:
                    Rmask = 0xFF << 16;
                    Gmask = 0xFF << 8;
                    Bmask = 0xFF;
                    Amask = 0xFF << 24;
                    break;
                default:
                    RAISE(PyExc_ValueError,
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

#if IS_SDLv1
    surface = SDL_CreateRGBSurface(flags, width, height, bpp, Rmask, Gmask,
                                   Bmask, Amask);
    if (!surface) {
        RAISE(pgExc_SDLError, SDL_GetError());
        return -1;
    }

    if (masks) {
        /* Confirm the surface was created correctly (masks were valid).
           Also ensure that 24 and 32 bit surfaces have 8 bit fields
           (no losses).
        */
        SDL_PixelFormat *format = surface->format;
        Rmask = (0xFF >> format->Rloss) << format->Rshift;
        Gmask = (0xFF >> format->Gloss) << format->Gshift;
        Bmask = (0xFF >> format->Bloss) << format->Bshift;
        Amask = (0xFF >> format->Aloss) << format->Ashift;
        if (format->Rmask != Rmask || format->Gmask != Gmask ||
            format->Bmask != Bmask || format->Amask != Amask ||
            (format->BytesPerPixel >= 3 &&
             (format->Rloss || format->Gloss || format->Bloss ||
              (surface->flags & SDL_SRCALPHA ? format->Aloss
                                             : format->Aloss != 8)))) {
            SDL_FreeSurface(surface);
            RAISE(PyExc_ValueError, "Invalid mask values");
            return -1;
        }
    }

    if (surface) {
        self->surf = surface;
        self->subsurface = NULL;
    }
#else  /* IS_SDLv2 */
    surface = SDL_CreateRGBSurface(0, width, height, bpp, Rmask, Gmask, Bmask,
                                   Amask);
    if (!surface) {
        _raise_create_surface_error();
        return -1;
    }

    if (SDL_ISPIXELFORMAT_INDEXED(surface->format->format)) {
        /* Give the surface something other than an all white palette.
         *          */
        if (SDL_SetPaletteColors(surface->format->palette,
                                 default_palette_colors, 0,
                                 default_palette_size - 1) != 0) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(surface);
            return -1;
        }
    }

    if (surface) {
        self->surf = surface;
        self->owner = 1;
        self->subsurface = NULL;
    }
#endif /* IS_SDLv2 */

    return 0;
}

#if IS_SDLv2
static PyObject *
_raise_create_surface_error(void)
{
    const char *msg = SDL_GetError();

    if (strcmp(msg, "Unknown pixel format") == 0)
        return RAISE(PyExc_ValueError, "Invalid mask values");
    return RAISE(pgExc_SDLError, msg);
}
#endif /* IS_SDLv2 */

/* surface object methods */
static PyObject *
surf_get_at(PyObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SDL_PixelFormat *format = surf->format;
    Uint8 *pixels = (Uint8 *)surf->pixels;
    int x, y;
    Uint32 color;
    Uint8 *pix;
#if IS_SDLv1
    Uint8 rgba[4];
#else  /* IS_SDLv2 */
    Uint8 rgba[4] = {0, 0, 0, 255};
#endif /* IS_SDLv2 */

    if (!PyArg_ParseTuple(args, "(ii)", &x, &y))
        return NULL;
    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

#if IS_SDLv1
    if (surf->flags & SDL_OPENGL)
        return RAISE(pgExc_SDLError, "Cannot call on OPENGL Surfaces");
#endif /* IS_SDLv1 */

    if (x < 0 || x >= surf->w || y < 0 || y >= surf->h)
        return RAISE(PyExc_IndexError, "pixel index out of range");

    if (format->BytesPerPixel < 1 || format->BytesPerPixel > 4)
        return RAISE(PyExc_RuntimeError, "invalid color depth for surface");

    if (!pgSurface_Lock(self))
        return NULL;

    pixels = (Uint8 *)surf->pixels;

    switch (format->BytesPerPixel) {
#if IS_SDLv1
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
#else /* IS_SDLv2 */
        case 1:
            color = (Uint32) * ((Uint8 *)pixels + y * surf->pitch + x);
            SDL_GetRGB(color, format, rgba, rgba + 1, rgba + 2);
            break;
        case 2:
            color = (Uint32) * ((Uint16 *)(pixels + y * surf->pitch) + x);
            SDL_GetRGBA(color, format, rgba, rgba + 1, rgba + 2, rgba + 3);
            break;
        case 3:
            pix = ((Uint8 *)(pixels + y * surf->pitch) + x * 3);
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
            color = (pix[0]) + (pix[1] << 8) + (pix[2] << 16);
#else
            color = (pix[2]) + (pix[1] << 8) + (pix[0] << 16);
#endif
            SDL_GetRGB(color, format, rgba, rgba + 1, rgba + 2);
            break;
        default: /* case 4: */
            assert(format->BytesPerPixel == 4);
            color = *((Uint32 *)(pixels + y * surf->pitch) + x);
            SDL_GetRGBA(color, format, rgba, rgba + 1, rgba + 2, rgba + 3);
            break;
#endif /* IS_SDLv2 */
    }
    if (!pgSurface_Unlock(self))
        return NULL;

#if IS_SDLv1
    SDL_GetRGBA(color, format, rgba, rgba + 1, rgba + 2, rgba + 3);
#endif /* IS_SDLv1 */
    return pgColor_New(rgba);
}

static PyObject *
surf_set_at(PyObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SDL_PixelFormat *format = surf->format;
    Uint8 *pixels;
    int x, y;
    Uint32 color;
    Uint8 rgba[4] = {0, 0, 0, 0};
    PyObject *rgba_obj;
    Uint8 *byte_buf;

    if (!PyArg_ParseTuple(args, "(ii)O", &x, &y, &rgba_obj))
        return NULL;
    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

#if IS_SDLv1
    if (surf->flags & SDL_OPENGL)
        return RAISE(pgExc_SDLError, "Cannot call on OPENGL Surfaces");
#endif /* IS_SDLv1 */

    if (format->BytesPerPixel < 1 || format->BytesPerPixel > 4)
        return RAISE(PyExc_RuntimeError, "invalid color depth for surface");

    if (x < surf->clip_rect.x || x >= surf->clip_rect.x + surf->clip_rect.w ||
        y < surf->clip_rect.y || y >= surf->clip_rect.y + surf->clip_rect.h) {
        /* out of clip area */
        Py_RETURN_NONE;
    }

    if (PyInt_Check(rgba_obj)) {
        color = (Uint32)PyInt_AsLong(rgba_obj);
        if (PyErr_Occurred() && (Sint32)color == -1)
            return RAISE(PyExc_TypeError, "invalid color argument");
    }
    else if (PyLong_Check(rgba_obj)) {
        color = (Uint32)PyLong_AsUnsignedLong(rgba_obj);
        if (PyErr_Occurred() && (Sint32)color == -1)
            return RAISE(PyExc_TypeError, "invalid color argument");
    }
    else if (pg_RGBAFromColorObj(rgba_obj, rgba))
        color = pg_map_rgba(surf, rgba[0], rgba[1], rgba[2], rgba[3]);
    else
        return RAISE(PyExc_TypeError, "invalid color argument");

    if (!pgSurface_Lock(self))
        return NULL;
    pixels = (Uint8 *)surf->pixels;

    switch (format->BytesPerPixel) {
        case 1:
            *((Uint8 *)pixels + y * surf->pitch + x) = (Uint8)color;
            break;
        case 2:
            *((Uint16 *)(pixels + y * surf->pitch) + x) = (Uint16)color;
            break;
        case 3:
            byte_buf = (Uint8 *)(pixels + y * surf->pitch) + x * 3;
#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
            *(byte_buf + (format->Rshift >> 3)) = (Uint8)(color >> 16);
            *(byte_buf + (format->Gshift >> 3)) = (Uint8)(color >> 8);
            *(byte_buf + (format->Bshift >> 3)) = (Uint8)color;
#else
            *(byte_buf + 2 - (format->Rshift >> 3)) = (Uint8)(color >> 16);
            *(byte_buf + 2 - (format->Gshift >> 3)) = (Uint8)(color >> 8);
            *(byte_buf + 2 - (format->Bshift >> 3)) = (Uint8)color;
#endif
            break;
        default: /* case 4: */
            *((Uint32 *)(pixels + y * surf->pitch) + x) = color;
            break;
    }

    if (!pgSurface_Unlock(self))
        return NULL;
    Py_RETURN_NONE;
}

static PyObject *
surf_get_at_mapped(PyObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SDL_PixelFormat *format = surf->format;
    Uint8 *pixels = (Uint8 *)surf->pixels;
    int x, y;
    Sint32 color;
    Uint8 *pix;

    if (!PyArg_ParseTuple(args, "(ii)", &x, &y))
        return NULL;
    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

#if IS_SDLv1
    if (surf->flags & SDL_OPENGL)
        return RAISE(pgExc_SDLError, "Cannot call on OPENGL Surfaces");
#endif /* IS_SDLv1 */

    if (x < 0 || x >= surf->w || y < 0 || y >= surf->h)
        return RAISE(PyExc_IndexError, "pixel index out of range");

    if (format->BytesPerPixel < 1 || format->BytesPerPixel > 4)
        return RAISE(PyExc_RuntimeError, "invalid color depth for surface");

    if (!pgSurface_Lock(self))
        return NULL;

    pixels = (Uint8 *)surf->pixels;

    switch (format->BytesPerPixel) {
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
    if (!pgSurface_Unlock(self))
        return NULL;

    return PyInt_FromLong((long)color);
}

static PyObject *
surf_map_rgb(PyObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    Uint8 rgba[4];
    int color;

    if (!pg_RGBAFromColorObj(args, rgba))
        return RAISE(PyExc_TypeError, "Invalid RGBA argument");
    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

    color = pg_map_rgba(surf, rgba[0], rgba[1], rgba[2], rgba[3]);
    return PyInt_FromLong(color);
}

static PyObject *
surf_unmap_rgb(PyObject *self, PyObject *arg)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    Uint32 col;
    Uint8 rgba[4];

    col = (Uint32)PyInt_AsLong(arg);
    if (col == (Uint32)-1 && PyErr_Occurred()) {
        PyErr_Clear();
        return RAISE(PyExc_TypeError, "unmap_rgb expects 1 number argument");
    }
    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

#if IS_SDLv1
    SDL_GetRGBA(col, surf->format, rgba, rgba + 1, rgba + 2, rgba + 3);
#else  /* IS_SDLv2 */
    if (SDL_ISPIXELFORMAT_ALPHA(surf->format->format))
        SDL_GetRGBA(col, surf->format, rgba, rgba + 1, rgba + 2, rgba + 3);
    else {
        SDL_GetRGB(col, surf->format, rgba, rgba + 1, rgba + 2);
        rgba[3] = 255;
    }
#endif /* IS_SDLv2 */

    return pgColor_New(rgba);
}

static PyObject *
surf_lock(PyObject *self)
{
    if (!pgSurface_Lock(self))
        return NULL;
    Py_RETURN_NONE;
}

static PyObject *
surf_unlock(PyObject *self)
{
    pgSurface_Unlock(self);
    Py_RETURN_NONE;
}

static PyObject *
surf_mustlock(PyObject *self)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    return PyInt_FromLong(SDL_MUSTLOCK(surf) ||
                          ((pgSurfaceObject *)self)->subsurface);
}

static PyObject *
surf_get_locked(PyObject *self)
{
    pgSurfaceObject *surf = (pgSurfaceObject *)self;

    if (surf->locklist && PyList_Size(surf->locklist) > 0)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
surf_get_locks(PyObject *self)
{
    pgSurfaceObject *surf = (pgSurfaceObject *)self;
    Py_ssize_t len, i = 0;
    PyObject *tuple, *tmp;
    if (!surf->locklist)
        return PyTuple_New(0);

    len = PyList_Size(surf->locklist);
    tuple = PyTuple_New(len);
    if (!tuple)
        return NULL;

    for (i = 0; i < len; i++) {
        tmp = PyWeakref_GetObject(PyList_GetItem(surf->locklist, i));
        Py_INCREF(tmp);
        PyTuple_SetItem(tuple, i, tmp);
    }
    return tuple;
}

static PyObject *
surf_get_palette(PyObject *self)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SDL_Palette *pal = surf->format->palette;
    PyObject *list;
    int i;
    PyObject *color;
    SDL_Color *c;
    Uint8 rgba[4] = {0, 0, 0, 255};

    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

    if (!pal)
        return RAISE(pgExc_SDLError, "Surface has no palette to get\n");

    list = PyTuple_New(pal->ncolors);
    if (!list)
        return NULL;

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
    SDL_Palette *pal = surf->format->palette;
    SDL_Color *c;
    int _index;
    Uint8 rgba[4];

    if (!PyArg_ParseTuple(args, "i", &_index))
        return NULL;
    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

    if (!pal)
        return RAISE(pgExc_SDLError, "Surface has no palette to set\n");
    if (_index >= pal->ncolors || _index < 0)
        return RAISE(PyExc_IndexError, "index out of bounds");

    c = &pal->colors[_index];
    rgba[0] = c->r;
    rgba[1] = c->g;
    rgba[2] = c->b;
    rgba[3] = 255;

    return pgColor_NewLength(rgba, 3);
}

static PyObject *
surf_set_palette(PyObject *self, PyObject *args)
{
#if IS_SDLv2
    /* This method works differently from the SDL 1.2 equivalent.
     * It replaces colors in the surface's existing palette. So, if the
     * color list is shorter than the existing palette, only the first
     * part of the palette is replaced. For the SDL 1.2 Pygame version,
     * the actual colors array is replaced, making it shorter.
     */
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SDL_Palette *pal = surf->format->palette;
    const SDL_Color *old_colors;
    SDL_Color colors[256];
#else  /* IS_SDLv1 */
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SDL_Palette *pal = surf->format->palette;
    SDL_Color *colors;
#endif /* IS_SDLv1 */
    PyObject *list, *item;
    int i, len;
    Uint8 rgba[4];
    int ecode;

    if (!PyArg_ParseTuple(args, "O", &list))
        return NULL;
    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");
    if (!PySequence_Check(list))
        return RAISE(PyExc_ValueError, "Argument must be a sequence type");

#if IS_SDLv2
    if (!SDL_ISPIXELFORMAT_INDEXED(surf->format->format))
        return RAISE(pgExc_SDLError, "Surface colors are not indexed\n");

    if (!pal)
        return RAISE(pgExc_SDLError, "Surface is not palettitized\n");
    old_colors = pal->colors;
#else  /* IS_SDLv1 */
    if (!pal)
        return RAISE(pgExc_SDLError, "Surface has no palette\n");
#endif /* IS_SDLv1 */

    if (!SDL_WasInit(SDL_INIT_VIDEO))
        return RAISE(pgExc_SDLError,
                     "cannot set palette without pygame.display initialized");

    len = MIN(pal->ncolors, PySequence_Length(list));

#if IS_SDLv1
    colors = (SDL_Color *)malloc(len * sizeof(SDL_Color));
    if (!colors) {
        PyErr_NoMemory();
        return NULL;
    }
#endif /* IS_SDLv1 */

    for (i = 0; i < len; i++) {
        item = PySequence_GetItem(list, i);

        ecode = pg_RGBAFromObj(item, rgba);
        Py_DECREF(item);
        if (!ecode) {
#if IS_SDLv1
            free(colors);
#endif /* IS_SDLv1 */
            return RAISE(PyExc_ValueError,
                         "takes a sequence of integers of RGB");
        }
        if (rgba[3] != 255) {
#if IS_SDLv1
            free(colors);
#endif /* IS_SDLv1 */
            return RAISE(PyExc_ValueError, "takes an alpha value of 255");
        }
        colors[i].r = (unsigned char)rgba[0];
        colors[i].g = (unsigned char)rgba[1];
        colors[i].b = (unsigned char)rgba[2];
#if IS_SDLv2
        /* Preserve palette alphas. Normally, a palette entry has alpha 255.
         * If, however, colorkey is set, the corresponding palette entry has
         * 0 alpha.
         */
        colors[i].a = (unsigned char)old_colors[i].a;
#endif /* IS_SDLv2 */
    }

#if IS_SDLv1
    SDL_SetColors(surf, colors, 0, len);
    free(colors);
#else  /* IS_SDLv2 */
    ecode = SDL_SetPaletteColors(pal, colors, 0, len);
    if (ecode != 0)
        return RAISE(pgExc_SDLError, SDL_GetError());
#endif /* IS_SDLv2 */
    Py_RETURN_NONE;
}

static PyObject *
surf_set_palette_at(PyObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SDL_Palette *pal = surf->format->palette;
    SDL_Color color;
    int _index;
    PyObject *color_obj;
    Uint8 rgba[4];

    if (!PyArg_ParseTuple(args, "iO", &_index, &color_obj))
        return NULL;
    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

    if (!pg_RGBAFromObj(color_obj, rgba)) {
        return RAISE(PyExc_ValueError,
                     "takes a sequence of integers of RGB for argument 2");
    }

#if IS_SDLv2
    if (!SDL_ISPIXELFORMAT_INDEXED(surf->format->format))
        return RAISE(pgExc_SDLError, "Surface colors are not indexed\n");
#endif /* IS_SDLv2 */

    if (!pal) {
        PyErr_SetString(pgExc_SDLError, "Surface is not palettized\n");
        return NULL;
    }

    if (_index >= pal->ncolors || _index < 0) {
        PyErr_SetString(PyExc_IndexError, "index out of bounds");
        return NULL;
    }

    if (!SDL_WasInit(SDL_INIT_VIDEO))
        return RAISE(pgExc_SDLError,
                     "cannot set palette without pygame.display initialized");

#if IS_SDLv1
    color.r = rgba[0];
    color.g = rgba[1];
    color.b = rgba[2];

    SDL_SetColors(surf, &color, _index, 1);
#else  /* IS_SDLv2 */
    color.r = rgba[0];
    color.g = rgba[1];
    color.b = rgba[2];
    color.a = pal->colors[_index].a; /* May be a colorkey color. */

    if (SDL_SetPaletteColors(pal, &color, _index, 1) != 0)
        return RAISE(pgExc_SDLError, SDL_GetError());
#endif /* IS_SDLv2 */

    Py_RETURN_NONE;
}

static PyObject *
surf_set_colorkey(PyObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    Uint32 flags = 0, color = 0;
    PyObject *rgba_obj = NULL;
    Uint8 rgba[4];
    int result;
    int hascolor = SDL_FALSE;

    if (!PyArg_ParseTuple(args, "|Oi", &rgba_obj, &flags))
        return NULL;
    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

#if IS_SDLv1
    if (surf->flags & SDL_OPENGL)
        return RAISE(pgExc_SDLError, "Cannot call on OPENGL Surfaces");
#endif /* IS_SDLv1 */

    if (rgba_obj && rgba_obj != Py_None) {
        if (PyInt_Check(rgba_obj)) {
            color = (Uint32)PyInt_AsLong(rgba_obj);
            if (PyErr_Occurred() && (Sint32)color == -1)
                return RAISE(PyExc_TypeError, "invalid color argument");
        }
        else if (PyLong_Check(rgba_obj)) {
            color = (Uint32)PyLong_AsUnsignedLong(rgba_obj);
            if (PyErr_Occurred() && (Sint32)color == -1)
                return RAISE(PyExc_TypeError, "invalid color argument");
        }
        else if (pg_RGBAFromColorObj(rgba_obj, rgba)) {
#if IS_SDLv1
            color =
                SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
#else  /* IS_SDLv2 */
            if (SDL_ISPIXELFORMAT_ALPHA(surf->format->format))
                color = pg_map_rgba(surf, rgba[0], rgba[1], rgba[2],
                                    rgba[3]);
            else
                color = pg_map_rgb(surf, rgba[0], rgba[1], rgba[2]);
#endif /* IS_SDLv2 */
        }
        else
            return RAISE(PyExc_TypeError, "invalid color argument");
        hascolor = SDL_TRUE;
    }
#if IS_SDLv1
    if (hascolor)
        flags |= SDL_SRCCOLORKEY;
#endif /* IS_SDLv1 */

    pgSurface_Prep(self);
#if IS_SDLv1
    result = SDL_SetColorKey(surf, flags, color);
#else  /* IS_SDLv2 */
    result = 0;
    if (hascolor) {
        /* For an indexed surface, remove the previous colorkey first.
         */
        result = SDL_SetColorKey(surf, SDL_FALSE, color);
    }
    if (result == 0 && hascolor) {
        result = SDL_SetSurfaceRLE(
            surf, (flags & PGS_RLEACCEL) ? SDL_TRUE : SDL_FALSE);
    }
    if (result == 0) {
        result = SDL_SetColorKey(surf, hascolor, color);
    }
#endif /* IS_SDLv2 */
    pgSurface_Unprep(self);

    if (result == -1)
        return RAISE(pgExc_SDLError, SDL_GetError());

    Py_RETURN_NONE;
}

static PyObject *
surf_get_colorkey(PyObject *self)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
#if IS_SDLv1
    Uint8 r, g, b, a;
#else  /* IS_SDLv2 */
    Uint32 mapped_color;
    Uint8 r, g, b, a = 255;
#endif /* IS_SDLv2 */

    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

#if IS_SDLv1
    if (surf->flags & SDL_OPENGL)
        return RAISE(pgExc_SDLError, "Cannot call on OPENGL Surfaces");

    if (!(surf->flags & SDL_SRCCOLORKEY)) {
        Py_RETURN_NONE;
    }

    SDL_GetRGBA(surf->format->colorkey, surf->format, &r, &g, &b, &a);
#else  /* IS_SDLv2 */
    if (SDL_GetColorKey(surf, &mapped_color) != 0) {
        SDL_ClearError();
        Py_RETURN_NONE;
    }

    if (SDL_ISPIXELFORMAT_ALPHA(surf->format->format))
        SDL_GetRGBA(mapped_color, surf->format, &r, &g, &b, &a);
    else
        SDL_GetRGB(mapped_color, surf->format, &r, &g, &b);
#endif /* IS_SDLv2 */

    return Py_BuildValue("(bbbb)", r, g, b, a);
}

static PyObject *
surf_set_alpha(PyObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    Uint32 flags = 0;
    PyObject *alpha_obj = NULL, *intobj = NULL;
    Uint8 alpha;
    int result, alphaval = 255;
#if IS_SDLv1
    int hasalpha = 0;
#endif /* IS_SDLv1 */

    if (!PyArg_ParseTuple(args, "|Oi", &alpha_obj, &flags))
        return NULL;
    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

#if IS_SDLv1
    if (surf->flags & SDL_OPENGL)
        return RAISE(pgExc_SDLError, "Cannot call on OPENGL Surfaces");
#endif /* IS_SDLv1 */

    if (alpha_obj && alpha_obj != Py_None) {
        if (PyNumber_Check(alpha_obj) && (intobj = PyNumber_Int(alpha_obj))) {
            if (PyInt_Check(intobj)) {
                alphaval = (int)PyInt_AsLong(intobj);
                Py_DECREF(intobj);
            }
            else
                return RAISE(PyExc_TypeError, "invalid alpha argument");
        }
        else
            return RAISE(PyExc_TypeError, "invalid alpha argument");
#if IS_SDLv1
        hasalpha = 1;
#else  /* IS_SDLv2 */
        if (SDL_SetSurfaceBlendMode(surf, SDL_BLENDMODE_BLEND) != 0)
            return RAISE(pgExc_SDLError, SDL_GetError());
#endif /* IS_SDLv2 */
    }
#if IS_SDLv2
    else {
        if (SDL_SetSurfaceBlendMode(surf, SDL_BLENDMODE_NONE) != 0)
            return RAISE(pgExc_SDLError, SDL_GetError());
    }
#endif /* IS_SDLv2 */
#if IS_SDLv1
    if (hasalpha)
        flags |= SDL_SRCALPHA;
#endif /* IS_SDLv1 */

    if (alphaval > 255)
        alpha = 255;
    else if (alphaval < 0)
        alpha = 0;
    else
        alpha = (Uint8)alphaval;

    pgSurface_Prep(self);
#if IS_SDLv1
    result = SDL_SetAlpha(surf, flags, alpha);
#else  /* IS_SDLv2 */
    result =
        SDL_SetSurfaceRLE(surf, (flags & PGS_RLEACCEL) ? SDL_TRUE : SDL_FALSE);
    if (result == 0)
        result = SDL_SetSurfaceAlphaMod(surf, alpha);
#endif /* IS_SDLv2 */
    pgSurface_Unprep(self);

    if (result == -1)
        return RAISE(pgExc_SDLError, SDL_GetError());

    Py_RETURN_NONE;
}

static PyObject *
surf_get_alpha(PyObject *self)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
#if IS_SDLv2
    SDL_BlendMode mode;
    Uint8 alpha;
#endif /* IS_SDLv2 */

    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

#if IS_SDLv1
    if (surf->flags & SDL_OPENGL)
        return RAISE(pgExc_SDLError, "Cannot call on OPENGL Surfaces");
#endif /* IS_SDLv1 */

#if IS_SDLv1
    if (surf->flags & SDL_SRCALPHA)
        return PyInt_FromLong(surf->format->alpha);

    Py_RETURN_NONE;
#else  /* IS_SDLv2 */
    if (SDL_GetSurfaceBlendMode(surf, &mode) != 0)
        return RAISE(pgExc_SDLError, SDL_GetError());

    if (mode != SDL_BLENDMODE_BLEND)
        Py_RETURN_NONE;

    if (SDL_GetSurfaceAlphaMod(surf, &alpha) != 0)
        return RAISE(pgExc_SDLError, SDL_GetError());

    return PyInt_FromLong(alpha);
#endif /* IS_SDLv2 */
}

#if IS_SDLv2

static PyObject *
surf_get_blendmode(PyObject *self)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SDL_BlendMode mode;

    if (SDL_GetSurfaceBlendMode(surf, &mode) != 0)
        return RAISE(pgExc_SDLError, SDL_GetError());
    return PyInt_FromLong((long)mode);
}
#endif /* IS_SDLv2 */

static PyObject *
surf_copy(PyObject *self)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    PyObject *final;
    SDL_Surface *newsurf;

    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

#if IS_SDLv1
    if (surf->flags & SDL_OPENGL)
        return RAISE(pgExc_SDLError, "Cannot copy opengl display");
#endif /* IS_SDLv1 */

    pgSurface_Prep(self);
#if IS_SDLv1
    newsurf = SDL_ConvertSurface(surf, surf->format, surf->flags);
    if (surf->flags & SDL_SRCALPHA)
        newsurf->format->alpha = surf->format->alpha;
#else
    newsurf = SDL_ConvertSurface(surf, surf->format, 0);
#endif
    pgSurface_Unprep(self);

#if IS_SDLv1
    final = surf_subtype_new(Py_TYPE(self), newsurf);
#else  /* IS_SDLv2 */
    final = surf_subtype_new(Py_TYPE(self), newsurf, 1);
#endif /* IS_SDLv2 */
    if (!final)
        SDL_FreeSurface(newsurf);
    return final;
}

static PyObject *
surf_convert(PyObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    PyObject *final;
    PyObject *argobject = NULL;
    SDL_Surface *src;
    SDL_Surface *newsurf;
    Uint32 flags = -1;

#if IS_SDLv2
    Uint8 alpha;
    Uint32 colorkey;
    Uint8 key_r, key_g, key_b, key_a = 255;
    int has_colorkey = SDL_FALSE;

#endif /* IS_SDLv2 */


    if (!SDL_WasInit(SDL_INIT_VIDEO))
        return RAISE(pgExc_SDLError,
                     "cannot convert without pygame.display initialized");

    if (!PyArg_ParseTuple(args, "|Oi", &argobject, &flags))
        return NULL;

#if IS_SDLv1
    if (surf->flags & SDL_OPENGL)
        return RAISE(pgExc_SDLError, "Cannot convert opengl display");
#endif /* IS_SDLv1 */

    pgSurface_Prep(self);

#if IS_SDLv2
    if (SDL_GetColorKey(surf, &colorkey) == 0){
        has_colorkey = SDL_TRUE;
        if (SDL_ISPIXELFORMAT_ALPHA(surf->format->format))
            SDL_GetRGBA(colorkey, surf->format, &key_r, &key_g, &key_b, &key_a);
        else
            SDL_GetRGB(colorkey, surf->format, &key_r, &key_g, &key_b);
    }
#endif

    if (argobject) {
        if (pgSurface_Check(argobject)) {
            src = pgSurface_AsSurface(argobject);
#if IS_SDLv1
            flags =
                src->flags | (surf->flags & (SDL_SRCCOLORKEY | SDL_SRCALPHA));
            newsurf = SDL_ConvertSurface(surf, src->format, flags);
#else  /* IS_SDLv2 */
            newsurf = SDL_ConvertSurface(surf, src->format, 0);
#endif /* IS_SDLv2 */
        }
        else {
            int bpp;
            SDL_PixelFormat format;

            memcpy(&format, surf->format, sizeof(format));
            if (pg_IntFromObj(argobject, &bpp)) {
                Uint32 Rmask, Gmask, Bmask, Amask;

#if IS_SDLv1
                if (flags != -1 && flags & SDL_SRCALPHA) {
#else  /* IS_SDLv2 */
                if (flags != -1 && flags & PGS_SRCALPHA) {
#endif /* IS_SDLv2 */
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
#if IS_SDLv1
                            Rmask = 0xFF >> 6 << 5;
                            Gmask = 0xFF >> 5 << 2;
                            Bmask = 0xFF >> 6;
#else  /* IS_SDLv2 */
                            Rmask = 0;
                            Gmask = 0;
                            Bmask = 0;
#endif /* IS_SDLv2 */
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
                format.Rmask = Rmask;
                format.Gmask = Gmask;
                format.Bmask = Bmask;
                format.Amask = Amask;
            }
            else if (PySequence_Check(argobject) &&
                     PySequence_Size(argobject) == 4) {
                Uint32 mask;

                if (!pg_UintFromObjIndex(argobject, 0, &format.Rmask) ||
                    !pg_UintFromObjIndex(argobject, 1, &format.Gmask) ||
                    !pg_UintFromObjIndex(argobject, 2, &format.Bmask) ||
                    !pg_UintFromObjIndex(argobject, 3, &format.Amask)) {
                    pgSurface_Unprep(self);
                    return RAISE(PyExc_ValueError,
                                 "invalid color masks given");
                }
                mask =
                    format.Rmask | format.Gmask | format.Bmask | format.Amask;
                for (bpp = 0; bpp < 32; ++bpp)
                    if (!(mask >> bpp))
                        break;
            }
            else {
                pgSurface_Unprep(self);
                return RAISE(
                    PyExc_ValueError,
                    "invalid argument specifying new format to convert to");
            }
            format.BitsPerPixel = (Uint8)bpp;
            format.BytesPerPixel = (bpp + 7) / 8;
            if (format.BitsPerPixel > 8)
                /* Allow a 8 bit source surface with an empty palette to be
                 * converted to a format without a palette (Issue #131).
                 * If the target format has a non-NULL palette pointer then
                 * SDL_ConvertSurface checks that the palette is not empty--
                 * that at least one entry is not black.
                 */
                format.palette = NULL;
#if IS_SDLv1
            if (flags == -1)
                flags = surf->flags;
            if (format.Amask)
                flags |= SDL_SRCALPHA;
            newsurf = SDL_ConvertSurface(surf, &format, flags);
#else /* IS_SDLv2 */
            newsurf = SDL_ConvertSurface(surf, &format, 0);
#endif /* IS_SDLv2 */

        }
    }
#if IS_SDLv1
    else {
        newsurf = SDL_DisplayFormat(surf);
    }
#else  /* IS_SDLv2 */
    else {
        newsurf = pg_DisplayFormat(surf);
    }

    if (has_colorkey) {
        colorkey = pg_map_rgba(newsurf, key_r, key_g, key_b, key_a);
        if (SDL_SetColorKey(newsurf, SDL_TRUE, colorkey) != 0) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(newsurf);
            return NULL;
        }
    }

#endif /* IS_SDLv2 */
    pgSurface_Unprep(self);

#if IS_SDLv1
    final = surf_subtype_new(Py_TYPE(self), newsurf);
#else  /* IS_SDLv2 */
    final = surf_subtype_new(Py_TYPE(self), newsurf, 1);
#endif /* IS_SDLv2 */
    if (!final)
        SDL_FreeSurface(newsurf);
    return final;
}

#if IS_SDLv2
static SDL_Surface *
pg_DisplayFormat(SDL_Surface *surface)
{
    SDL_Surface *newsurf = NULL, *displaysurf;
    if (!pg_GetDefaultWindowSurface()) {
        SDL_SetError("No video mode has been set");
        return NULL;
    }
    displaysurf = pgSurface_AsSurface(pg_GetDefaultWindowSurface());
    return SDL_ConvertSurface(surface, displaysurf->format, 0);
}

static SDL_Surface *
pg_DisplayFormatAlpha(SDL_Surface *surface)
{
    SDL_Surface *displaysurf;
    SDL_PixelFormat *dformat;
    Uint32 pfe;
    Uint32 amask = 0xff000000;
    Uint32 rmask = 0x00ff0000;
    Uint32 gmask = 0x0000ff00;
    Uint32 bmask = 0x000000ff;

    if (!pg_GetDefaultWindowSurface()) {
        SDL_SetError("No video mode has been set");
        return NULL;
    }
    displaysurf = pgSurface_AsSurface(pg_GetDefaultWindowSurface());
    dformat = displaysurf->format;

    switch (dformat->BytesPerPixel) {
        case 2:
            /* same behavior as SDL1 */
            if ((dformat->Rmask == 0x1f) && (dformat->Bmask == 0xf800 || dformat->Bmask == 0x7c00)) {
                rmask = 0xff;
                bmask = 0xff0000;
            }
            break;
        case 3:
        case 4:
            /* keep the format if the high bits are free */
            if ((dformat->Rmask == 0xff) && (dformat->Bmask == 0xff0000)) {
                rmask = 0xff;
                bmask = 0xff0000;
            }
            else if (dformat->Rmask == 0xff00 && (dformat->Bmask == 0xff000000) ) {
                amask = 0x000000ff;
                rmask = 0x0000ff00;
                gmask = 0x00ff0000;
                bmask = 0xff000000;
            }
            break;
        default: /* ARGB8888 */
            break;
    }
    pfe = SDL_MasksToPixelFormatEnum(32, rmask, gmask, bmask, amask);
    if (pfe == SDL_PIXELFORMAT_UNKNOWN) {
        SDL_SetError("unknown pixel format");
        return NULL;
    }
    return SDL_ConvertSurfaceFormat(surface, pfe, 0);
}
#endif /* IS_SDLv2 */

static PyObject *
surf_convert_alpha(PyObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    PyObject *final;
    pgSurfaceObject *srcsurf = NULL;
    SDL_Surface *newsurf, *src;

    if (!SDL_WasInit(SDL_INIT_VIDEO))
        return RAISE(pgExc_SDLError,
                     "cannot convert without pygame.display initialized");

    if (!PyArg_ParseTuple(args, "|O!", &pgSurface_Type, &srcsurf))
        return NULL;

#pragma PG_WARN("srcsurf doesn't actually do anything?")

#if IS_SDLv2
    /*if (!srcsurf) {}*/
    /*
     * hmm, we have to figure this out, not all depths have good
     * support for alpha
     */
    newsurf = pg_DisplayFormatAlpha(surf);
    final = surf_subtype_new(Py_TYPE(self), newsurf, 1);
#else /* IS_SDLv1 */
    pgSurface_Prep(self);
    if (srcsurf) {
        /*
         * hmm, we have to figure this out, not all depths have good
         * support for alpha
         */
        src = pgSurface_AsSurface(srcsurf);
        newsurf = SDL_DisplayFormatAlpha(surf);
    }
    else
        newsurf = SDL_DisplayFormatAlpha(surf);
    pgSurface_Unprep(self);

    final = surf_subtype_new(Py_TYPE(self), newsurf);
#endif /* IS_SDLv1 */

    if (!final)
        SDL_FreeSurface(newsurf);
    return final;
}

static PyObject *
surf_set_clip(PyObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    PyObject *item;
    GAME_Rect *rect = NULL, temp;
    SDL_Rect sdlrect;
    int result;

    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");
    if (PyTuple_Size(args)) {
        item = PyTuple_GET_ITEM(args, 0);
        if (item == Py_None && PyTuple_Size(args) == 1) {
            result = SDL_SetClipRect(surf, NULL);
        }
        else {
            rect = pgRect_FromObject(args, &temp);
            if (!rect)
                return RAISE(PyExc_ValueError, "invalid rectstyle object");
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
surf_get_clip(PyObject *self)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);

    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");
    return pgRect_New(&surf->clip_rect);
}

static PyObject *
surf_fill(PyObject *self, PyObject *args, PyObject *keywds)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    GAME_Rect *rect, temp;
    PyObject *r = NULL;
    Uint32 color;
    int result;
    PyObject *rgba_obj;
    Uint8 rgba[4];
    SDL_Rect sdlrect;
    int blendargs = 0;

    static char *kwids[] = {"color", "rect", "special_flags", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|Oi", kwids, &rgba_obj,
                                     &r, &blendargs))
        return NULL;
    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

#if IS_SDLv1
    if (surf->flags & SDL_OPENGL)
        return RAISE(pgExc_SDLError, "Cannot call on OPENGL Surfaces");
#endif /* IS_SDLv1 */

    if (PyInt_Check(rgba_obj))
        color = (Uint32)PyInt_AsLong(rgba_obj);
    else if (PyLong_Check(rgba_obj))
        color = (Uint32)PyLong_AsUnsignedLong(rgba_obj);
    else if (pg_RGBAFromColorObj(rgba_obj, rgba))
        color = pg_map_rgba(surf, rgba[0], rgba[1], rgba[2], rgba[3]);
    else
        return RAISE(PyExc_TypeError, "invalid color argument");

    if (!r || r == Py_None) {
        rect = &temp;
        temp.x = temp.y = 0;
        temp.w = surf->w;
        temp.h = surf->h;
    }
    else if (!(rect = pgRect_FromObject(r, &temp)))
        return RAISE(PyExc_ValueError, "invalid rectstyle object");

    /* we need a fresh copy so our Rect values don't get munged */
    if (rect != &temp) {
        memcpy(&temp, rect, sizeof(temp));
        rect = &temp;
    }

    if (rect->w < 0 || rect->h < 0 || rect->x > surf->w || rect->y > surf->h) {
        sdlrect.x = sdlrect.y = 0;
        sdlrect.w = sdlrect.h = 0;
    }
    else {
        sdlrect.x = rect->x;
        sdlrect.y = rect->y;
        sdlrect.w = rect->w;
        sdlrect.h = rect->h;

        // clip the rect to be within the surface.
        if (sdlrect.x + sdlrect.w <= 0 || sdlrect.y + sdlrect.h <= 0) {
            sdlrect.w = 0;
            sdlrect.h = 0;
        }

        if (sdlrect.x < 0) {
            sdlrect.x = 0;
        }
        if (sdlrect.y < 0) {
            sdlrect.y = 0;
        }

        if (sdlrect.x + sdlrect.w > surf->w) {
            sdlrect.w = sdlrect.w + (surf->w - (sdlrect.x + sdlrect.w));
        }
        if (sdlrect.y + sdlrect.h > surf->h) {
            sdlrect.h = sdlrect.h + (surf->h - (sdlrect.y + sdlrect.h));
        }

        /* printf("%d, %d, %d, %d\n", sdlrect.x, sdlrect.y, sdlrect.w,
         * sdlrect.h); */

        if (blendargs != 0) {
            /*
            printf ("Using blendargs: %d\n", blendargs);
            */
            result = surface_fill_blend(surf, &sdlrect, color, blendargs);
        }
        else {
            pgSurface_Prep(self);
            result = SDL_FillRect(surf, &sdlrect, color);
            pgSurface_Unprep(self);
        }
        if (result == -1)
            return RAISE(pgExc_SDLError, SDL_GetError());
    }
    return pgRect_New(&sdlrect);
}

static PyObject *
surf_blit(PyObject *self, PyObject *args, PyObject *keywds)
{
    SDL_Surface *src, *dest = pgSurface_AsSurface(self);
    GAME_Rect *src_rect, temp;
    PyObject *srcobject, *argpos, *argrect = NULL;
    int dx, dy, result;
    SDL_Rect dest_rect, sdlsrc_rect;
    int sx, sy;
    int the_args = 0;

    static char *kwids[] = {"source", "dest", "area", "special_flags", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!O|Oi", kwids,
                                     &pgSurface_Type, &srcobject, &argpos,
                                     &argrect, &the_args))
        return NULL;

    src = pgSurface_AsSurface(srcobject);
    if (!dest || !src)
        return RAISE(pgExc_SDLError, "display Surface quit");

#if IS_SDLv1
    if (dest->flags & SDL_OPENGL &&
        !(dest->flags & (SDL_OPENGLBLIT & ~SDL_OPENGL)))
        return RAISE(pgExc_SDLError,
                     "Cannot blit to OPENGL Surfaces (OPENGLBLIT is ok)");
#endif /* IS_SDLv1 */

    if ((src_rect = pgRect_FromObject(argpos, &temp))) {
        dx = src_rect->x;
        dy = src_rect->y;
    }
    else if (pg_TwoIntsFromObj(argpos, &sx, &sy)) {
        dx = sx;
        dy = sy;
    }
    else
        return RAISE(PyExc_TypeError, "invalid destination position for blit");

    if (argrect && argrect != Py_None) {
        if (!(src_rect = pgRect_FromObject(argrect, &temp)))
            return RAISE(PyExc_TypeError, "Invalid rectstyle argument");
    }
    else {
        temp.x = temp.y = 0;
        temp.w = src->w;
        temp.h = src->h;
        src_rect = &temp;
    }

    dest_rect.x = (short)dx;
    dest_rect.y = (short)dy;
    dest_rect.w = (unsigned short)src_rect->w;
    dest_rect.h = (unsigned short)src_rect->h;
    sdlsrc_rect.x = (short)src_rect->x;
    sdlsrc_rect.y = (short)src_rect->y;
    sdlsrc_rect.w = (unsigned short)src_rect->w;
    sdlsrc_rect.h = (unsigned short)src_rect->h;

    if (!the_args)
        the_args = 0;

    result =
        pgSurface_Blit(self, srcobject, &dest_rect, &sdlsrc_rect, the_args);
    if (result != 0)
        return NULL;

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

static PyObject *
surf_blits(PyObject *self, PyObject *args, PyObject *keywds)
{
    SDL_Surface *src, *dest = pgSurface_AsSurface(self);
    GAME_Rect *src_rect, temp;
    PyObject *srcobject = NULL, *argpos = NULL, *argrect = NULL;
    int dx, dy, result;
    SDL_Rect dest_rect, sdlsrc_rect;
    int sx, sy;
    int the_args = 0;

    PyObject *blitsequence = NULL;
    PyObject *iterator = NULL;
    PyObject *item = NULL;
    PyObject *special_flags = NULL;
    PyObject *ret = NULL;
    PyObject *retrect = NULL;
    int itemlength;
    int doreturn = 1;
    int bliterrornum = 0;
    static char *kwids[] = {"blit_sequence", "doreturn", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|i", kwids, &blitsequence,
                                     &doreturn))
        return NULL;

    if (doreturn) {
        ret = PyList_New(0);
        if (!ret)
            return NULL;
    }
    if (!PyIter_Check(blitsequence) && !PySequence_Check(blitsequence)) {
        bliterrornum = BLITS_ERR_SEQUENCE_REQUIRED;
        goto bliterror;
    }
    iterator = PyObject_GetIter(blitsequence);
    if (!iterator) {
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
        bliterrornum = 0;
        srcobject = NULL;
        argpos = NULL;
        argrect = NULL;
        special_flags = NULL;
        the_args = 0;
        if (itemlength >= 2) {
            /* (Surface, dest) */
            srcobject = PySequence_GetItem(item, 0);
            argpos = PySequence_GetItem(item, 1);
        }
        if (itemlength >= 3) {
            /* (Surface, dest, area) */
            argrect = PySequence_GetItem(item, 2);
        }
        if (itemlength == 4) {
            /* (Surface, dest, area, special_flags) */
            special_flags = PySequence_GetItem(item, 3);
        }
        Py_DECREF(item);

        src = pgSurface_AsSurface(srcobject);
        if (!dest) {
            bliterrornum = BLITS_ERR_DISPLAY_SURF_QUIT;
            goto bliterror;
        }
        if (!src) {
            bliterrornum = BLITS_ERR_SEQUENCE_SURF;
            goto bliterror;
        }

#if IS_SDLv1
        if (dest->flags & SDL_OPENGL &&
            !(dest->flags & (SDL_OPENGLBLIT & ~SDL_OPENGL))) {
            bliterrornum = BLITS_ERR_NO_OPENGL_SURF;
            goto bliterror;
        }
#endif /* IS_SDLv1 */

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

        dest_rect.x = (short)dx;
        dest_rect.y = (short)dy;
        dest_rect.w = (unsigned short)src_rect->w;
        dest_rect.h = (unsigned short)src_rect->h;
        sdlsrc_rect.x = (short)src_rect->x;
        sdlsrc_rect.y = (short)src_rect->y;
        sdlsrc_rect.w = (unsigned short)src_rect->w;
        sdlsrc_rect.h = (unsigned short)src_rect->h;

        if (special_flags) {
            if (!pg_IntFromObj(special_flags, &the_args)) {
                bliterrornum = BLITS_ERR_MUST_ASSIGN_NUMERIC;
                goto bliterror;
            }
        }

        Py_DECREF(srcobject);
        Py_DECREF(argpos);
        Py_XDECREF(argrect);
        Py_XDECREF(special_flags);

        result = pgSurface_Blit(self, srcobject, &dest_rect, &sdlsrc_rect,
                                the_args);
        if (result != 0) {
            bliterrornum = BLITS_ERR_BLIT_FAIL;
            goto bliterror;
        }

        if (doreturn) {
            retrect = NULL;
            retrect = pgRect_New(&dest_rect);
            PyList_Append(ret, retrect);
            Py_DECREF(retrect);
        }
    }

    Py_DECREF(iterator);
    if (PyErr_Occurred()) {
        goto bliterror;
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
    Py_XDECREF(special_flags);
    Py_XDECREF(iterator);
    Py_XDECREF(item);

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
        case BLITS_ERR_NO_OPENGL_SURF:
            return RAISE(pgExc_SDLError,
                         "Cannot blit to OPENGL Surfaces (OPENGLBLIT is ok)");
        case BLITS_ERR_INVALID_DESTINATION:
            return RAISE(PyExc_TypeError,
                         "invalid destination position for blit");
        case BLITS_ERR_INVALID_RECT_STYLE:
            return RAISE(PyExc_TypeError, "Invalid rectstyle argument");
        case BLITS_ERR_MUST_ASSIGN_NUMERIC:
            return RAISE(PyExc_TypeError, "Must assign numeric values");
        case BLITS_ERR_BLIT_FAIL:
            return RAISE(PyExc_TypeError, "Blit failed");
    }
    return RAISE(PyExc_TypeError, "Unknown error");
}

static PyObject *
surf_scroll(PyObject *self, PyObject *args, PyObject *keywds)
{
    int dx = 0, dy = 0;
    SDL_Surface *surf;
    int bpp;
    int pitch;
    SDL_Rect *clip_rect;
    int w, h;
    Uint8 *src, *dst;

    static char *kwids[] = {"dx", "dy", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|ii", kwids, &dx, &dy)) {
        return NULL;
    }

    surf = pgSurface_AsSurface(self);
    if (!surf) {
        return RAISE(pgExc_SDLError, "display Surface quit");
    }

#if IS_SDLv1
    if (surf->flags & SDL_OPENGL &&
        !(surf->flags & (SDL_OPENGLBLIT & ~SDL_OPENGL))) {
        return RAISE(pgExc_SDLError,
                     "Cannot scroll an OPENGL Surfaces (OPENGLBLIT is ok)");
    }
#endif /* IS_SDLv1 */

    if (dx == 0 && dy == 0) {
        Py_RETURN_NONE;
    }

    clip_rect = &surf->clip_rect;
    w = clip_rect->w;
    h = clip_rect->h;
    if (dx >= w || dx <= -w || dy >= h || dy <= -h) {
        Py_RETURN_NONE;
    }

    if (!pgSurface_Lock(self)) {
        return NULL;
    }

    bpp = surf->format->BytesPerPixel;
    pitch = surf->pitch;
    src = dst =
        (Uint8 *)surf->pixels + clip_rect->y * pitch + clip_rect->x * bpp;
    if (dx >= 0) {
        w -= dx;
        if (dy > 0) {
            h -= dy;
            dst += dy * pitch + dx * bpp;
        }
        else {
            h += dy;
            src -= dy * pitch;
            dst += dx * bpp;
        }
    }
    else {
        w += dx;
        if (dy > 0) {
            h -= dy;
            src -= dx * bpp;
            dst += dy * pitch;
        }
        else {
            h += dy;
            src -= dy * pitch + dx * bpp;
        }
    }
    surface_move(src, dst, h, w * bpp, pitch, pitch);

    if (!pgSurface_Unlock(self)) {
        return NULL;
    }

    Py_RETURN_NONE;
}


#if IS_SDLv2
static int
_PgSurface_SrcAlpha(SDL_Surface *surf)
{
    SDL_BlendMode mode;
    if (SDL_GetSurfaceBlendMode(surf, &mode) < 0) {
        RAISE(pgExc_SDLError, SDL_GetError());
        return -1;
    }
    return (mode != SDL_BLENDMODE_NONE);
}
#endif /* IS_SDLv2 */

static PyObject *
surf_get_flags(PyObject *self)
{
#if IS_SDLv2
    Uint32 sdl_flags = 0;
    Uint32 flags = 0;
    int is_alpha;
#endif /* IS_SDLv2 */

    SDL_Surface *surf = pgSurface_AsSurface(self);

    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");
#if IS_SDLv1
    return PyInt_FromLong((long)surf->flags);
#else  /* IS_SDLv2 */
    sdl_flags = surf->flags;
    if ((is_alpha = _PgSurface_SrcAlpha(surf)) == -1)
        return NULL;
    if (is_alpha)
        flags |= PGS_SRCALPHA;
    if (SDL_GetColorKey(surf, NULL) == 0)
        flags |= PGS_SRCCOLORKEY;
    if (sdl_flags & SDL_PREALLOC)
        flags |= PGS_PREALLOC;
    if (sdl_flags & SDL_RLEACCEL)
        flags |= PGS_RLEACCEL;
    return PyInt_FromLong((long)flags);
#endif /* IS_SDLv2 */
}

static PyObject *
surf_get_pitch(PyObject *self)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);

    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");
    return PyInt_FromLong(surf->pitch);
}

static PyObject *
surf_get_size(PyObject *self)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);

    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");
    return Py_BuildValue("(ii)", surf->w, surf->h);
}

static PyObject *
surf_get_width(PyObject *self)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);

    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");
    return PyInt_FromLong(surf->w);
}

static PyObject *
surf_get_height(PyObject *self)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);

    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");
    return PyInt_FromLong(surf->h);
}

static PyObject *
surf_get_rect(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *rect;
    SDL_Surface *surf = pgSurface_AsSurface(self);

    if (PyTuple_GET_SIZE(args) > 0) {
        return RAISE(PyExc_TypeError,
                     "get_rect only accepts keyword arguments");
    }

    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

    rect = pgRect_New4(0, 0, surf->w, surf->h);
    if (rect && kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;

        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if ((PyObject_SetAttr(rect, key, value) == -1)) {
                Py_DECREF(rect);
                return NULL;
            }
        }
    }
    return rect;
}

static PyObject *
surf_get_bitsize(PyObject *self)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

    return PyInt_FromLong(surf->format->BitsPerPixel);
}

static PyObject *
surf_get_bytesize(PyObject *self)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");
    return PyInt_FromLong(surf->format->BytesPerPixel);
}

static PyObject *
surf_get_masks(PyObject *self)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);

    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");
    return Py_BuildValue("(IIII)", surf->format->Rmask, surf->format->Gmask,
                         surf->format->Bmask, surf->format->Amask);
}

static PyObject *
surf_set_masks(PyObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    /* Need to use 64bit vars so this works on 64 bit pythons. */
    unsigned long r, g, b, a;

    if (!PyArg_ParseTuple(args, "(kkkk)", &r, &g, &b, &a))
        return NULL;
    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

    /*
    printf("passed in: %d, %d, %d, %d\n", r,g,b,a );
    printf("what are: %d, %d, %d, %d\n", surf->format->Rmask,
    surf->format->Gmask, surf->format->Bmask, surf->format->Amask);
    */

    surf->format->Rmask = (Uint32)r;
    surf->format->Gmask = (Uint32)g;
    surf->format->Bmask = (Uint32)b;
    surf->format->Amask = (Uint32)a;

    Py_RETURN_NONE;
}

static PyObject *
surf_get_shifts(PyObject *self)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);

    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");
    return Py_BuildValue("(iiii)", surf->format->Rshift, surf->format->Gshift,
                         surf->format->Bshift, surf->format->Ashift);
}

static PyObject *
surf_set_shifts(PyObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    unsigned long r, g, b, a;

    if (!PyArg_ParseTuple(args, "(kkkk)", &r, &g, &b, &a))
        return NULL;
    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

    surf->format->Rshift = (Uint8)r;
    surf->format->Gshift = (Uint8)g;
    surf->format->Bshift = (Uint8)b;
    surf->format->Ashift = (Uint8)a;

    Py_RETURN_NONE;
}

static PyObject *
surf_get_losses(PyObject *self)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);

    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");
    return Py_BuildValue("(iiii)", surf->format->Rloss, surf->format->Gloss,
                         surf->format->Bloss, surf->format->Aloss);
}

static PyObject *
surf_subsurface(PyObject *self, PyObject *args)
{
    SDL_Surface *surf = pgSurface_AsSurface(self);
    SDL_PixelFormat *format;
    GAME_Rect *rect, temp;
    SDL_Surface *sub;
    PyObject *subobj;
    int pixeloffset;
    char *startpixel;
    struct pgSubSurface_Data *data;
#if IS_SDLv2
    Uint8 alpha;
    Uint32 colorkey;
    int ecode;
#endif /* IS_SDLv2 */

    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");
#if IS_SDLv1
    if (surf->flags & SDL_OPENGL)
        return RAISE(pgExc_SDLError, "Cannot call on OPENGL Surfaces");
#endif /* IS_SDLv1 */

    format = surf->format;
    if (!(rect = pgRect_FromObject(args, &temp)))
        return RAISE(PyExc_ValueError, "invalid rectstyle argument");
    if (rect->x < 0 || rect->y < 0 || rect->x + rect->w > surf->w ||
        rect->y + rect->h > surf->h)
        return RAISE(PyExc_ValueError,
                     "subsurface rectangle outside surface area");

    pgSurface_Lock(self);

    pixeloffset = rect->x * format->BytesPerPixel + rect->y * surf->pitch;
    startpixel = ((char *)surf->pixels) + pixeloffset;

    sub = SDL_CreateRGBSurfaceFrom(
        startpixel, rect->w, rect->h, format->BitsPerPixel, surf->pitch,
        format->Rmask, format->Gmask, format->Bmask, format->Amask);

    pgSurface_Unlock(self);

#if IS_SDLv1
    if (!sub)
        return RAISE(pgExc_SDLError, SDL_GetError());

    /* copy the colormap if we need it */
    if (surf->format->BytesPerPixel == 1 && surf->format->palette)
        SDL_SetPalette(sub, SDL_LOGPAL, surf->format->palette->colors, 0,
                       surf->format->palette->ncolors);
    if (surf->flags & SDL_SRCALPHA)
        SDL_SetAlpha(sub, surf->flags & SDL_SRCALPHA, format->alpha);
    if (surf->flags & SDL_SRCCOLORKEY)
        SDL_SetColorKey(sub, surf->flags & (SDL_SRCCOLORKEY | SDL_RLEACCEL),
                        format->colorkey);
#else  /* IS_SDLv2 */
    if (!sub)
        return _raise_create_surface_error();

    /* copy the colormap if we need it */
    if (SDL_ISPIXELFORMAT_INDEXED(surf->format->format) &&
        surf->format->palette) {
        SDL_Color *colors = surf->format->palette->colors;
        int ncolors = surf->format->palette->ncolors;
        SDL_Palette *pal = SDL_AllocPalette(ncolors);

        if (!pal) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(sub);
            return NULL;
        }
        if (SDL_SetPaletteColors(pal, colors, 0, ncolors) != 0) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreePalette(pal);
            SDL_FreeSurface(sub);
            return NULL;
        }
        if (SDL_SetSurfacePalette(sub, pal) != 0) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreePalette(pal);
            SDL_FreeSurface(sub);
            return NULL;
        }
        SDL_FreePalette(pal);
    }
    if (SDL_GetSurfaceAlphaMod(surf, &alpha) != 0) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        SDL_FreeSurface(sub);
        return NULL;
    }
    if (alpha != 255) {
        if (SDL_SetSurfaceAlphaMod(sub, alpha) != 0) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(sub);
            return NULL;
        }
    }
    ecode = SDL_GetColorKey(surf, &colorkey);
    if (ecode == 0) {
        if (SDL_SetColorKey(sub, SDL_TRUE, colorkey) != 0) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(sub);
            return NULL;
        }
    }
    else if (ecode == -1)
        SDL_ClearError();
    else {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        SDL_FreeSurface(sub);
        return NULL;
    }
#endif /* IS_SDLv2 */

    data = PyMem_New(struct pgSubSurface_Data, 1);
    if (!data)
        return NULL;

#if IS_SDLv1
    subobj = surf_subtype_new(Py_TYPE(self), sub);
#else  /* IS_SDLv2 */
    subobj = surf_subtype_new(Py_TYPE(self), sub, 1);
#endif /* IS_SDLv2 */
    if (!subobj) {
        PyMem_Del(data);
        return NULL;
    }
    Py_INCREF(self);
    data->owner = self;
    data->pixeloffset = pixeloffset;
    data->offsetx = rect->x;
    data->offsety = rect->y;
    ((pgSurfaceObject *)subobj)->subsurface = data;

    return subobj;
}

static PyObject *
surf_get_offset(PyObject *self)
{
    struct pgSubSurface_Data *subdata;

    subdata = ((pgSurfaceObject *)self)->subsurface;
    if (!subdata)
        return Py_BuildValue("(ii)", 0, 0);
    return Py_BuildValue("(ii)", subdata->offsetx, subdata->offsety);
}

static PyObject *
surf_get_abs_offset(PyObject *self)
{
    struct pgSubSurface_Data *subdata;
    PyObject *owner;
    int offsetx, offsety;

    subdata = ((pgSurfaceObject *)self)->subsurface;
    if (!subdata)
        return Py_BuildValue("(ii)", 0, 0);

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
    return Py_BuildValue("(ii)", offsetx, offsety);
}

static PyObject *
surf_get_parent(PyObject *self)
{
    struct pgSubSurface_Data *subdata;

    subdata = ((pgSurfaceObject *)self)->subsurface;
    if (!subdata)
        Py_RETURN_NONE;

    Py_INCREF(subdata->owner);
    return subdata->owner;
}

static PyObject *
surf_get_abs_parent(PyObject *self)
{
    struct pgSubSurface_Data *subdata;
    PyObject *owner;

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
    SDL_PixelFormat *format = surf->format;
    Uint8 *pixels = (Uint8 *)surf->pixels;
    Uint8 *pixel;
    int x, y;
    int min_x, min_y, max_x, max_y;
    int min_alpha = 1;
    int found_alpha = 0;
    Uint32 value;
    Uint8 r, g, b, a;
    int has_colorkey = 0;
#if IS_SDLv2
    Uint32 colorkey;
#endif /* IS_SDLv2 */
    Uint8 keyr, keyg, keyb;

    char *kwids[] = {"min_alpha", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", kwids, &min_alpha))
        return RAISE(PyExc_ValueError,
                     "get_bounding_rect only accepts a single optional "
                     "min_alpha argument");

    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

    if (!pgSurface_Lock(self))
        return RAISE(pgExc_SDLError, "could not lock surface");

#if IS_SDLv1
    if (surf->flags & SDL_SRCCOLORKEY) {
        has_colorkey = 1;
        SDL_GetRGBA(surf->format->colorkey, surf->format, &keyr, &keyg, &keyb,
                    &a);
    }
#else  /* IS_SDLv2 */
    if (SDL_GetColorKey(surf, &colorkey) == 0) {
        has_colorkey = 1;
        SDL_GetRGBA(colorkey, surf->format, &keyr, &keyg, &keyb, &a);
    }
#endif /* IS_SDLv2 */

    pixels = (Uint8 *)surf->pixels;
    min_y = 0;
    min_x = 0;
    max_x = surf->w;
    max_y = surf->h;

    found_alpha = 0;
    for (y = max_y - 1; y >= min_y; --y) {
        for (x = min_x; x < max_x; ++x) {
            pixel = (pixels + y * surf->pitch) + x * format->BytesPerPixel;
            switch (format->BytesPerPixel) {
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
                    assert(format->BytesPerPixel == 4);
                    value = *(Uint32 *)pixel;
            }
            SDL_GetRGBA(value, surf->format, &r, &g, &b, &a);
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
            pixel = (pixels + y * surf->pitch) + x * format->BytesPerPixel;
            switch (format->BytesPerPixel) {
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
                    assert(format->BytesPerPixel == 4);
                    value = *(Uint32 *)pixel;
            }
            SDL_GetRGBA(value, surf->format, &r, &g, &b, &a);
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
            pixel = (pixels + y * surf->pitch) + x * format->BytesPerPixel;
            switch (format->BytesPerPixel) {
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
                    assert(format->BytesPerPixel == 4);
                    value = *(Uint32 *)pixel;
            }
            SDL_GetRGBA(value, surf->format, &r, &g, &b, &a);
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
            pixel = (pixels + y * surf->pitch) + x * format->BytesPerPixel;
            switch (format->BytesPerPixel) {
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
                    assert(format->BytesPerPixel == 4);
                    value = *(Uint32 *)pixel;
            }
            SDL_GetRGBA(value, surf->format, &r, &g, &b, &a);
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
    if (!pgSurface_Unlock(self))
        return RAISE(pgExc_SDLError, "could not unlock surface");

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
            name = "contigous pixels";
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
    SDL_PixelFormat *format;
    Uint32 mask = 0;
    SurfViewKind view_kind = VIEWKIND_2D;
    getbufferproc get_buffer = 0;

    if (!PyArg_ParseTuple(args, "|O&", _view_kind, &view_kind)) {
        return 0;
    }

    if (!surface) {
        return RAISE(pgExc_SDLError, "display Surface quit");
    }

    format = surface->format;
    switch (view_kind) {
            /* This switch statement is exhaustive over the SurfViewKind enum
             */

        case VIEWKIND_0D:
            if (surface->pitch != format->BytesPerPixel * surface->w) {
                PyErr_SetString(PyExc_ValueError,
                                "Surface data is not contiguous");
                return 0;
            }
            get_buffer = _get_buffer_0D;
            break;
        case VIEWKIND_1D:
            if (surface->pitch != format->BytesPerPixel * surface->w) {
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
            if (format->BytesPerPixel < 3) {
                return _raise_get_view_ndim_error(format->BytesPerPixel * 8,
                                                  view_kind);
            }
            if (format->Gmask != 0x00ff00 &&
                (format->BytesPerPixel != 4 || format->Gmask != 0xff0000)) {
                return RAISE(PyExc_ValueError,
                             "unsupport colormasks for 3D reference array");
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
    return pgBufproxy_New(self, get_buffer);
}

static PyObject *
surf_get_buffer(PyObject *self)
{
    SDL_Surface *surface = pgSurface_AsSurface(self);
    PyObject *proxy_obj;

    if (!surface) {
        return RAISE(pgExc_SDLError, "display Surface quit");
    }

    proxy_obj = pgBufproxy_New(self, _get_buffer_0D);
    if (proxy_obj) {
        if (pgBufproxy_Trip(proxy_obj)) {
            Py_DECREF(proxy_obj);
            proxy_obj = 0;
        }
    }
    return proxy_obj;
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
    view_p->len = surface->pitch * surface->h;
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
    Py_ssize_t itemsize = surface->format->BytesPerPixel;

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
                             "unknown pixel size %i. Please report",
                             (int)__LINE__, __FILE__, itemsize);
                return -1;
#endif
        }
    }
    view_p->buf = surface->pixels;
    view_p->itemsize = itemsize;
    view_p->readonly = 0;
    view_p->len = surface->pitch * surface->h;
    if (PyBUF_HAS_FLAG(flags, PyBUF_ND)) {
        view_p->ndim = 1;
        view_p->shape[0] = surface->w * surface->h;
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
    int itemsize = surface->format->BytesPerPixel;

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
    view_p->len = surface->w * surface->h * itemsize;
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
    const int lilendian = (SDL_BYTEORDER == SDL_LIL_ENDIAN);
    SDL_Surface *surface = pgSurface_AsSurface(obj);
    int pixelsize = surface->format->BytesPerPixel;
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
    if (_init_buffer(obj, view_p, flags)) {
        return -1;
    }
    if (PyBUF_HAS_FLAG(flags, PyBUF_FORMAT)) {
        view_p->format = FormatUint8;
    }
    view_p->itemsize = 1;
    view_p->ndim = 3;
    view_p->readonly = 0;
    view_p->len = surface->w * surface->h * 3;
    view_p->shape[0] = surface->w;
    view_p->shape[1] = surface->h;
    view_p->shape[2] = 3;
    view_p->strides[0] = pixelsize;
    view_p->strides[1] = surface->pitch;
    switch (surface->format->Rmask) {
        case 0xffU:
            view_p->strides[2] = lilendian ? 1 : -1;
            startpixel += lilendian ? 0 : pixelsize - 1;
            break;
        case 0xff00U:
            assert(pixelsize == 4);
            view_p->strides[2] = lilendian ? 1 : -1;
            startpixel += lilendian ? 1 : pixelsize - 2;
            break;
        case 0xff0000U:
            view_p->strides[2] = lilendian ? -1 : 1;
            startpixel += lilendian ? 2 : pixelsize - 3;
            break;
        default: /* 0xff000000U */
            assert(pixelsize == 4);
            view_p->strides[2] = lilendian ? -1 : 1;
            startpixel += lilendian ? 3 : 0;
    }
    view_p->buf = startpixel;
    Py_INCREF(obj);
    view_p->obj = obj;
    return 0;
}

static int
_get_buffer_red(PyObject *obj, Py_buffer *view_p, int flags)
{
    return _get_buffer_colorplane(obj, view_p, flags, "red",
                                  pgSurface_AsSurface(obj)->format->Rmask);
}

static int
_get_buffer_green(PyObject *obj, Py_buffer *view_p, int flags)
{
    return _get_buffer_colorplane(obj, view_p, flags, "green",
                                  pgSurface_AsSurface(obj)->format->Gmask);
}

static int
_get_buffer_blue(PyObject *obj, Py_buffer *view_p, int flags)
{
    return _get_buffer_colorplane(obj, view_p, flags, "blue",
                                  pgSurface_AsSurface(obj)->format->Bmask);
}

static int
_get_buffer_alpha(PyObject *obj, Py_buffer *view_p, int flags)
{
    return _get_buffer_colorplane(obj, view_p, flags, "alpha",
                                  pgSurface_AsSurface(obj)->format->Amask);
}

static int
_get_buffer_colorplane(PyObject *obj, Py_buffer *view_p, int flags, char *name,
                       Uint32 mask)
{
    const int lilendian = (SDL_BYTEORDER == SDL_LIL_ENDIAN);
    SDL_Surface *surface = pgSurface_AsSurface(obj);
    int pixelsize = surface->format->BytesPerPixel;
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

        case 0x000000ffU:
            startpixel += lilendian ? 0 : pixelsize - 1;
            break;
        case 0x0000ff00U:
            startpixel += lilendian ? 1 : pixelsize - 2;
            break;
        case 0x00ff0000U:
            startpixel += lilendian ? 2 : pixelsize - 3;
            break;
        case 0xff000000U:
            startpixel += lilendian ? 3 : 0;
            break;

#ifndef NDEBUG
            /* Assert this switch statement is exhaustive */
        default:
            /* Should not be here */
            PyErr_Format(PyExc_SystemError,
                         "Pygame bug caught at line %i in file %s: "
                         "unknown mask value %p. Please report",
                         (int)__LINE__, __FILE__, (void *)mask);
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
    view_p->len = surface->w * surface->h;
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
    if (!pgSurface_LockBy(surf, consumer)) {
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
    PyObject *consumer;

    assert(view_p && view_p->obj && view_p->internal);
    internal = (pg_bufferinternal *)view_p->internal;
    consumer_ref = internal->consumer_ref;
    assert(consumer_ref && PyWeakref_CheckRef(consumer_ref));
    consumer = PyWeakref_GetObject(consumer_ref);
    if (consumer) {
        if (!pgSurface_UnlockBy(view_p->obj, consumer)) {
            PyErr_Clear();
        }
    }
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
        if (PyUnicode_GET_SIZE(obj) != 1) {
            PyErr_SetString(PyExc_TypeError,
                            "expected a length 1 string for argument 1");
            return 0;
        }
        ch = *PyUnicode_AS_UNICODE(obj);
    }
    else if (Bytes_Check(obj)) {
        if (Bytes_GET_SIZE(obj) != 1) {
            PyErr_SetString(PyExc_TypeError,
                            "expected a length 1 string for argument 1");
            return 0;
        }
        ch = *Bytes_AS_STRING(obj);
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

static void
surface_move(Uint8 *src, Uint8 *dst, int h, int span, int srcpitch,
             int dstpitch)
{
    if (src < dst) {
        src += (h - 1) * srcpitch;
        dst += (h - 1) * dstpitch;
        srcpitch = -srcpitch;
        dstpitch = -dstpitch;
    }
    while (h--) {
        memmove(dst, src, span);
        src += srcpitch;
        dst += dstpitch;
    }
}

static int
surface_do_overlap(SDL_Surface *src, SDL_Rect *srcrect, SDL_Surface *dst,
                   SDL_Rect *dstrect)
{
    Uint8 *srcpixels;
    Uint8 *dstpixels;
    int srcx = srcrect->x, srcy = srcrect->y;
    int dstx = dstrect->x, dsty = dstrect->y;
    int x, y;
    int w = srcrect->w, h = srcrect->h;
    int maxw, maxh;
    SDL_Rect *clip = &dst->clip_rect;
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
    srcy = srcrect->y;
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

#if IS_SDLv1
    srcpixels = ((Uint8 *)src->pixels + src->offset + srcy * src->pitch +
                 srcx * src->format->BytesPerPixel);
    dstpixels = ((Uint8 *)dst->pixels + src->offset + dsty * dst->pitch +
                 dstx * dst->format->BytesPerPixel);
#else  /* IS_SDLv2 */
    srcpixels = ((Uint8 *)src->pixels + srcy * src->pitch +
                 srcx * src->format->BytesPerPixel);
    dstpixels = ((Uint8 *)dst->pixels + dsty * dst->pitch +
                 dstx * dst->format->BytesPerPixel);
#endif /* IS_SDLv2 */

    if (dstpixels <= srcpixels) {
        return 0;
    }

    span = w * src->format->BytesPerPixel;

    if (dstpixels >= srcpixels + (h - 1) * src->pitch + span) {
        return 0;
    }

    dstoffset = (dstpixels - srcpixels) % src->pitch;

    return dstoffset < span || dstoffset > src->pitch - span;
}

/*this internal blit function is accessable through the C api*/
int
pgSurface_Blit(PyObject *dstobj, PyObject *srcobj, SDL_Rect *dstrect,
               SDL_Rect *srcrect, int the_args)
{
    SDL_Surface *src = pgSurface_AsSurface(srcobj);
    SDL_Surface *dst = pgSurface_AsSurface(dstobj);
    SDL_Surface *subsurface = NULL;
    int result, suboffsetx = 0, suboffsety = 0;
    SDL_Rect orig_clip, sub_clip;
#if IS_SDLv2
    Uint8 alpha;
    Uint32 key;
#endif /* IS_SDLv2 */

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

#if IS_SDLv1
    /* This test fails if this first condition is not used.
       File "test/surface_test.py", in test_pixel_alpha
    */
    if (dst->format->Amask && (dst->flags & SDL_SRCALPHA) &&
        !(src->format->Amask && !(src->flags & SDL_SRCALPHA)) &&
        /* special case, SDL works */
        (dst->format->BytesPerPixel == 2 || dst->format->BytesPerPixel == 4)) {
        /* Py_BEGIN_ALLOW_THREADS */
        result = pygame_AlphaBlit(src, srcrect, dst, dstrect, the_args);
        /* Py_END_ALLOW_THREADS */
    }
    else if (the_args != 0 ||
             (src->flags & (SDL_SRCALPHA | SDL_SRCCOLORKEY) &&
              /* This simplification is possible because a source subsurface
                 is converted to its owner with a clip rect and a dst
                 subsurface cannot be blitted to its owner because the
                 owner is locked.
                 */
              dst->pixels == src->pixels &&
              surface_do_overlap(src, srcrect, dst, dstrect))) {
        /* Py_BEGIN_ALLOW_THREADS */
        result = pygame_Blit(src, srcrect, dst, dstrect, the_args);
        /* Py_END_ALLOW_THREADS */
    }
    /* can't blit alpha to 8bit, crashes SDL */
    else if (dst->format->BytesPerPixel == 1 &&
             (src->format->Amask || src->flags & SDL_SRCALPHA)) {
        /* Py_BEGIN_ALLOW_THREADS */
        if (src->format->BytesPerPixel == 1) {
            result = pygame_Blit(src, srcrect, dst, dstrect, 0);
        }
        else {
            SDL_PixelFormat *fmt = src->format;
            SDL_PixelFormat newfmt;

            newfmt.palette = 0; /* Set NULL (or SDL gets confused) */
            newfmt.BitsPerPixel = fmt->BitsPerPixel;
            newfmt.BytesPerPixel = fmt->BytesPerPixel;
            newfmt.Amask = 0;
            newfmt.Rmask = fmt->Rmask;
            newfmt.Gmask = fmt->Gmask;
            newfmt.Bmask = fmt->Bmask;
            newfmt.Ashift = 0;
            newfmt.Rshift = fmt->Rshift;
            newfmt.Gshift = fmt->Gshift;
            newfmt.Bshift = fmt->Bshift;
            newfmt.Aloss = 0;
            newfmt.Rloss = fmt->Rloss;
            newfmt.Gloss = fmt->Gloss;
            newfmt.Bloss = fmt->Bloss;
            newfmt.colorkey = 0;
            newfmt.alpha = 0;
            src = SDL_ConvertSurface(src, &newfmt, SDL_SWSURFACE);
            if (src) {
                result = SDL_BlitSurface(src, srcrect, dst, dstrect);
                SDL_FreeSurface(src);
            }
            else {
                result = -1;
            }
        }
        /* Py_END_ALLOW_THREADS */
    }
#else  /* IS_SDLv2 */
    if (the_args != 0 ||
             ((SDL_GetColorKey(src, &key) == 0 ||
               _PgSurface_SrcAlpha(src) == 1) &&
              /* This simplification is possible because a source subsurface
                 is converted to its owner with a clip rect and a dst
                 subsurface cannot be blitted to its owner because the
                 owner is locked.
                 */
              dst->pixels == src->pixels &&
              surface_do_overlap(src, srcrect, dst, dstrect))) {
        /* Py_BEGIN_ALLOW_THREADS */
        result = pygame_Blit(src, srcrect, dst, dstrect, the_args);
        /* Py_END_ALLOW_THREADS */
    }
    /* can't blit alpha to 8bit, crashes SDL */
    else if (dst->format->BytesPerPixel == 1 &&
             (SDL_ISPIXELFORMAT_ALPHA(src->format->format) ||
              ((SDL_GetSurfaceAlphaMod(src, &alpha) == 0 && alpha != 255)))) {
        /* Py_BEGIN_ALLOW_THREADS */
        if (src->format->BytesPerPixel == 1) {
            result = pygame_Blit(src, srcrect, dst, dstrect, 0);
        }
        else {
            SDL_PixelFormat *fmt = src->format;
            SDL_PixelFormat newfmt;

            newfmt.palette = 0; /* Set NULL (or SDL gets confused) */
            newfmt.BitsPerPixel = fmt->BitsPerPixel;
            newfmt.BytesPerPixel = fmt->BytesPerPixel;
            newfmt.Amask = 0;
            newfmt.Rmask = fmt->Rmask;
            newfmt.Gmask = fmt->Gmask;
            newfmt.Bmask = fmt->Bmask;
            newfmt.Ashift = 0;
            newfmt.Rshift = fmt->Rshift;
            newfmt.Gshift = fmt->Gshift;
            newfmt.Bshift = fmt->Bshift;
            newfmt.Aloss = 0;
            newfmt.Rloss = fmt->Rloss;
            newfmt.Gloss = fmt->Gloss;
            newfmt.Bloss = fmt->Bloss;
            src = SDL_ConvertSurface(src, &newfmt, 0);
            if (src) {
                result = SDL_BlitSurface(src, srcrect, dst, dstrect);
                SDL_FreeSurface(src);
            }
            else {
                result = -1;
            }
        }
        /* Py_END_ALLOW_THREADS */
    }
#endif /* IS_SDLv2 */
    else {
        /* Py_BEGIN_ALLOW_THREADS */
        result = SDL_BlitSurface(src, srcrect, dst, dstrect);
        /* Py_END_ALLOW_THREADS */
    }

    if (subsurface) {
        SDL_SetClipRect(subsurface, &orig_clip);
        dstrect->x -= suboffsetx;
        dstrect->y -= suboffsety;
    }
    else
        pgSurface_Unprep(dstobj);
    pgSurface_Unprep(srcobj);

    if (result == -1)
        RAISE(pgExc_SDLError, SDL_GetError());
    if (result == -2)
        RAISE(pgExc_SDLError, "Surface was lost");

    return result != 0;
}

static PyMethodDef _surface_methods[] = {{NULL, NULL, 0, NULL}};

MODINIT_DEFINE(surface)
{
    PyObject *module, *dict, *apiobj, *lockmodule;
    int ecode;
    static void *c_api[PYGAMEAPI_SURFACE_NUMSLOTS];

#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "surface",
                                         DOC_PYGAMESURFACE,
                                         -1,
                                         _surface_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};
#endif

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
    import_pygame_color();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
    import_pygame_rect();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
    import_pygame_bufferproxy();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    /* import the surflock module manually */
    lockmodule = PyImport_ImportModule(IMPPREFIX "surflock");
    if (lockmodule != NULL) {
        PyObject *_dict = PyModule_GetDict(lockmodule);
        PyObject *_c_api = PyDict_GetItemString(_dict, PYGAMEAPI_LOCAL_ENTRY);

        if (PyCapsule_CheckExact(_c_api)) {
            int i;
            void **localptr = (void *)PyCapsule_GetPointer(
                _c_api, PG_CAPSULE_NAME("surflock"));

            for (i = 0; i < PYGAMEAPI_SURFLOCK_NUMSLOTS; ++i)
                PyGAME_C_API[i + PYGAMEAPI_SURFLOCK_FIRSTSLOT] = localptr[i];
        }
        Py_DECREF(lockmodule);
    }
    else {
        MODINIT_ERROR;
    }

    /* type preparation */
    if (PyType_Ready(&pgSurface_Type) < 0) {
        MODINIT_ERROR;
    }

    /* create the module */
#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3(MODPREFIX "surface", _surface_methods,
                            DOC_PYGAMESURFACE);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    dict = PyModule_GetDict(module);

    if (PyDict_SetItemString(dict, "SurfaceType",
                             (PyObject *)&pgSurface_Type)) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    if (PyDict_SetItemString(dict, "Surface", (PyObject *)&pgSurface_Type)) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    /* export the c api */
    c_api[0] = &pgSurface_Type;
    c_api[1] = pgSurface_New;
    c_api[2] = pgSurface_Blit;
    apiobj = encapsulate_api(c_api, "surface");
    if (apiobj == NULL) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF(apiobj);
    if (ecode) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    /* Py_INCREF (pgSurface_Type.tp_dict); INCREF's done in SetItemString */
    if (PyDict_SetItemString(dict, "_dict", pgSurface_Type.tp_dict)) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN(module);
}
