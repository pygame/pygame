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
#include "doc/surface_doc.h"
#include "structmember.h"
#include "pgcompat.h"
#include "pgview.h"
#include "pgarrinter.h"

typedef enum {
    VIEWKIND_0D = 0,
    VIEWKIND_1D = 1,
    VIEWKIND_2D = 2,
    VIEWKIND_3D = 3,
    VIEWKIND_RED,
    VIEWKIND_GREEN,
    VIEWKIND_BLUE,
    VIEWKIND_ALPHA,
    VIEWKIND_RAW
} SurfViewKind;

int
PySurface_Blit (PyObject * dstobj, PyObject * srcobj, SDL_Rect * dstrect,
                SDL_Rect * srcrect, int the_args);

/* statics */
static PyObject *PySurface_New (SDL_Surface * info);
static PyObject *surface_new (PyTypeObject *type, PyObject *args,
                              PyObject *kwds);
static intptr_t surface_init (PySurfaceObject *self, PyObject *args,
                              PyObject *kwds);
static PyObject* surface_str (PyObject *self);
static void surface_dealloc (PyObject *self);
static void surface_cleanup (PySurfaceObject * self);
static void surface_move (Uint8 *src, Uint8 *dst, int h,
              int span, int srcpitch, int dstpitch);

static PyObject *surf_get_at (PyObject *self, PyObject *args);
static PyObject *surf_set_at (PyObject *self, PyObject *args);
static PyObject *surf_get_at_mapped (PyObject *self, PyObject *args);
static PyObject *surf_map_rgb (PyObject *self, PyObject *args);
static PyObject *surf_unmap_rgb (PyObject *self, PyObject *arg);
static PyObject *surf_lock (PyObject *self);
static PyObject *surf_unlock (PyObject *self);
static PyObject *surf_mustlock (PyObject *self);
static PyObject *surf_get_locked (PyObject *self);
static PyObject *surf_get_locks (PyObject *self);
static PyObject *surf_get_palette (PyObject *self);
static PyObject *surf_get_palette_at (PyObject *self, PyObject *args);
static PyObject *surf_set_palette (PyObject *self, PyObject *args);
static PyObject *surf_set_palette_at (PyObject *self, PyObject *args);
static PyObject *surf_set_colorkey (PyObject *self, PyObject *args);
static PyObject *surf_get_colorkey (PyObject *self);
static PyObject *surf_set_alpha (PyObject *self, PyObject *args);
static PyObject *surf_get_alpha (PyObject *self);
static PyObject *surf_copy (PyObject *self);
static PyObject *surf_convert (PyObject *self, PyObject *args);
static PyObject *surf_convert_alpha (PyObject *self, PyObject *args);
static PyObject *surf_set_clip (PyObject *self, PyObject *args);
static PyObject *surf_get_clip (PyObject *self);
static PyObject *surf_blit (PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *surf_fill (PyObject *self, PyObject *args, PyObject *keywds);
static PyObject *surf_scroll (PyObject *self,
                              PyObject *args, PyObject *keywds);
static PyObject *surf_get_abs_offset (PyObject *self);
static PyObject *surf_get_abs_parent (PyObject *self);
static PyObject *surf_get_bitsize (PyObject *self);
static PyObject *surf_get_bytesize (PyObject *self);
static PyObject *surf_get_flags (PyObject *self);
static PyObject *surf_get_height (PyObject *self);
static PyObject *surf_get_pitch (PyObject *self);
static PyObject *surf_get_rect (PyObject *self, PyObject *args,
                                PyObject *kwargs);
static PyObject *surf_get_width (PyObject *self);
static PyObject *surf_get_shifts (PyObject *self);
static PyObject *surf_set_shifts (PyObject *self, PyObject *args);
static PyObject *surf_get_size (PyObject *self);
static PyObject *surf_get_losses (PyObject *self);
static PyObject *surf_get_masks (PyObject *self);
static PyObject *surf_set_masks (PyObject *self, PyObject *args);
static PyObject *surf_get_offset (PyObject *self);
static PyObject *surf_get_parent (PyObject *self);
static PyObject *surf_subsurface (PyObject *self, PyObject *args);
static PyObject *surf_get_buffer (PyObject *self);
static PyObject *surf_get_view (PyObject *self, PyObject *args, PyObject *kwds);
static PyObject *surf_get_bounding_rect (PyObject *self, PyObject *args,
                                         PyObject *kwargs);
static PyObject *surf_get_pixels_address (PyObject *self,
                                          PyObject *closure);
static int _view_kind(PyObject *obj, void *view_kind_vptr);
static int _view_before(PyObject *view);
static void _view_after(PyObject *view);
static PyObject *_raise_get_view_ndim_error(int bitsize, SurfViewKind kind);


static PyGetSetDef surface_getsets[] = {
    { "_pixels_address", (getter)surf_get_pixels_address,
      NULL, "pixel buffer address (readonly)", NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

static struct PyMethodDef surface_methods[] = {
    { "get_at", surf_get_at, METH_VARARGS, DOC_SURFACEGETAT },
    { "set_at", surf_set_at, METH_VARARGS, DOC_SURFACESETAT },
    { "get_at_mapped", surf_get_at_mapped, METH_VARARGS,
      DOC_SURFACEGETATMAPPED },
    { "map_rgb", surf_map_rgb, METH_VARARGS, DOC_SURFACEMAPRGB },
    { "unmap_rgb", surf_unmap_rgb, METH_O, DOC_SURFACEUNMAPRGB },

    { "get_palette", (PyCFunction) surf_get_palette, METH_NOARGS,
      DOC_SURFACEGETPALETTE },
    { "get_palette_at", surf_get_palette_at, METH_VARARGS,
      DOC_SURFACEGETPALETTEAT },
    { "set_palette", surf_set_palette, METH_VARARGS, DOC_SURFACESETPALETTE },
    { "set_palette_at", surf_set_palette_at, METH_VARARGS,
      DOC_SURFACESETPALETTEAT },

    { "lock", (PyCFunction) surf_lock, METH_NOARGS, DOC_SURFACELOCK },
    { "unlock", (PyCFunction) surf_unlock, METH_NOARGS, DOC_SURFACEUNLOCK },
    { "mustlock", (PyCFunction) surf_mustlock, METH_NOARGS,
      DOC_SURFACEMUSTLOCK },
    { "get_locked", (PyCFunction) surf_get_locked, METH_NOARGS,
      DOC_SURFACEGETLOCKED },
    { "get_locks", (PyCFunction) surf_get_locks, METH_NOARGS,
      DOC_SURFACEGETLOCKS },

    { "set_colorkey", surf_set_colorkey, METH_VARARGS, DOC_SURFACESETCOLORKEY },
    { "get_colorkey", (PyCFunction) surf_get_colorkey, METH_NOARGS,
      DOC_SURFACEGETCOLORKEY },
    { "set_alpha", surf_set_alpha, METH_VARARGS, DOC_SURFACESETALPHA },
    { "get_alpha", (PyCFunction) surf_get_alpha, METH_NOARGS,
      DOC_SURFACEGETALPHA },

    { "copy", (PyCFunction) surf_copy, METH_NOARGS, DOC_SURFACECOPY },
    { "__copy__", (PyCFunction) surf_copy, METH_NOARGS, DOC_SURFACECOPY },
    { "convert", surf_convert, METH_VARARGS, DOC_SURFACECONVERT },
    { "convert_alpha", surf_convert_alpha, METH_VARARGS,
      DOC_SURFACECONVERTALPHA },

    { "set_clip", surf_set_clip, METH_VARARGS, DOC_SURFACESETCLIP },
    { "get_clip", (PyCFunction) surf_get_clip, METH_NOARGS,
      DOC_SURFACEGETCLIP },

    { "fill", (PyCFunction) surf_fill, METH_VARARGS | METH_KEYWORDS,
      DOC_SURFACEFILL },
    { "blit", (PyCFunction) surf_blit, METH_VARARGS | METH_KEYWORDS,
      DOC_SURFACEBLIT },

    { "scroll", (PyCFunction) surf_scroll, METH_VARARGS | METH_KEYWORDS,
      DOC_SURFACESCROLL },

    { "get_flags", (PyCFunction) surf_get_flags, METH_NOARGS,
      DOC_SURFACEGETFLAGS },
    { "get_size", (PyCFunction) surf_get_size, METH_NOARGS,
      DOC_SURFACEGETSIZE },
    { "get_width", (PyCFunction) surf_get_width, METH_NOARGS,
      DOC_SURFACEGETWIDTH },
    { "get_height", (PyCFunction) surf_get_height, METH_NOARGS,
      DOC_SURFACEGETHEIGHT },
    { "get_rect", (PyCFunction) surf_get_rect, METH_VARARGS | METH_KEYWORDS,
      DOC_SURFACEGETRECT },
    { "get_pitch", (PyCFunction) surf_get_pitch, METH_NOARGS,
      DOC_SURFACEGETPITCH },
    { "get_bitsize", (PyCFunction) surf_get_bitsize, METH_NOARGS,
      DOC_SURFACEGETBITSIZE },
    { "get_bytesize", (PyCFunction) surf_get_bytesize, METH_NOARGS,
      DOC_SURFACEGETBYTESIZE },
    { "get_masks", (PyCFunction) surf_get_masks, METH_NOARGS,
      DOC_SURFACEGETMASKS },
    { "get_shifts", (PyCFunction) surf_get_shifts, METH_NOARGS,
      DOC_SURFACEGETSHIFTS },
    { "set_masks", (PyCFunction) surf_set_masks, METH_VARARGS,
      DOC_SURFACESETMASKS },
    { "set_shifts", (PyCFunction) surf_set_shifts, METH_VARARGS,
      DOC_SURFACESETSHIFTS },

    { "get_losses", (PyCFunction) surf_get_losses, METH_NOARGS,
      DOC_SURFACEGETLOSSES },

    { "subsurface", surf_subsurface, METH_VARARGS, DOC_SURFACESUBSURFACE },
    { "get_offset", (PyCFunction) surf_get_offset, METH_NOARGS,
      DOC_SURFACEGETOFFSET },
    { "get_abs_offset", (PyCFunction) surf_get_abs_offset, METH_NOARGS,
      DOC_SURFACEGETABSOFFSET },
    { "get_parent", (PyCFunction) surf_get_parent, METH_NOARGS,
      DOC_SURFACEGETPARENT },
    { "get_abs_parent", (PyCFunction) surf_get_abs_parent, METH_NOARGS,
      DOC_SURFACEGETABSPARENT },
    { "get_bounding_rect", (PyCFunction) surf_get_bounding_rect,
      METH_VARARGS | METH_KEYWORDS,
      DOC_SURFACEGETBOUNDINGRECT},
    { "get_buffer", (PyCFunction) surf_get_buffer, METH_NOARGS,
      DOC_SURFACEGETBUFFER},
    { "get_view", (PyCFunction) surf_get_view, METH_VARARGS | METH_KEYWORDS,
      DOC_SURFACEGETVIEW},

    { NULL, NULL, 0, NULL }
};

static PyTypeObject PySurface_Type = {
    TYPE_HEAD (NULL, 0)
    "pygame.Surface",          /* name */
    sizeof (PySurfaceObject),  /* basic size */
    0,                         /* itemsize */
    surface_dealloc,           /* dealloc */
    0,                         /* print */
    NULL,                      /* getattr */
    NULL,                      /* setattr */
    NULL,                      /* compare */
    surface_str,               /* repr */
    NULL,                      /* as_number */
    NULL,                      /* as_sequence */
    NULL,                      /* as_mapping */
    (hashfunc) NULL,           /* hash */
    (ternaryfunc) NULL,        /* call */
    (reprfunc) NULL,           /* str */
    0,
    0L, 0L,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  /* tp_flags */
    DOC_PYGAMESURFACE,         /* Documentation string */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    offsetof (PySurfaceObject, weakreflist),   /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    surface_methods,           /* tp_methods */
    0,                         /* tp_members */
    surface_getsets,           /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc) surface_init,   /* tp_init */
    0,                         /* tp_alloc */
    surface_new,               /* tp_new */
};

#define PySurface_Check(x) ((x)->ob_type == &PySurface_Type)

static PyObject*
PySurface_New (SDL_Surface *s)
{
    PySurfaceObject *self;

    if (!s)
        return RAISE (PyExc_SDLError, SDL_GetError ());

    self = (PySurfaceObject *)
        PySurface_Type.tp_new (&PySurface_Type, NULL, NULL);

    if (self)
        self->surf = s;

    return (PyObject *) self;
}

static PyObject*
surface_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PySurfaceObject *self;

    self = (PySurfaceObject *) type->tp_alloc (type, 0);
    if (self) {
        self->surf = NULL;
        self->subsurface = NULL;
        self->weakreflist = NULL;
        self->dependency = NULL;
        self->locklist = NULL;
    }
    return (PyObject *) self;
}

/* surface object internals */
static void
surface_cleanup (PySurfaceObject *self)
{
    if (self->surf) {
        if (!(self->surf->flags & SDL_HWSURFACE) ||
            SDL_WasInit (SDL_INIT_VIDEO)) {
            /* unsafe to free hardware surfaces without video init */
            /* i question SDL's ability to free a locked hardware surface */
            SDL_FreeSurface (self->surf);
        }
        self->surf = NULL;
    }
    if (self->subsurface) {
        Py_XDECREF (self->subsurface->owner);
        PyMem_Del (self->subsurface);
        self->subsurface = NULL;
    }
    if (self->dependency) {
        Py_DECREF (self->dependency);
        self->dependency = NULL;
    }

    if (self->locklist) {
        Py_DECREF (self->locklist);
        self->locklist = NULL;
    }
}

static void
surface_dealloc (PyObject *self)
{
    if (((PySurfaceObject *) self)->weakreflist)
        PyObject_ClearWeakRefs (self);
    surface_cleanup ((PySurfaceObject *) self);
    self->ob_type->tp_free (self);
}

static PyObject*
surface_str (PyObject *self)
{
    char str[1024];
    SDL_Surface *surf = PySurface_AsSurface (self);
    const char *type;

    if (surf) {
        type = (surf->flags & SDL_HWSURFACE) ? "HW" : "SW";
        sprintf (str, "<Surface(%dx%dx%d %s)>", surf->w,
                 surf->h, surf->format->BitsPerPixel, type);
    }
    else {
        strcpy (str, "<Surface(Dead Display)>");
    }

    return Text_FromUTF8 (str);
}

static intptr_t
surface_init (PySurfaceObject *self, PyObject *args, PyObject *kwds)
{
    Uint32 flags = 0;
    int width, height;
    PyObject *depth = NULL, *masks = NULL, *size = NULL;
    int bpp;
    Uint32 Rmask, Gmask, Bmask, Amask;
    SDL_Surface *surface;
    SDL_PixelFormat default_format, *format;
    int length;

    char *kwids[] = { "size", "flags", "depth", "masks", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iOO", kwids, &size, &flags, &depth, &masks))
       return -1;

    if (PySequence_Check (size) && (length = PySequence_Length (size)) == 2) {
        if ( (!IntFromObjIndex (size, 0, &width)) ||
             (!IntFromObjIndex (size, 1, &height)) ) {
            RAISE (PyExc_ValueError, "size needs to be (int width, int height)");
            return -1;
        }
    } else {
        RAISE (PyExc_ValueError, "size needs to be (int width, int height)");
        return -1;
    }

    if (width < 0 || height < 0) {
        RAISE (PyExc_SDLError, "Invalid resolution for Surface");
        return -1;
    }

    surface_cleanup (self);

    if (depth && masks) {      /* all info supplied, most errorchecking
                                * needed */
        if (PySurface_Check (depth)) {
            RAISE (PyExc_ValueError, "cannot pass surface for depth and color masks");
            return -1;
        }
        if (!IntFromObj (depth, &bpp)) {
            RAISE (PyExc_ValueError, "invalid bits per pixel depth argument");
            return -1;
        }
        if (!PySequence_Check (masks) || PySequence_Length (masks) != 4) {
            RAISE (PyExc_ValueError, "masks argument must be sequence of four numbers");
            return -1;
        }
        if (!UintFromObjIndex (masks, 0, &Rmask) ||
            !UintFromObjIndex (masks, 1, &Gmask) ||
            !UintFromObjIndex (masks, 2, &Bmask) ||
            !UintFromObjIndex (masks, 3, &Amask)) {
            RAISE (PyExc_ValueError, "invalid mask values in masks sequence");
            return -1;
        }
    }
    else if (depth && PyNumber_Check (depth)) { /* use default masks */
        if (!IntFromObj (depth, &bpp)) {
            RAISE (PyExc_ValueError, "invalid bits per pixel depth argument");
            return -1;
        }
        if (flags & SDL_SRCALPHA){
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
                RAISE (PyExc_ValueError, "no standard masks exist for given bitdepth with alpha");
                return -1;
            }
        }
        else {
            Amask = 0;
            switch (bpp) {

            case 8:
                Rmask = 0xFF >> 6 << 5;
                Gmask = 0xFF >> 5 << 2;
                Bmask = 0xFF >> 6;
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
                RAISE (PyExc_ValueError, "nonstandard bit depth given");
                return -1;
            }
        }
    }
    else { /* no depth or surface */
        SDL_PixelFormat* pix;
        if (depth && PySurface_Check (depth))
            pix = ((PySurfaceObject*) depth)->surf->format;
        else if (SDL_GetVideoSurface ())
            pix = SDL_GetVideoSurface ()->format;
        else if (SDL_WasInit (SDL_INIT_VIDEO))
            pix = SDL_GetVideoInfo ()->vfmt;
        else {
            pix = &default_format;
            pix->BitsPerPixel = 32;
            pix->Amask = 0;
            pix->Rmask = 0xFF0000;
            pix->Gmask = 0xFF00;
            pix->Bmask = 0xFF;
        }
        bpp = pix->BitsPerPixel;

        if (flags & SDL_SRCALPHA) {
            switch (bpp) {

            case 16:
                Rmask = 0xF << 8;
                Gmask = 0xF << 4;
                Bmask = 0xF;
                Amask = 0xF << 12;
                break;
            case 24:
                bpp = 32;
                // we automatically step up to 32 if video is 24, fall through to case below
            case 32:
                Rmask = 0xFF << 16;
                Gmask = 0xFF << 8;
                Bmask = 0xFF;
                Amask = 0xFF << 24;
                break;
            default:
                RAISE (PyExc_ValueError, "no standard masks exist for given bitdepth with alpha");
                return -1;
            }
        } else {
            Rmask = pix->Rmask;
            Gmask = pix->Gmask;
            Bmask = pix->Bmask;
            Amask = pix->Amask;
        }

    }

    surface = SDL_CreateRGBSurface (flags, width, height, bpp, Rmask, Gmask,
                                    Bmask, Amask);

    if (!surface) {
        RAISE (PyExc_SDLError, SDL_GetError ());
        return -1;
    }

    if (masks) {
    /* Confirm the surface was created correctly (masks were valid).
       Also ensure that 24 and 32 bit surfaces have 8 bit fields
       (no losses).
    */
        format = surface->format;
        Rmask = (0xFF >> format->Rloss) << format->Rshift;
        Gmask = (0xFF >> format->Gloss) << format->Gshift;
        Bmask = (0xFF >> format->Bloss) << format->Bshift;
        Amask = (0xFF >> format->Aloss) << format->Ashift;
        if (format->Rmask != Rmask ||
            format->Gmask != Gmask ||
            format->Bmask != Bmask ||
            format->Amask != Amask ||
            (format->BytesPerPixel >= 3 &&
             (format->Rloss || format->Gloss || format->Bloss ||
              (surface->flags & SDL_SRCALPHA ?
               format->Aloss : format->Aloss != 8)))) {
        SDL_FreeSurface (surface);
        RAISE (PyExc_ValueError, "Invalid mask values");
        return -1;
        }
    }

    if (surface) {
        self->surf = surface;
        self->subsurface = NULL;
    }

    return 0;
}

/* surface object methods */
static PyObject*
surf_get_at (PyObject *self, PyObject *args)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    SDL_PixelFormat *format = surf->format;
    Uint8 *pixels = (Uint8 *) surf->pixels;
    int x, y;
    Uint32 color;
    Uint8 *pix;
    Uint8 rgba[4];

    if (!PyArg_ParseTuple (args, "(ii)", &x, &y))
        return NULL;
    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");

    if (surf->flags & SDL_OPENGL)
        return RAISE (PyExc_SDLError, "Cannot call on OPENGL Surfaces");

    if (x < 0 || x >= surf->w || y < 0 || y >= surf->h)
        return RAISE (PyExc_IndexError, "pixel index out of range");

    if (format->BytesPerPixel < 1 || format->BytesPerPixel > 4)
        return RAISE (PyExc_RuntimeError, "invalid color depth for surface");

    if (!PySurface_Lock (self))
        return NULL;

    pixels = (Uint8 *) surf->pixels;

    switch (format->BytesPerPixel) {

    case 1:
        color = (Uint32)*((Uint8 *) pixels + y * surf->pitch + x);
        break;
    case 2:
        color = (Uint32)*((Uint16 *) (pixels + y * surf->pitch) + x);
        break;
    case 3:
        pix = ((Uint8 *) (pixels + y * surf->pitch) + x * 3);
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
        color = (pix[0]) + (pix[1] << 8) + (pix[2] << 16);
#else
        color = (pix[2]) + (pix[1] << 8) + (pix[0] << 16);
#endif
        break;
    default:                  /* case 4: */
        color = *((Uint32 *) (pixels + y * surf->pitch) + x);
        break;
    }
    if (!PySurface_Unlock (self))
        return NULL;

    SDL_GetRGBA (color, format, rgba, rgba+1, rgba+2, rgba+3);
    return PyColor_New (rgba);
}

static PyObject*
surf_set_at (PyObject *self, PyObject *args)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    SDL_PixelFormat *format = surf->format;
    Uint8 *pixels;
    int x, y;
    Uint32 color;
    Uint8 rgba[4] = {0, 0, 0, 0 };
    PyObject *rgba_obj;
    Uint8 *byte_buf;

    if (!PyArg_ParseTuple (args, "(ii)O", &x, &y, &rgba_obj))
        return NULL;
    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");

    if (surf->flags & SDL_OPENGL)
        return RAISE (PyExc_SDLError, "Cannot call on OPENGL Surfaces");

    if (format->BytesPerPixel < 1 || format->BytesPerPixel > 4)
        return RAISE (PyExc_RuntimeError, "invalid color depth for surface");

    if (x < surf->clip_rect.x || x >= surf->clip_rect.x + surf->clip_rect.w ||
        y < surf->clip_rect.y || y >= surf->clip_rect.y + surf->clip_rect.h)  {
        /* out of clip area */
        Py_RETURN_NONE;
    }

    if (PyInt_Check (rgba_obj)) {
        color = (Uint32) PyInt_AsLong (rgba_obj);
        if (PyErr_Occurred () && (Sint32) color == -1)
            return RAISE (PyExc_TypeError, "invalid color argument");
    }
    else if (PyLong_Check (rgba_obj)) {
        color = (Uint32) PyLong_AsUnsignedLong (rgba_obj);
        if (PyErr_Occurred () && (Sint32) color == -1)
            return RAISE (PyExc_TypeError, "invalid color argument");
    }
    else if (RGBAFromColorObj (rgba_obj, rgba))
        color = SDL_MapRGBA (surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
    else
        return RAISE (PyExc_TypeError, "invalid color argument");

    if (!PySurface_Lock (self))
        return NULL;
    pixels = (Uint8 *) surf->pixels;

    switch (format->BytesPerPixel) {

    case 1:
        *((Uint8 *) pixels + y * surf->pitch + x) = (Uint8) color;
        break;
    case 2:
        *((Uint16 *) (pixels + y * surf->pitch) + x) = (Uint16) color;
        break;
    case 3:
        byte_buf = (Uint8 *) (pixels + y * surf->pitch) + x * 3;
#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
        *(byte_buf + (format->Rshift >> 3)) = (Uint8) (color >> 16);
        *(byte_buf + (format->Gshift >> 3)) = (Uint8) (color >> 8);
        *(byte_buf + (format->Bshift >> 3)) = (Uint8) color;
#else
        *(byte_buf + 2 - (format->Rshift >> 3)) = (Uint8) (color >> 16);
        *(byte_buf + 2 - (format->Gshift >> 3)) = (Uint8) (color >> 8);
        *(byte_buf + 2 - (format->Bshift >> 3)) = (Uint8) color;
#endif
        break;
    default:                  /* case 4: */
        *((Uint32 *) (pixels + y * surf->pitch) + x) = color;
        break;
    }

    if (!PySurface_Unlock (self))
        return NULL;
    Py_RETURN_NONE;
}

static PyObject*
surf_get_at_mapped (PyObject *self, PyObject *args)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    SDL_PixelFormat *format = surf->format;
    Uint8 *pixels = (Uint8 *) surf->pixels;
    int x, y;
    Uint32 color;
    Uint8 *pix;

    if (!PyArg_ParseTuple (args, "(ii)", &x, &y))
        return NULL;
    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");

    if (surf->flags & SDL_OPENGL)
        return RAISE (PyExc_SDLError, "Cannot call on OPENGL Surfaces");

    if (x < 0 || x >= surf->w || y < 0 || y >= surf->h)
        return RAISE (PyExc_IndexError, "pixel index out of range");

    if (format->BytesPerPixel < 1 || format->BytesPerPixel > 4)
        return RAISE (PyExc_RuntimeError, "invalid color depth for surface");

    if (!PySurface_Lock (self))
        return NULL;

    pixels = (Uint8 *) surf->pixels;

    switch (format->BytesPerPixel) {

    case 1:
        color = (Uint32)*((Uint8 *) pixels + y * surf->pitch + x);
        break;
    case 2:
        color = (Uint32)*((Uint16 *) (pixels + y * surf->pitch) + x);
        break;
    case 3:
        pix = ((Uint8 *) (pixels + y * surf->pitch) + x * 3);
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
        color = (pix[0]) + (pix[1] << 8) + (pix[2] << 16);
#else
        color = (pix[2]) + (pix[1] << 8) + (pix[0] << 16);
#endif
        break;
    default:                  /* case 4: */
        color = *((Uint32 *) (pixels + y * surf->pitch) + x);
        break;
    }
    if (!PySurface_Unlock (self))
        return NULL;

    return PyInt_FromLong ((long)color);
}

static PyObject*
surf_map_rgb (PyObject *self, PyObject *args)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    Uint8 rgba[4];
    int color;

    if (!RGBAFromColorObj (args, rgba))
        return RAISE (PyExc_TypeError, "Invalid RGBA argument");
    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");

    color = SDL_MapRGBA (surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
    return PyInt_FromLong (color);
}

static PyObject*
surf_unmap_rgb (PyObject *self, PyObject *arg)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    Uint32 col;
    Uint8  rgba[4];

    col = (Uint32)PyInt_AsLong (arg);
    if (col == (Uint32) -1 && PyErr_Occurred()) {
    PyErr_Clear();
    return RAISE (PyExc_TypeError, "unmap_rgb expects 1 number argument");
    }
    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");

    SDL_GetRGBA (col, surf->format, rgba, rgba+1, rgba+2, rgba+3);

    return PyColor_New (rgba);
}

static PyObject*
surf_lock (PyObject *self)
{
    if (!PySurface_Lock (self))
        return NULL;
    Py_RETURN_NONE;
}

static PyObject*
surf_unlock (PyObject *self)
{
    PySurface_Unlock (self);
    Py_RETURN_NONE;
}

static PyObject*
surf_mustlock (PyObject *self)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    return PyInt_FromLong (SDL_MUSTLOCK (surf) ||
        ((PySurfaceObject *) self)->subsurface);
}

static PyObject*
surf_get_locked (PyObject *self)
{
    PySurfaceObject *surf = (PySurfaceObject *) self;

    if (surf->locklist && PyList_Size (surf->locklist) > 0)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject*
surf_get_locks (PyObject *self)
{
    PySurfaceObject *surf = (PySurfaceObject *) self;
    Py_ssize_t len, i = 0;
    PyObject *tuple, *tmp;
    if (!surf->locklist)
        return PyTuple_New (0);

    len = PyList_Size (surf->locklist);
    tuple = PyTuple_New (len);
    if (!tuple)
        return NULL;

    for (i = 0; i < len; i++) {
        tmp = PyWeakref_GetObject (PyList_GetItem (surf->locklist, i));
        Py_INCREF (tmp);
        PyTuple_SetItem (tuple, i, tmp);
    }
    return tuple;
}

static PyObject*
surf_get_palette (PyObject *self)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    SDL_Palette *pal = surf->format->palette;
    PyObject *list;
    int i;
    PyObject *color;
    SDL_Color *c;
    Uint8 rgba[4] = {0, 0, 0, 255};

    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");

    if (!pal)
        return RAISE (PyExc_SDLError, "Surface has no palette to get\n");

    list = PyTuple_New (pal->ncolors);
    if (!list)
        return NULL;

    for (i = 0; i < pal->ncolors; i++) {
        c = &pal->colors[i];
        rgba[0] = c->r;
        rgba[1] = c->g;
        rgba[2] = c->b;
        color = PyColor_NewLength (rgba, 3);

        if (!color) {
            Py_DECREF (list);
            return NULL;
        }
        PyTuple_SET_ITEM (list, i, color);
    }

    return list;
}

static PyObject*
surf_get_palette_at (PyObject * self, PyObject * args)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    SDL_Palette *pal = surf->format->palette;
    SDL_Color *c;
    int _index;
    Uint8 rgba[4];

    if (!PyArg_ParseTuple (args, "i", &_index))
        return NULL;
    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");

    if (!pal)
        return RAISE (PyExc_SDLError, "Surface has no palette to set\n");
    if (_index >= pal->ncolors || _index < 0)
        return RAISE (PyExc_IndexError, "index out of bounds");

    c = &pal->colors[_index];
    rgba[0] = c->r;
    rgba[1] = c->g;
    rgba[2] = c->b;
    rgba[3] = 255;

    return PyColor_NewLength (rgba, 3);
}

static PyObject*
surf_set_palette (PyObject *self, PyObject *args)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    SDL_Palette *pal = surf->format->palette;
    SDL_Color *colors;
    PyObject *list, *item;
    int i, len;
    Uint8 rgba[4];
    int ecode;

    if (!PyArg_ParseTuple (args, "O", &list))
        return NULL;
    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");
    if (!PySequence_Check (list))
        return RAISE (PyExc_ValueError, "Argument must be a sequence type");

    if (!pal)
        return RAISE (PyExc_SDLError, "Surface has no palette\n");

    if (!SDL_WasInit (SDL_INIT_VIDEO))
        return RAISE (PyExc_SDLError,
                      "cannot set palette without pygame.display initialized");

    len = MIN (pal->ncolors, PySequence_Length (list));

    colors = (SDL_Color *) malloc (len * sizeof (SDL_Color));
    if (!colors)
        return NULL;

    for (i = 0; i < len; i++) {
        item = PySequence_GetItem (list, i);

        ecode = RGBAFromObj (item, rgba);
        Py_DECREF (item);
        if (!ecode) {
            free (colors);
            return RAISE (PyExc_ValueError,
                          "takes a sequence of integers of RGB");
        }
        if (rgba[3] != 255) {
            free (colors);
            return RAISE (PyExc_ValueError,
                          "takes an alpha value of 255");
        }
        colors[i].r = (unsigned char) rgba[0];
        colors[i].g = (unsigned char) rgba[1];
        colors[i].b = (unsigned char) rgba[2];
    }

    SDL_SetColors (surf, colors, 0, len);
    free (colors);
    Py_RETURN_NONE;
}

static PyObject*
surf_set_palette_at (PyObject *self, PyObject *args)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    SDL_Palette *pal = surf->format->palette;
    SDL_Color color;
    int _index;
    PyObject *color_obj;
    Uint8 rgba[4];

    if (!PyArg_ParseTuple (args, "iO", &_index, &color_obj))
        return NULL;
    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");

    if (!RGBAFromObj (color_obj, rgba)) {
        return RAISE (PyExc_ValueError,
                      "takes a sequence of integers of RGB for argument 2");
    }

    if (!pal) {
        PyErr_SetString (PyExc_SDLError, "Surface is not palettized\n");
        return NULL;
    }

    if (_index >= pal->ncolors || _index < 0) {
        PyErr_SetString (PyExc_IndexError, "index out of bounds");
        return NULL;
    }

    if (!SDL_WasInit (SDL_INIT_VIDEO))
        return RAISE (PyExc_SDLError,
                      "cannot set palette without pygame.display initialized");

    color.r = rgba[0];
    color.g = rgba[1];
    color.b = rgba[2];

    SDL_SetColors (surf, &color, _index, 1);

    Py_RETURN_NONE;
}

static PyObject*
surf_set_colorkey (PyObject *self, PyObject *args)
{
    SDL_Surface    *surf = PySurface_AsSurface (self);
    Uint32 flags = 0, color = 0;
    PyObject *rgba_obj = NULL;
    Uint8 rgba[4];
    int result, hascolor = 0;

    if (!PyArg_ParseTuple (args, "|Oi", &rgba_obj, &flags))
        return NULL;
    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");

    if (surf->flags & SDL_OPENGL)
        return RAISE (PyExc_SDLError, "Cannot call on OPENGL Surfaces");

    if (rgba_obj && rgba_obj != Py_None) {
        if (PyInt_Check (rgba_obj)) {
            color = (Uint32) PyInt_AsLong (rgba_obj);
            if (PyErr_Occurred () && (Sint32) color == -1)
                return RAISE (PyExc_TypeError, "invalid color argument");
        }
        else if (PyLong_Check (rgba_obj)) {
            color = (Uint32) PyLong_AsUnsignedLong (rgba_obj);
            if (PyErr_Occurred () && (Sint32) color == -1)
                return RAISE (PyExc_TypeError, "invalid color argument");
        }
        else if (RGBAFromColorObj (rgba_obj, rgba)) {
            color = SDL_MapRGBA (surf->format, rgba[0], rgba[1], rgba[2],
                rgba[3]);
        }
        else
            return RAISE (PyExc_TypeError, "invalid color argument");
        hascolor = 1;
    }
    if (hascolor)
        flags |= SDL_SRCCOLORKEY;

    PySurface_Prep (self);
    result = SDL_SetColorKey (surf, flags, color);
    PySurface_Unprep (self);

    if (result == -1)
        return RAISE (PyExc_SDLError, SDL_GetError ());

    Py_RETURN_NONE;
}

static PyObject*
surf_get_colorkey (PyObject *self)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    Uint8 r, g, b, a;

    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");

    if (surf->flags & SDL_OPENGL)
        return RAISE (PyExc_SDLError, "Cannot call on OPENGL Surfaces");

    if (!(surf->flags & SDL_SRCCOLORKEY))
        Py_RETURN_NONE;

    SDL_GetRGBA (surf->format->colorkey, surf->format, &r, &g, &b, &a);
    return Py_BuildValue ("(bbbb)", r, g, b, a);
}

static PyObject*
surf_set_alpha (PyObject *self, PyObject *args)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    Uint32 flags = 0;
    PyObject *alpha_obj = NULL, *intobj = NULL;
    Uint8 alpha;
    int result, alphaval = 255, hasalpha = 0;

    if (!PyArg_ParseTuple (args, "|Oi", &alpha_obj, &flags))
        return NULL;
    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");

    if (surf->flags & SDL_OPENGL)
        return RAISE (PyExc_SDLError, "Cannot call on OPENGL Surfaces");

    if (alpha_obj && alpha_obj != Py_None) {
        if (PyNumber_Check (alpha_obj) &&
            (intobj = PyNumber_Int (alpha_obj))) {
            if (PyInt_Check (intobj)) {
                alphaval = (int) PyInt_AsLong (intobj);
                Py_DECREF (intobj);
            }
            else
                return RAISE (PyExc_TypeError, "invalid alpha argument");
        }
        else
            return RAISE (PyExc_TypeError, "invalid alpha argument");
        hasalpha = 1;
    }
    if (hasalpha)
        flags |= SDL_SRCALPHA;

    if (alphaval > 255)
        alpha = 255;
    else if (alphaval < 0)
        alpha = 0;
    else
        alpha = (Uint8) alphaval;

    PySurface_Prep (self);
    result = SDL_SetAlpha (surf, flags, alpha);
    PySurface_Unprep (self);

    if (result == -1)
        return RAISE (PyExc_SDLError, SDL_GetError ());

    Py_RETURN_NONE;
}

static PyObject*
surf_get_alpha (PyObject *self)
{
    SDL_Surface *surf = PySurface_AsSurface (self);

    if (surf->flags & SDL_OPENGL)
        return RAISE (PyExc_SDLError, "Cannot call on OPENGL Surfaces");
    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");

    if (surf->flags & SDL_SRCALPHA)
        return PyInt_FromLong (surf->format->alpha);

    Py_RETURN_NONE;
}

static PyObject*
surf_copy (PyObject *self)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    PyObject *final;
    SDL_Surface *newsurf;

    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");

    if (surf->flags & SDL_OPENGL)
        return RAISE (PyExc_SDLError, "Cannot copy opengl display");

    PySurface_Prep (self);
    newsurf = SDL_ConvertSurface (surf, surf->format, surf->flags);
    PySurface_Unprep (self);

    final = PySurface_New (newsurf);
    if (!final)
        SDL_FreeSurface (newsurf);
    return final;
}

static PyObject*
surf_convert (PyObject *self, PyObject *args)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    PyObject *final;
    PyObject *argobject = NULL;
    SDL_Surface *src;
    SDL_Surface *newsurf;
    Uint32 flags = -1;

    if (!SDL_WasInit (SDL_INIT_VIDEO))
        return RAISE (PyExc_SDLError,
                      "cannot convert without pygame.display initialized");

    if (!PyArg_ParseTuple (args, "|Oi", &argobject, &flags))
        return NULL;

    if (surf->flags & SDL_OPENGL)
        return RAISE (PyExc_SDLError, "Cannot convert opengl display");

    PySurface_Prep (self);
    if (argobject) {
        if (PySurface_Check (argobject)) {
            src = PySurface_AsSurface (argobject);
            flags = src->flags |
                (surf->flags & (SDL_SRCCOLORKEY | SDL_SRCALPHA));
            newsurf = SDL_ConvertSurface (surf, src->format, flags);
        }
        else {
            int bpp;
            SDL_PixelFormat format;

            memcpy (&format, surf->format, sizeof (format));
            if (IntFromObj (argobject, &bpp)) {
                Uint32 Rmask, Gmask, Bmask, Amask;

                if (flags != -1 && flags & SDL_SRCALPHA) {
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
                        return RAISE (PyExc_ValueError,
                                      "no standard masks exist for given "
                                      "bitdepth with alpha");
                    }
                }
                else {
                    Amask = 0;
                    switch (bpp) {

                    case 8:
                        Rmask = 0xFF >> 6 << 5;
                        Gmask = 0xFF >> 5 << 2;
                        Bmask = 0xFF >> 6;
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
                        return RAISE (PyExc_ValueError,
                                      "nonstandard bit depth given");
                    }
                }
                format.Rmask = Rmask;
                format.Gmask = Gmask;
                format.Bmask = Bmask;
                format.Amask = Amask;
            }
            else if (PySequence_Check (argobject) &&
                     PySequence_Size (argobject) == 4) {
                Uint32 mask;

                if (!UintFromObjIndex (argobject, 0, &format.Rmask) ||
                    !UintFromObjIndex (argobject, 1, &format.Gmask) ||
                    !UintFromObjIndex (argobject, 2, &format.Bmask) ||
                    !UintFromObjIndex (argobject, 3, &format.Amask))   {
                    PySurface_Unprep (self);
                    return RAISE (PyExc_ValueError,
                                  "invalid color masks given");
                }
                mask = format.Rmask | format.Gmask | format.Bmask |
                    format.Amask;
                for (bpp = 0; bpp < 32; ++bpp)
                    if (!(mask >> bpp))
                        break;
            }
            else {
                PySurface_Unprep (self);
                return RAISE
                    (PyExc_ValueError,
                     "invalid argument specifying new format to convert to");
            }
            format.BitsPerPixel = (Uint8) bpp;
            format.BytesPerPixel = (bpp + 7) / 8;
            if (flags == -1)
                flags = surf->flags;
            if (format.Amask)
                flags |= SDL_SRCALPHA;
            newsurf = SDL_ConvertSurface (surf, &format, flags);
        }
    }
    else {
        if (SDL_WasInit (SDL_INIT_VIDEO))
            newsurf = SDL_DisplayFormat (surf);
        else
            newsurf = SDL_ConvertSurface (surf, surf->format, surf->flags);
    }
    PySurface_Unprep (self);

    final = PySurface_New (newsurf);
    if (!final)
        SDL_FreeSurface (newsurf);
    return final;
}

static PyObject*
surf_convert_alpha (PyObject *self, PyObject *args)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    PyObject *final;
    PySurfaceObject *srcsurf = NULL;
    SDL_Surface *newsurf, *src;

    if (!SDL_WasInit (SDL_INIT_VIDEO))
        return RAISE (PyExc_SDLError,
                      "cannot convert without pygame.display initialized");

    if (!PyArg_ParseTuple (args, "|O!", &PySurface_Type, &srcsurf))
        return NULL;

    PySurface_Prep (self);
    if (srcsurf) {
        /*
         * hmm, we have to figure this out, not all depths have good
         * support for alpha
         */
        src = PySurface_AsSurface (srcsurf);
        newsurf = SDL_DisplayFormatAlpha (surf);
    }
    else
        newsurf = SDL_DisplayFormatAlpha (surf);
    PySurface_Unprep (self);

    final = PySurface_New (newsurf);
    if (!final)
        SDL_FreeSurface (newsurf);
    return final;
}

static PyObject*
surf_set_clip (PyObject *self, PyObject *args)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    PyObject *item;
    GAME_Rect *rect = NULL, temp;
    SDL_Rect sdlrect;
    int result;

    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");
    if (PyTuple_Size (args)) {
        item = PyTuple_GET_ITEM (args, 0);
        if (item == Py_None && PyTuple_Size (args) == 1) {
            result = SDL_SetClipRect (surf, NULL);
        }
        else {
            rect = GameRect_FromObject (args, &temp);
            if (!rect)
                return RAISE (PyExc_ValueError, "invalid rectstyle object");
            sdlrect.x = rect->x;
            sdlrect.y = rect->y;
            sdlrect.h = rect->h;
            sdlrect.w = rect->w;
            result = SDL_SetClipRect (surf, &sdlrect);
        }
    }
    else {
        result = SDL_SetClipRect (surf, NULL);
    }

    if (result == -1) {
        return RAISE (PyExc_SDLError, SDL_GetError ());
    }

    Py_RETURN_NONE;
}

static PyObject*
surf_get_clip (PyObject *self)
{
    SDL_Surface *surf = PySurface_AsSurface (self);

    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");
    return PyRect_New (&surf->clip_rect);
}


static PyObject*
surf_fill (PyObject *self, PyObject *args, PyObject *keywds)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    GAME_Rect *rect, temp;
    PyObject *r = NULL;
    Uint32 color;
    int result;
    PyObject *rgba_obj;
    Uint8 rgba[4];
    SDL_Rect sdlrect;
    int blendargs = 0;

    static char *kwids[] = {"color", "rect", "special_flags", NULL};
    if (!PyArg_ParseTupleAndKeywords (args, keywds, "O|Oi", kwids,
                                      &rgba_obj, &r, &blendargs))
        return NULL;
    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");

    if (surf->flags & SDL_OPENGL)
        return RAISE (PyExc_SDLError, "Cannot call on OPENGL Surfaces");

    if (PyInt_Check (rgba_obj))
        color = (Uint32) PyInt_AsLong (rgba_obj);
    else if (PyLong_Check (rgba_obj))
        color = (Uint32) PyLong_AsUnsignedLong (rgba_obj);
    else if (RGBAFromColorObj (rgba_obj, rgba))
        color = SDL_MapRGBA (surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
    else
        return RAISE (PyExc_TypeError, "invalid color argument");

    if (!r || r == Py_None) {
        rect = &temp;
        temp.x = temp.y = 0;
        temp.w = surf->w;
        temp.h = surf->h;
    }
    else if (!(rect = GameRect_FromObject (r, &temp)))
        return RAISE (PyExc_ValueError, "invalid rectstyle object");

    /* we need a fresh copy so our Rect values don't get munged */
    if (rect != &temp) {
        memcpy (&temp, rect, sizeof (temp));
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
        if(sdlrect.x + sdlrect.w <= 0 || sdlrect.y + sdlrect.h <= 0) {
            sdlrect.w = 0;
            sdlrect.h = 0;
        }

        if (sdlrect.x < 0) {
            sdlrect.x = 0;
        }
        if (sdlrect.y < 0) {
            sdlrect.y = 0;
        }

        if(sdlrect.x + sdlrect.w > surf->w) {
            sdlrect.w = sdlrect.w + (surf->w - (sdlrect.x + sdlrect.w));
        }
        if(sdlrect.y + sdlrect.h > surf->h) {
            sdlrect.h = sdlrect.h + (surf->h - (sdlrect.y + sdlrect.h));
        }

        /* printf("%d, %d, %d, %d\n", sdlrect.x, sdlrect.y, sdlrect.w, sdlrect.h); */


        if (blendargs != 0) {

            /*
            printf ("Using blendargs: %d\n", blendargs);
            */
            result = surface_fill_blend (surf, &sdlrect, color, blendargs);
        }
        else {
            PySurface_Prep (self);
            result = SDL_FillRect (surf, &sdlrect, color);
            PySurface_Unprep (self);
        }
        if (result == -1)
            return RAISE (PyExc_SDLError, SDL_GetError ());
    }
    return PyRect_New (&sdlrect);
}

static PyObject*
surf_blit (PyObject *self, PyObject *args, PyObject *keywds)
{
    SDL_Surface *src, *dest = PySurface_AsSurface (self);
    GAME_Rect *src_rect, temp;
    PyObject *srcobject, *argpos, *argrect = NULL;
    int dx, dy, result;
    SDL_Rect dest_rect, sdlsrc_rect;
    int sx, sy;
    int the_args = 0;

    static char *kwids[] = {"source", "dest", "area", "special_flags", NULL};
    if (!PyArg_ParseTupleAndKeywords (args, keywds, "O!O|Oi", kwids,
                                      &PySurface_Type, &srcobject, &argpos,
                                      &argrect, &the_args))
        return NULL;

    src = PySurface_AsSurface (srcobject);
    if (!dest || !src)
        return RAISE (PyExc_SDLError, "display Surface quit");

    if (dest->flags & SDL_OPENGL &&
        !(dest->flags & (SDL_OPENGLBLIT & ~SDL_OPENGL)))
        return RAISE (PyExc_SDLError,
                      "Cannot blit to OPENGL Surfaces (OPENGLBLIT is ok)");

    if ((src_rect = GameRect_FromObject (argpos, &temp))) {
        dx = src_rect->x;
        dy = src_rect->y;
    }
    else if (TwoIntsFromObj (argpos, &sx, &sy)) {
        dx = sx;
        dy = sy;
    }
    else
        return RAISE (PyExc_TypeError, "invalid destination position for blit");

    if (argrect && argrect != Py_None) {
        if (!(src_rect = GameRect_FromObject (argrect, &temp)))
            return RAISE (PyExc_TypeError, "Invalid rectstyle argument");
    }
    else {
        temp.x = temp.y = 0;
        temp.w = src->w;
        temp.h = src->h;
        src_rect = &temp;
    }

    dest_rect.x = (short) dx;
    dest_rect.y = (short) dy;
    dest_rect.w = (unsigned short) src_rect->w;
    dest_rect.h = (unsigned short) src_rect->h;
    sdlsrc_rect.x = (short) src_rect->x;
    sdlsrc_rect.y = (short) src_rect->y;
    sdlsrc_rect.w = (unsigned short) src_rect->w;
    sdlsrc_rect.h = (unsigned short) src_rect->h;

    if (!the_args)
        the_args = 0;

    result = PySurface_Blit (self, srcobject, &dest_rect, &sdlsrc_rect,
                             the_args);
    if (result != 0)
        return NULL;

    return PyRect_New (&dest_rect);
}

static PyObject*
surf_scroll (PyObject *self, PyObject *args, PyObject *keywds)
{
    int dx = 0, dy = 0;
    SDL_Surface *surf;
    int bpp;
    int pitch;
    SDL_Rect *clip_rect;
    int w, h;
    Uint8 *src, *dst;

    static char *kwids[] = {"dx", "dy", NULL};
    if (!PyArg_ParseTupleAndKeywords (args, keywds, "|ii", kwids, &dx, &dy)) {
        return NULL;
    }

    surf = PySurface_AsSurface (self);
    if (!surf) {
        return RAISE (PyExc_SDLError, "display Surface quit");
    }

    if (surf->flags & SDL_OPENGL &&
        !(surf->flags & (SDL_OPENGLBLIT & ~SDL_OPENGL))) {
        return RAISE (PyExc_SDLError,
                      "Cannot scroll an OPENGL Surfaces (OPENGLBLIT is ok)");
    }

    if (dx == 0 && dy == 0) {
        Py_RETURN_NONE;
    }

    clip_rect = &surf->clip_rect;
    w = clip_rect->w;
    h = clip_rect->h;
    if (dx >= w || dx <= -w || dy >= h || dy <= -h) {
        Py_RETURN_NONE;
    }

    if (!PySurface_Lock (self)) {
        return NULL;
    }

    bpp = surf->format->BytesPerPixel;
    pitch = surf->pitch;
    src = dst = (Uint8 *) surf->pixels +
            clip_rect->y * pitch + clip_rect->x * bpp;
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
    surface_move (src, dst, h, w * bpp, pitch, pitch);

    if (!PySurface_Unlock (self)) {
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject*
surf_get_flags (PyObject *self)
{
    SDL_Surface *surf = PySurface_AsSurface (self);

    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");
    return PyInt_FromLong ((long)surf->flags);
}

static PyObject*
surf_get_pitch (PyObject *self)
{
    SDL_Surface *surf = PySurface_AsSurface (self);

    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");
    return PyInt_FromLong (surf->pitch);
}

static PyObject*
surf_get_size (PyObject *self)
{
    SDL_Surface *surf = PySurface_AsSurface (self);

    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");
    return Py_BuildValue ("(ii)", surf->w, surf->h);
}

static PyObject*
surf_get_width (PyObject *self)
{
    SDL_Surface *surf = PySurface_AsSurface (self);

    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");
    return PyInt_FromLong (surf->w);
}

static PyObject*
surf_get_height (PyObject *self)
{
    SDL_Surface *surf = PySurface_AsSurface (self);

    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");
    return PyInt_FromLong (surf->h);
}

static PyObject*
surf_get_rect (PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *rect;
    SDL_Surface *surf = PySurface_AsSurface (self);

    if (PyTuple_GET_SIZE (args) > 0) {
        return RAISE (PyExc_TypeError,
                      "get_rect only accepts keyword arguments");
    }

    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");

    rect = PyRect_New4 (0, 0, surf->w, surf->h);
    if (rect && kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;

        while (PyDict_Next (kwargs, &pos, &key, &value)) {
            if ((PyObject_SetAttr (rect, key, value) == -1)) {
                Py_DECREF (rect);
                return NULL;
            }
        }
    }
    return rect;
}

static PyObject*
surf_get_bitsize (PyObject *self)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");

    return PyInt_FromLong (surf->format->BitsPerPixel);
}

static PyObject*
surf_get_bytesize (PyObject *self)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");
    return PyInt_FromLong (surf->format->BytesPerPixel);
}

static PyObject*
surf_get_masks (PyObject *self)
{
    SDL_Surface *surf = PySurface_AsSurface (self);

    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");
    return Py_BuildValue ("(IIII)", surf->format->Rmask, surf->format->Gmask,
                          surf->format->Bmask, surf->format->Amask);
}


static PyObject*
surf_set_masks (PyObject *self, PyObject *args)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    /* Need to use 64bit vars so this works on 64 bit pythons. */
    Uint64 r, g, b, a;

    if (!PyArg_ParseTuple (args, "(kkkk)", &r, &g, &b, &a))
        return NULL;
    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");

    /*
    printf("passed in: %d, %d, %d, %d\n", r,g,b,a );
    printf("what are: %d, %d, %d, %d\n", surf->format->Rmask, surf->format->Gmask, surf->format->Bmask, surf->format->Amask);
    */

    surf->format->Rmask = (Uint32)r;
    surf->format->Gmask = (Uint32)g;
    surf->format->Bmask = (Uint32)b;
    surf->format->Amask = (Uint32)a;

    Py_RETURN_NONE;
}





static PyObject*
surf_get_shifts (PyObject *self)
{
    SDL_Surface *surf = PySurface_AsSurface (self);

    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");
    return Py_BuildValue ("(iiii)", surf->format->Rshift, surf->format->Gshift,
                          surf->format->Bshift, surf->format->Ashift);
}


static PyObject*
surf_set_shifts (PyObject *self, PyObject *args)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    Uint64 r, g, b, a;

    if (!PyArg_ParseTuple (args, "(kkkk)", &r, &g, &b, &a))
        return NULL;
    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");

    surf->format->Rshift = (Uint8)r;
    surf->format->Gshift = (Uint8)g;
    surf->format->Bshift = (Uint8)b;
    surf->format->Ashift = (Uint8)a;

    Py_RETURN_NONE;
}







static PyObject*
surf_get_losses (PyObject *self)
{
    SDL_Surface *surf = PySurface_AsSurface (self);

    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");
    return Py_BuildValue ("(iiii)", surf->format->Rloss, surf->format->Gloss,
                          surf->format->Bloss, surf->format->Aloss);
}

static PyObject*
surf_subsurface (PyObject *self, PyObject *args)
{
    SDL_Surface *surf = PySurface_AsSurface (self);
    SDL_PixelFormat *format;
    GAME_Rect *rect, temp;
    SDL_Surface *sub;
    PyObject *subobj;
    int pixeloffset;
    char *startpixel;
    struct SubSurface_Data *data;

    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");
    if (surf->flags & SDL_OPENGL)
        return RAISE (PyExc_SDLError, "Cannot call on OPENGL Surfaces");

    format = surf->format;
    if (!(rect = GameRect_FromObject (args, &temp)))
        return RAISE (PyExc_ValueError, "invalid rectstyle argument");
    if (rect->x < 0 || rect->y < 0 || rect->x + rect->w > surf->w
        || rect->y + rect->h > surf->h)
        return RAISE (PyExc_ValueError,
                      "subsurface rectangle outside surface area");

    PySurface_Lock (self);

    pixeloffset = rect->x * format->BytesPerPixel + rect->y * surf->pitch;
    startpixel = ((char *) surf->pixels) + pixeloffset;

    sub = SDL_CreateRGBSurfaceFrom (startpixel, rect->w, rect->h,
                                    format->BitsPerPixel, surf->pitch,
                                    format->Rmask, format->Gmask,
                                    format->Bmask, format->Amask);

    PySurface_Unlock (self);

    if (!sub)
        return RAISE (PyExc_SDLError, SDL_GetError ());

    /* copy the colormap if we need it */
    if (surf->format->BytesPerPixel == 1 && surf->format->palette)
        SDL_SetPalette (sub, SDL_LOGPAL, surf->format->palette->colors, 0,
                        surf->format->palette->ncolors);
    if (surf->flags & SDL_SRCALPHA)
        SDL_SetAlpha (sub, surf->flags & SDL_SRCALPHA, format->alpha);
    if (surf->flags & SDL_SRCCOLORKEY)
        SDL_SetColorKey (sub, surf->flags & (SDL_SRCCOLORKEY | SDL_RLEACCEL),
                         format->colorkey);

    data = PyMem_New (struct SubSurface_Data, 1);
    if (!data)
        return NULL;

    subobj = PySurface_New (sub);
    if (!subobj) {
        PyMem_Del (data);
        return NULL;
    }
    Py_INCREF (self);
    data->owner = self;
    data->pixeloffset = pixeloffset;
    data->offsetx = rect->x;
    data->offsety = rect->y;
    ((PySurfaceObject *) subobj)->subsurface = data;

    return subobj;
}

static PyObject*
surf_get_offset (PyObject *self)
{
    struct SubSurface_Data *subdata;

    subdata = ((PySurfaceObject *) self)->subsurface;
    if (!subdata)
        return Py_BuildValue ("(ii)", 0, 0);
    return Py_BuildValue ("(ii)", subdata->offsetx, subdata->offsety);
}

static PyObject*
surf_get_abs_offset (PyObject *self)
{
    struct SubSurface_Data *subdata;
    PyObject *owner;
    int offsetx, offsety;

    subdata = ((PySurfaceObject *) self)->subsurface;
    if (!subdata)
        return Py_BuildValue ("(ii)", 0, 0);

    subdata = ((PySurfaceObject *) self)->subsurface;
    owner = subdata->owner;
    offsetx = subdata->offsetx;
    offsety = subdata->offsety;

    while (((PySurfaceObject *) owner)->subsurface) {
        subdata = ((PySurfaceObject *) owner)->subsurface;
        owner = subdata->owner;
        offsetx += subdata->offsetx;
        offsety += subdata->offsety;
    }
    return Py_BuildValue ("(ii)", offsetx, offsety);
}

static PyObject*
surf_get_parent (PyObject *self)
{
    struct SubSurface_Data *subdata;

    subdata = ((PySurfaceObject *) self)->subsurface;
    if (!subdata)
        Py_RETURN_NONE;

    Py_INCREF (subdata->owner);
    return subdata->owner;
}

static PyObject*
surf_get_abs_parent (PyObject *self)
{
    struct SubSurface_Data *subdata;
    PyObject *owner;

    subdata = ((PySurfaceObject *) self)->subsurface;
    if (!subdata) {
        Py_INCREF (self);
        return self;
    }

    subdata = ((PySurfaceObject *) self)->subsurface;
    owner = subdata->owner;

    while (((PySurfaceObject *) owner)->subsurface) {
        subdata = ((PySurfaceObject *) owner)->subsurface;
        owner = subdata->owner;
    }

    Py_INCREF (owner);
    return owner;
}

static PyObject *
surf_get_bounding_rect (PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *rect;
    SDL_Surface *surf = PySurface_AsSurface (self);
    SDL_PixelFormat *format = surf->format;
    Uint8 *pixels = (Uint8 *) surf->pixels;
    Uint8 *pixel;
    int x, y;
    int min_x, min_y, max_x, max_y;
    int min_alpha = 1;
    int found_alpha = 0;
    Uint8 r, g, b, a;
    int has_colorkey = 0;
    Uint8 keyr, keyg, keyb;

    char *kwids[] = { "min_alpha", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", kwids, &min_alpha))
       return RAISE (PyExc_ValueError,
                     "get_bounding_rect only accepts a single optional min_alpha argument");

    if (!surf)
        return RAISE (PyExc_SDLError, "display Surface quit");

    if (!PySurface_Lock (self))
        return RAISE (PyExc_SDLError, "could not lock surface");

    if (surf->flags & SDL_SRCCOLORKEY) {
        has_colorkey = 1;
        SDL_GetRGBA (surf->format->colorkey,
                     surf->format,
                     &keyr, &keyg, &keyb, &a);
    }

    pixels = (Uint8 *) surf->pixels;

    min_y = 0;
    min_x = 0;
    max_x = surf->w;
    max_y = surf->h;

    found_alpha = 0;
    for (y = max_y - 1; y >= min_y; --y) {
        for (x = min_x; x < max_x; ++x) {
            pixel = (pixels + y * surf->pitch) + x*format->BytesPerPixel;
            SDL_GetRGBA (*((Uint32*)pixel), surf->format, &r, &g, &b, &a);
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
            pixel = (pixels + y * surf->pitch) + x*format->BytesPerPixel;
            SDL_GetRGBA (*((Uint32*)pixel), surf->format, &r, &g, &b, &a);
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
            pixel = (pixels + y * surf->pitch) + x*format->BytesPerPixel;
            SDL_GetRGBA (*((Uint32*)pixel), surf->format, &r, &g, &b, &a);
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
            pixel = (pixels + y * surf->pitch) + x*format->BytesPerPixel;
            SDL_GetRGBA (*((Uint32*)pixel), surf->format, &r, &g, &b, &a);
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
    if (!PySurface_Unlock (self))
        return RAISE (PyExc_SDLError, "could not unlock surface");

    rect = PyRect_New4 (min_x, min_y, max_x - min_x, max_y - min_y);
    return rect;
}

static PyObject*
surf_get_buffer (PyObject *self)
{
    PyObject *buffer;
    PyObject *lock;
    SDL_Surface *surface = PySurface_AsSurface (self);
    Py_ssize_t length;

    length = (Py_ssize_t) surface->pitch * surface->h;

    buffer = PyBufferProxy_New (self, NULL, length, NULL);
    if (!buffer) {
        return RAISE (PyExc_SDLError,
            "could not acquire a buffer for the surface");
    }

    lock = PySurface_LockLifetime (self, buffer);
    if (!lock) {
        Py_DECREF (buffer);
        return RAISE (PyExc_SDLError, "could not lock surface");
    }
    ((PyBufferProxy *) buffer)->buffer = surface->pixels;
    ((PyBufferProxy *) buffer)->lock = lock;

    return buffer;
}

static PyObject *
_raise_get_view_ndim_error(int bitsize, SurfViewKind kind) {
    const char *name;

    switch (kind) {

    case VIEWKIND_RAW:
        name = "raw";
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
    default:
        PyErr_Format(PyExc_SystemError,
                     "pygame bug in _raise_get_view_ndim_error:"
                     " unknown view kind %d", (int) kind);
        return 0;
    }
    PyErr_Format(PyExc_ValueError,
         "unsupported bit depth %d for %s reference array",
                 bitsize, name);
    return 0;
}

static PyObject*
surf_get_view (PyObject *self, PyObject *args, PyObject *kwds)
{
#   define SURF_GET_VIEW_MAXDIM 3
    const int lilendian = (SDL_BYTEORDER == SDL_LIL_ENDIAN);
    const int maxdim = SURF_GET_VIEW_MAXDIM;
    Py_ssize_t shape[SURF_GET_VIEW_MAXDIM];
    Py_ssize_t strides[SURF_GET_VIEW_MAXDIM];
    char format[] = {'B', '\0', '\0'};
    Py_buffer view;
#   undef SURF_GET_VIEW_MAXDIM
    SurfViewKind view_kind = VIEWKIND_RAW;
    int ndim = maxdim;
    Py_ssize_t len = 0;
    SDL_Surface *surface = PySurface_AsSurface (self);
    int pixelsize;
    Py_ssize_t itemsize = 0;
    Uint8 *startpixel;
    Uint32 mask = 0;
    int pixelstep;
    int flags = 0;
    PyObject *proxy_obj;

    char *keywords[] = {"kind", 0};

    if (!PyArg_ParseTupleAndKeywords (args, kwds, "|O&", keywords,
                                      _view_kind, &view_kind)) {
        return 0;
    }

    if (!surface) {
        return RAISE (PyExc_SDLError, "display Surface quit");
    }

    view.readonly = 0;
    view.shape = shape;
    view.strides = strides;
    view.format = format;
    view.suboffsets = 0;
    view.internal = 0;

    startpixel = surface->pixels;
    pixelsize = surface->format->BytesPerPixel;
    shape[0] = (Py_ssize_t)surface->w;
    shape[1] = (Py_ssize_t)surface->h;
    strides[0] = (Py_ssize_t)pixelsize;
    strides[1] = (Py_ssize_t)surface->pitch;
    switch (view_kind) {

    case VIEWKIND_0D:
        ndim = 0;
        itemsize = pixelsize;
        if (strides[1] != itemsize * shape[0]) {
            PyErr_SetString (PyExc_ValueError,
                             "Surface data is not contiguous");
            return 0;
        }
        flags |= BUFPROXY_CONTIGUOUS;
        len = itemsize * shape[0] * shape[1];
        break;
    case VIEWKIND_2D:
        ndim = 2;
        itemsize = pixelsize;
        len = itemsize * shape[0] * shape[1];
        flags |= BUFPROXY_F_ORDER;
        if (strides[1] == shape[0] * itemsize) {
            flags |= BUFPROXY_CONTIGUOUS;
        }
        break;
    case VIEWKIND_3D:
        if (pixelsize < 3) {
            return _raise_get_view_ndim_error (pixelsize * 8, view_kind);
        }
        ndim = 3;
        itemsize = 1;
        shape[2] = 3;
        len = itemsize * shape[0] * shape[1] * shape[2];
        if (surface->format->Rmask == 0xff0000 &&
            surface->format->Gmask == 0x00ff00 &&
            surface->format->Bmask == 0x0000ff)   {
            strides[2] = lilendian ? -1 : 1;
            startpixel += lilendian ? 2 : 1;
        }
        else if (surface->format->Bmask == 0xff0000 &&
                 surface->format->Gmask == 0x00ff00 &&
                 surface->format->Rmask == 0x0000ff)   {
            strides[2] = lilendian ? 1 : -1;
            startpixel += lilendian ? 0 : (pixelsize - 1);
        }
        else {
            return RAISE (PyExc_ValueError,
                          "unsupport colormasks for 3D reference array");
        }
        if (strides[1] == shape[2] * shape[1]) {
            flags |= BUFPROXY_CONTIGUOUS;
        }
        break;
    case VIEWKIND_RED:
        mask = surface->format->Rmask;
        break;
    case VIEWKIND_GREEN:
        mask = surface->format->Gmask;
        break;
    case VIEWKIND_BLUE:
        mask = surface->format->Bmask;
        break;
    case VIEWKIND_ALPHA:
        mask = surface->format->Amask;
        break;
    case VIEWKIND_RAW:
        ndim = 0;
        itemsize = 1;
        flags |= BUFPROXY_CONTIGUOUS; /* Assumes knowledgable consumers */
        len = surface->pitch * surface->h;
        break;
    default:
        PyErr_Format (PyExc_SystemError,
                      "pygame bug in surf_get_view:"
                      " unrecognized view kind %d", (int)view_kind);
        return 0;
    }
    if (!itemsize) {
        /* Color plane */
        ndim = 2;
        pixelstep = pixelsize;
        itemsize = 1;
        len = itemsize * shape[0] * shape[1];
        flags |= BUFPROXY_F_ORDER;
        switch (mask) {

        case 0x000000ffU:
            startpixel += lilendian ? 0 : 3;
            break;
        case 0x0000ff00U:
            startpixel += lilendian ? 1 : 2;
            break;
        case 0x00ff0000U:
            startpixel += lilendian ? 2 : 1;
            break;
        case 0xff000000U:
            startpixel += lilendian ? 3 : 0;
            break;
        default:
            return RAISE (PyExc_ValueError,
                          "unsupported colormasks for alpha reference array");
        }
    }
    view.ndim = ndim;
    view.buf = startpixel;
    view.itemsize = itemsize;
    switch (itemsize) {

    case 1:
        break; /* default */
    case 2:
        format[0] = '=';
        format[1] = 'H';
        break;
    case 3:
        format[0] = 's';
        break;
    case 4:
        format[0] = '=';
        format[1] = 'I';
        break;
    default:
        /* Should not get here. */
        PyErr_Format (PyExc_SystemError,
                      "Pygame: Surface.get_item: unrecognized itemsize %d",
                      (int)itemsize);
        return 0;
    }
    view.len = len;
    view.obj = self;
    Py_INCREF (self);

    proxy_obj = PgBufproxy_New (&view, flags, _view_before, _view_after);
    if (proxy_obj && view_kind == VIEWKIND_RAW) {
        if (PgBufproxy_Trip (proxy_obj)) {
            Py_DECREF (proxy_obj);
            proxy_obj = 0;
        }
    }
    return proxy_obj;
}

static int
_view_kind (PyObject *obj, void *view_kind_vptr)
{
    unsigned long ch;
    SurfViewKind *view_kind_ptr = (SurfViewKind *)view_kind_vptr;

    if (PyUnicode_Check (obj)) {
        if (PyUnicode_GET_SIZE (obj) != 1) {
            PyErr_SetString (PyExc_TypeError,
                             "expected a length 1 string for argument 1");
            return 0;
        }
        ch = *PyUnicode_AS_UNICODE (obj);
    }
    else if (Bytes_Check (obj)) {
        if (Bytes_GET_SIZE (obj) != 1) {
            PyErr_SetString (PyExc_TypeError,
                             "expected a length 1 string for argument 1");
            return 0;
        }
        ch = *Bytes_AS_STRING (obj);
    }
    else {
        PyErr_Format (PyExc_TypeError,
                      "expected a length one string for argument 1: got '%s'",
                      Py_TYPE (obj)->tp_name);
        return 0;
    }
    switch (ch) {

    case '0':
        *view_kind_ptr = VIEWKIND_0D;
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
    case '&':
        *view_kind_ptr = VIEWKIND_RAW;
        break;
    default:
        PyErr_Format (PyExc_TypeError,
                      "unrecognized view kind '%c' for argument 1", (int)ch);
        return 0;
    }
    return 1;
}

static int
_view_before (PyObject *view)
{
    PyObject *surf = PgBufproxy_GetParent (view);
    int rcode;

    rcode = PySurface_LockBy (surf, view) ? 0 : -1;
    Py_DECREF (surf);
    return rcode;
}

static void
_view_after (PyObject *view)
{
    PyObject *surf = PgBufproxy_GetParent(view);

    PySurface_UnlockBy (surf, view);
    Py_DECREF (surf);
}

static PyObject *
surf_get_pixels_address(PyObject *self, PyObject *closure)
{
    SDL_Surface *surface = PySurface_AsSurface(self);
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
surface_move (Uint8 *src, Uint8 *dst, int h, int span,
          int srcpitch, int dstpitch)
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
surface_do_overlap (SDL_Surface *src, SDL_Rect *srcrect,
            SDL_Surface *dst, SDL_Rect *dstrect)
{
    Uint8 *srcpixels;
    Uint8 *dstpixels;
    int srcx = srcrect->x, srcy = srcrect->y;
    int dstx = dstrect->x, dsty = dstrect->y;
    int x, y;
    int w = srcrect-> w, h= srcrect->h;
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

    srcpixels = ((Uint8 *) src->pixels + src->offset +
          srcy * src->pitch +
          srcx * src->format->BytesPerPixel);
    dstpixels = ((Uint8 *) dst->pixels + src->offset +
          dsty * dst->pitch +
          dstx * dst->format->BytesPerPixel);

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
PySurface_Blit (PyObject * dstobj, PyObject * srcobj, SDL_Rect * dstrect,
                SDL_Rect * srcrect, int the_args)
{
    SDL_Surface *src = PySurface_AsSurface (srcobj);
    SDL_Surface *dst = PySurface_AsSurface (dstobj);
    SDL_Surface *subsurface = NULL;
    int result, suboffsetx = 0, suboffsety = 0;
    SDL_Rect orig_clip, sub_clip;

    /* passthrough blits to the real surface */
    if (((PySurfaceObject *) dstobj)->subsurface) {
        PyObject *owner;
        struct SubSurface_Data *subdata;

        subdata = ((PySurfaceObject *) dstobj)->subsurface;
        owner = subdata->owner;
        subsurface = PySurface_AsSurface (owner);
        suboffsetx = subdata->offsetx;
        suboffsety = subdata->offsety;

        while (((PySurfaceObject *) owner)->subsurface) {
            subdata = ((PySurfaceObject *) owner)->subsurface;
            owner = subdata->owner;
            subsurface = PySurface_AsSurface (owner);
            suboffsetx += subdata->offsetx;
            suboffsety += subdata->offsety;
        }

        SDL_GetClipRect (subsurface, &orig_clip);
        SDL_GetClipRect (dst, &sub_clip);
        sub_clip.x += suboffsetx;
        sub_clip.y += suboffsety;
        SDL_SetClipRect (subsurface, &sub_clip);
        dstrect->x += suboffsetx;
        dstrect->y += suboffsety;
        dst = subsurface;
    }
    else {
        PySurface_Prep (dstobj);
        subsurface = NULL;
    }

    PySurface_Prep (srcobj);

    /* see if we should handle alpha ourselves */
    if (dst->format->Amask && (dst->flags & SDL_SRCALPHA) &&
        !(src->format->Amask && !(src->flags & SDL_SRCALPHA)) &&
        /* special case, SDL works */
        (dst->format->BytesPerPixel == 2 || dst->format->BytesPerPixel == 4)) {
    /* Py_BEGIN_ALLOW_THREADS */
        result = pygame_AlphaBlit (src, srcrect, dst, dstrect, the_args);
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
          surface_do_overlap (src, srcrect, dst, dstrect))) {
    /* Py_BEGIN_ALLOW_THREADS */
        result = pygame_Blit (src, srcrect, dst, dstrect, the_args);
    /* Py_END_ALLOW_THREADS */
    }
    /* can't blit alpha to 8bit, crashes SDL */
    else if (dst->format->BytesPerPixel == 1 &&
         (src->format->Amask || src->flags & SDL_SRCALPHA)) {
    /* Py_BEGIN_ALLOW_THREADS */
    if (src->format->BytesPerPixel == 1) {
        result = pygame_Blit (src, srcrect, dst, dstrect, 0);
    }
    else if (SDL_WasInit (SDL_INIT_VIDEO)) {
        src = SDL_DisplayFormat (src);
        if (src) {
        result = SDL_BlitSurface (src, srcrect, dst, dstrect);
        SDL_FreeSurface (src);
        }
        else {
        result = -1;
        }
    }
    else {
        SDL_PixelFormat *fmt = src->format;
        SDL_PixelFormat newfmt;

        newfmt.palette = 0;  /* Set NULL (or SDL gets confused) */
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
        src = SDL_ConvertSurface (src, &newfmt, SDL_SWSURFACE);
        if (src) {
            result = SDL_BlitSurface (src, srcrect, dst, dstrect);
            SDL_FreeSurface (src);
        }
        else {
            result = -1;
        }
    }
    /* Py_END_ALLOW_THREADS */
    }
    else {
    /* Py_BEGIN_ALLOW_THREADS */
        result = SDL_BlitSurface (src, srcrect, dst, dstrect);
    /* Py_END_ALLOW_THREADS */
    }

    if (subsurface) {
        SDL_SetClipRect (subsurface, &orig_clip);
        dstrect->x -= suboffsetx;
        dstrect->y -= suboffsety;
    }
    else
        PySurface_Unprep (dstobj);
    PySurface_Unprep (srcobj);

    if (result == -1)
        RAISE (PyExc_SDLError, SDL_GetError ());
    if (result == -2)
        RAISE (PyExc_SDLError, "Surface was lost");

    return result != 0;
}

static PyMethodDef _surface_methods[] =
{
    { NULL, NULL, 0, NULL }
};

MODINIT_DEFINE (surface)
{
    PyObject *module, *dict, *apiobj, *lockmodule;
    int ecode;
    static void* c_api[PYGAMEAPI_SURFACE_NUMSLOTS];

#if PY3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "surface",
        DOC_PYGAMESURFACE,
        -1,
        _surface_methods,
        NULL, NULL, NULL, NULL
    };
#endif

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }
    import_pygame_color ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }
    import_pygame_rect ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }
    import_pygame_bufferproxy ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }
    import_pygame_view ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }

    /* import the surflock module manually */
    lockmodule = PyImport_ImportModule (IMPPREFIX "surflock");
    if (lockmodule != NULL) {
        PyObject *_dict = PyModule_GetDict (lockmodule);
        PyObject *_c_api = PyDict_GetItemString (_dict, PYGAMEAPI_LOCAL_ENTRY);

        if (PyCapsule_CheckExact (_c_api)) {
            int i;
            void **localptr =
                (void *)PyCapsule_GetPointer (_c_api,
                                              PG_CAPSULE_NAME("surflock"));

            for (i = 0; i < PYGAMEAPI_SURFLOCK_NUMSLOTS; ++i)
                PyGAME_C_API[i + PYGAMEAPI_SURFLOCK_FIRSTSLOT] = localptr[i];
        }
        Py_DECREF (lockmodule);
    }
    else {
        MODINIT_ERROR;
    }

    /* type preparation */
    if (PyType_Ready(&PySurface_Type) < 0) {
        MODINIT_ERROR;
    }

    /* create the module */
#if PY3
    module = PyModule_Create (&_module);
#else
    module = Py_InitModule3 (MODPREFIX "surface", _surface_methods, DOC_PYGAMESURFACE);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    dict = PyModule_GetDict (module);

    if (PyDict_SetItemString (dict, "SurfaceType", (PyObject *) &PySurface_Type)) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    if (PyDict_SetItemString (dict, "Surface", (PyObject *) &PySurface_Type)) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }

    /* export the c api */
    c_api[0] = &PySurface_Type;
    c_api[1] = PySurface_New;
    c_api[2] = PySurface_Blit;
    apiobj = encapsulate_api (c_api, "surface");
    if (apiobj == NULL) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString (dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF (apiobj);
    if (ecode) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    /* Py_INCREF (PySurface_Type.tp_dict); INCREF's done in SetItemString */
    if (PyDict_SetItemString (dict, "_dict", PySurface_Type.tp_dict)) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN (module);
}
