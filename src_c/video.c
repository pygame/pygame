/*
  pygame - Python Game Library

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

#include "pygame.h"
#include "pgcompat.h"
#include <structmember.h>

static PyTypeObject pgRenderer_Type;
static PyTypeObject pgTexture_Type;

typedef struct pgRendererObject pgRendererObject;

#define pgRenderer_Check(x) (((PyObject*)(x))->ob_type == &pgRenderer_Type)
#define pgTexture_Check(x) (((PyObject*)(x))->ob_type == &pgTexture_Type)

static PyObject *drawfnc_str = NULL;

typedef struct {
    PyObject_HEAD
    SDL_Texture *texture;
    pgRendererObject *renderer;
    int width;
    int height;
    pgColorObject *color;
    Uint8 alpha;
} pgTextureObject;

typedef struct {
    PyObject_HEAD
    SDL_Window *_win;
} pgWindowObject;

static int
pgTexture_DrawObj(pgTextureObject *self, PyObject *srcrect, PyObject *dstrect);

static int
pgTexture_Draw(pgTextureObject *self,
               SDL_Rect *srcrect, SDL_Rect *dstrect,
               float angle, const int * origin,
               int flipX, int flipY);


/*
 * RENDERER
 */

struct pgRendererObject {
    PyObject_HEAD
    pgWindowObject *window;
    SDL_Renderer *renderer;
    pgColorObject *drawcolor;
    pgTextureObject *target;
};

static PyObject *
pg_renderer_get_viewport(pgRendererObject *self, PyObject *args);


static PyObject *
pg_renderer_clear(pgRendererObject *self, PyObject *args)
{
    if (SDL_RenderClear(self->renderer) < 0) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    Py_RETURN_NONE;
}

static PyObject *
pg_renderer_present(pgRendererObject *self, PyObject *args)
{
    SDL_RenderPresent(self->renderer);
    Py_RETURN_NONE;
}

static PyObject *
pg_renderer_blit(pgRendererObject *self, PyObject *args, PyObject *kw)
{
    /* only for compatibility with pygame.sprite */
    char* keywords[] = {
        "source",
        "dest",
        "area",
        "special_flags",
        NULL
    };
    PyObject *source;
    PyObject *dest = Py_None;
    PyObject *area = Py_None;
    int flags = 0;
    PyObject *ret;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|OOi", keywords,
                                     &source, &dest, &area, &flags))
    {
        return NULL;
    }

    if (pgTexture_Check(source)) {
        if (pgTexture_DrawObj((pgTextureObject*) source, area, dest))
            return NULL;
        goto RETURN_VIEWPORT;
    }
    if (!PyObject_HasAttr(source, drawfnc_str)) {
        return RAISE(PyExc_TypeError, "source must be drawable");
    }

    ret = PyObject_CallMethodObjArgs(source, drawfnc_str, area, dest, NULL);
    if (ret == NULL) {
        return NULL;
    }
    Py_DECREF(ret);

    if (dest == Py_None) {

RETURN_VIEWPORT:

        return pg_renderer_get_viewport(self, NULL);
    }
    Py_INCREF(dest);
    return dest;
}

static PyObject *
pg_renderer_get_viewport(pgRendererObject *self, PyObject *args)
{
    SDL_Rect rect;
    SDL_RenderGetViewport(self->renderer, &rect);
    return pgRect_New(&rect);
}

static PyObject *
pg_renderer_set_viewport(pgRendererObject *self, PyObject *area)
{
    SDL_Rect rect;

    if (area == NULL || area == Py_None) {
        if (SDL_RenderSetViewport(self->renderer, NULL) < 0) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
        Py_RETURN_NONE;
    }

    if (pgRect_FromObject(area, &rect) == NULL) {
        return RAISE(PyExc_TypeError, "the argument must be a rectangle "
                                      "or None");
    }
    if (SDL_RenderSetViewport(self->renderer, &rect) < 0) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    Py_RETURN_NONE;
}

/*
IMPORTANT!

Renderer needs to add

def draw_line(self, p1, p2):
    # https://wiki.libsdl.org/SDL_RenderDrawLine
    res = SDL_RenderDrawLine(self._renderer,
                                p1[0], p1[1],
                                p2[0], p2[1])
    if res < 0:
        raise error()

def draw_point(self, point):
    # https://wiki.libsdl.org/SDL_RenderDrawPoint
    res = SDL_RenderDrawPoint(self._renderer,
                                point[0], point[1])
    if res < 0:
        raise error()

def draw_rect(self, rect):
    # https://wiki.libsdl.org/SDL_RenderDrawRect
    cdef SDL_Rect _rect
    cdef SDL_Rect *rectptr = pgRect_FromObject(rect, &_rect)
    if rectptr == NULL:
        raise TypeError('expected a rectangle')
    res = SDL_RenderDrawRect(self._renderer, rectptr)
    if res < 0:
        raise error()
*/

static PyObject *
pg_renderer_draw_line(pgRendererObject *self, PyObject *args, PyObject *kw) {}

static PyObject *
pg_renderer_draw_point(pgRendererObject *self, PyObject *args, PyObject *kw) {}

static PyObject *
pg_renderer_draw_rect(pgRendererObject *self, PyObject *args, PyObject *kw) {}

static PyObject *
pg_renderer_fill_rect(pgRendererObject *self, PyObject *args, PyObject *kw) {
    // https://wiki.libsdl.org/SDL_RenderFillRect
    static char *keywords[] = {
        "rect",
        NULL
    };

    PyObject *area = NULL;
    SDL_Rect rect;
    SDL_Rect *rectptr = NULL;

    if(!PyArg_ParseTupleAndKeywords(args, kw, "O", keywords, &area)) {
        return NULL;
    }
    
    rectptr = pgRect_FromObject(area, &rect);
    if (rectptr == NULL)
        return RAISE(PyExc_TypeError, "expected a rectangle");

    if (SDL_RenderFillRect(self->renderer, rectptr) < 0) {
        RAISE(pgExc_SDLError, SDL_GetError());
        return -1;
    }
    Py_RETURN_NONE;
}

static PyMethodDef pg_renderer_methods[] = {
    { "clear", (PyCFunction)pg_renderer_clear, METH_NOARGS, NULL /* TODO */ },
    { "present", (PyCFunction)pg_renderer_present, METH_NOARGS, NULL /* TODO */ },
    { "blit", (PyCFunction)pg_renderer_blit, METH_VARARGS | METH_KEYWORDS, NULL /* TODO */ },
    { "get_viewport", (PyCFunction)pg_renderer_get_viewport, METH_NOARGS, NULL /* TODO */ },
    { "set_viewport", (PyCFunction)pg_renderer_set_viewport, METH_O, NULL /* TODO */ },
    { "draw_line", (PyCFunction)pg_renderer_draw_line, METH_VARARGS | METH_KEYWORDS, NULL /* TODO */ },
    { "draw_point", (PyCFunction)pg_renderer_draw_point, METH_VARARGS | METH_KEYWORDS, NULL /* TODO */ },
    { "draw_rect", (PyCFunction)pg_renderer_draw_rect, METH_VARARGS | METH_KEYWORDS, NULL /* TODO */ },
    { "fill_rect", (PyCFunction)pg_renderer_fill_rect, METH_VARARGS | METH_KEYWORDS, NULL /* TODO */ },
    { NULL }
};

static PyObject *
pg_renderer_get_color(pgRendererObject *self, void *closure)
{
    Py_INCREF(self->drawcolor);
    return (PyObject*)self->drawcolor;
}

static int
pg_renderer_set_color(pgRendererObject *self, PyObject *val, void *closure)
{
    Uint8 *colarray = pgColor_AsArray(self->drawcolor);
    if (!pg_RGBAFromColorObj(val, colarray)) {
        RAISE(PyExc_TypeError, "expected a color (sequence of color object)");
        return -1;
    }

    if (SDL_SetRenderDrawColor(self->renderer,
                               colarray[0], colarray[1],
                               colarray[2], colarray[3]) < 0) {
        RAISE(pgExc_SDLError, SDL_GetError());
        return -1;
    }
    return 0;
}

static PyObject *
pg_renderer_get_target(pgRendererObject *self, void *closure)
{
    //TODO
}

static PyObject *
pg_renderer_set_target(pgRendererObject *self, PyObject *val, void *closure)
{
    pgTextureObject* newtarget = NULL;

    if(PyObject_IsInstance(val, (PyObject*)&pgTexture_Type))
        newtarget = val;

    else if (val == Py_None) {}

    else {
        return NULL; //TODO: raise TypeError('target must be a Texture or None')
    }
        
    self->target = newtarget;

    //TODO: error checking SDL_SetRendererTarget calls
    if (self->target)
        SDL_SetRenderTarget(self->renderer, self->target->texture);
    else //target is NULL
        SDL_SetRenderTarget(self->renderer, NULL);
}

static PyGetSetDef pg_renderer_getset[] = {
    { "draw_color", (getter)pg_renderer_get_color, (setter)pg_renderer_set_color, NULL /*TODO*/, NULL },
    { "target", (getter)pg_renderer_get_target, (setter)pg_renderer_set_target, NULL /*TODO*/, NULL },
    { NULL }
};

static int
pg_renderer_init(pgRendererObject *self, PyObject *args, PyObject *kw)
{
    char* keywords[] = {
        "window",
        "index",
        "accelerated",
        "vsync",
        "target_texture",
        NULL
    };
    PyObject *winobj;
    int index = -1;
    int accelerated = -1;
    int vsync = 0;
    int target_texture = 0;
    int flags = 0;

#if PY3
    const char *formatstr = "O|iipp";
#else
    const char *formatstr = "O|iiii";
#endif

    if (!PyArg_ParseTupleAndKeywords(args, kw, formatstr, keywords,
                                     &winobj,
                                     &index, &accelerated, &vsync,
                                     &target_texture)) {
        return -1;
    }

    /* TODO: check window */
    Py_INCREF(winobj);
    self->window = (pgWindowObject*)winobj;

    if (accelerated > 0)
        flags |= SDL_RENDERER_ACCELERATED;
    else if (accelerated == 0)
        flags |= SDL_RENDERER_SOFTWARE;
    if (vsync)
        flags |= SDL_RENDERER_PRESENTVSYNC;
    if (target_texture)
        flags |= SDL_RENDERER_TARGETTEXTURE;

    self->renderer =
        SDL_CreateRenderer(self->window->_win, index, flags);
    if (self->renderer == NULL) {
        RAISE(pgExc_SDLError, SDL_GetError());
        return 0;
    }

    return 0;
}

static PyObject *
pg_renderer_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Uint8 rgba[] = { 255, 255, 255, 255 };
    pgColorObject *col = (pgColorObject*) pgColor_NewLength(rgba, 4);
    pgRendererObject *obj;
    if (!col) {
        return NULL;
    }
    obj = (pgRendererObject*) type->tp_alloc(type, 0);
    obj->drawcolor = col;
    return (PyObject*)obj;
}

static void
pg_renderer_dealloc(pgRendererObject *self)
{
    if (self->renderer) {
        SDL_DestroyRenderer(self->renderer);
        self->renderer = NULL;
    }
    Py_XDECREF(self->window);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyTypeObject pgRenderer_Type = {
    PyVarObject_HEAD_INIT(NULL,0)
    "pygame.Renderer", /*name*/
    sizeof(pgRendererObject), /*basicsize*/
    0, /*itemsize*/
    (destructor)pg_renderer_dealloc, /*dealloc*/
    NULL, /*print*/
    NULL, /*getattr*/
    NULL, /*setattr*/
    NULL, /*compare/reserved*/
    NULL, /*repr*/
    NULL, /*as_number*/
    NULL, /*as_sequence*/
    NULL, /*as_mapping*/
    NULL, /*hash*/
    NULL, /*call*/
    NULL, /*str*/
    0L,
    0L,
    0L,
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    NULL, /* TODO: docstring */
    NULL, /* tp_traverse */
    NULL, /* tp_clear */
    NULL, /* tp_richcompare */
    NULL, /* tp_weaklistoffset */
    NULL, /* tp_iter */
    NULL, /* tp_iternext */
    pg_renderer_methods, /* tp_methods */
    NULL, /* tp_members */
    pg_renderer_getset, /* tp_getset */
    NULL, /* tp_base */
    NULL, /* tp_dict */
    NULL, /* tp_descr_get */
    NULL, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc)pg_renderer_init, /* tp_init */
    NULL, /* tp_alloc */
    pg_renderer_new /* tp_new */
};

/*
 * TEXTURE
 */


static pgTextureObject *
pg_texture_from_surface(PyObject *self, PyObject *args, PyObject *kw);

static PyMemberDef pg_texture_members[] = {
    { "renderer", T_OBJECT_EX, offsetof(pgTextureObject, renderer), READONLY, NULL /* TODO */ },
    { "width", T_INT, offsetof(pgTextureObject, width), READONLY, NULL /* TODO */ },
    { "height", T_INT, offsetof(pgTextureObject, height), READONLY, NULL /* TODO */ },
    { NULL }
};

static PyObject *
pg_texture_get_rect(pgTextureObject *self, PyObject *args, PyObject *kw)
{
    PyObject *rectobj;
    PyObject *key, *value;
    Py_ssize_t pos = 0;

    rectobj = pgRect_New4(0, 0, self->width, self->height);
    if (!rectobj)
        return NULL;

    if (kw) {
#if PY3
        if (PyArg_ValidateKeywordArguments(kw)) {
            Py_DECREF(rectobj);
            return NULL;
        }
#endif /* PY3 */
        while (PyDict_Next(kw, &pos, &key, &value)) {
            if (PyObject_SetAttr(rectobj, key, value)) {
                Py_DECREF(rectobj);
                return NULL;
            }
        }
    }
    return rectobj;
}

static int
pgTexture_Draw(pgTextureObject *self,
               SDL_Rect *srcrect, SDL_Rect *dstrect,
               float angle, const int * origin,
               int flipX, int flipY)
{
    SDL_RendererFlip flip = SDL_FLIP_NONE;
    SDL_Point pointorigin;

    if (origin) {
        pointorigin.x = origin[0];
        pointorigin.y = origin[1];
    }

    if (flipX)
        flip |= SDL_FLIP_HORIZONTAL;
    if (flipY)
        flip |= SDL_FLIP_VERTICAL;

    if (SDL_RenderCopyEx(self->renderer->renderer, self->texture,
                         srcrect, dstrect,
                         angle, origin ? &pointorigin : NULL,
                         flip) < 0) {
        RAISE(pgExc_SDLError, SDL_GetError());
        return -1;
    }
    return 0;
}

static int
pgTexture_DrawObj(pgTextureObject *self, PyObject *srcrect, PyObject *dstrect)
{
    SDL_Rect src;
    SDL_Rect dst;
    SDL_Rect *srcptr = NULL;
    SDL_Rect *dstptr = NULL;

    if (srcrect && srcrect != Py_None) {
        if (!(srcptr = pgRect_FromObject(srcrect, &src))) {
            RAISE(PyExc_TypeError, "srcrect must be a rectangle");
            return -1;
        }
    }
    if (dstrect && dstrect != Py_None) {
        if (!(dstptr = pgRect_FromObject(dstrect, &dst))) {
            if (pg_TwoIntsFromObj(dstrect, &dst.x, &dst.y)) {
                dst.w = ((pgTextureObject*) self)->width;
                dst.h = ((pgTextureObject*) self)->height;
                dstptr = &dst;
            }
            else {
                RAISE(PyExc_TypeError, "dstrect must be a rectangle or "
                                       "a position");
                return -1;
            }
        }
    }

    if (pgTexture_Draw(self,
                       srcptr, dstptr,
                       0, NULL,
                       0, 0))
    {
        return -1;
    }
    return 0;
}

static PyObject *
pg_texture_draw(pgTextureObject *self, PyObject *args, PyObject *kw)
{

    printf("drawing a texture get hyped!\n");
    static char *keywords[] = {
        "srcrect",
        "dstrect",
        "angle",
        "origin",
        "flipX",
        "flipY",
        NULL
    };
    PyObject *srcrect = NULL;
    PyObject *dstrect = NULL;
    float angle = .0f;
    PyObject *originobj = NULL;
    int flipX = 0;
    int flipY = 0;
    int origin[2];
    SDL_Rect src, dst;
    SDL_Rect *srcptr = NULL;
    SDL_Rect *dstptr = NULL;

#if PY3
    const char *format = "|OOfOpp";
#else
    const char *format = "|OOfOii";
#endif

    if (!PyArg_ParseTupleAndKeywords(args, kw, format, keywords,
                                     &srcrect, &dstrect,
                                     &angle, &originobj,
                                     &flipX, &flipY)) {
        return NULL;
    }
    if (srcrect && srcrect != Py_None) {
        if (!(srcptr = pgRect_FromObject(srcrect, &src)))
            return RAISE(PyExc_TypeError, "srcrect must be a rectangle");
    }
    if (dstrect && dstrect != Py_None) {
        if (!(dstptr = pgRect_FromObject(dstrect, &dst))) {
            if (pg_TwoIntsFromObj(dstrect, &dst.x, &dst.y)) {
                dst.w = self->width;
                dst.h = self->height;
                dstptr = &dst;
            }
            else {
                return RAISE(PyExc_TypeError, "dstrect must be a rectangle or "
                                              "a position");
            }
        }
    }
    if (originobj) {
        if (originobj == Py_None) {
            originobj = NULL;
        }
        else if (!pg_TwoIntsFromObj(originobj, &origin[0], &origin[1])) {
            return RAISE(PyExc_TypeError, "origin must be a pair of two "
                                          "numbers");
        }
    }

    if (pgTexture_Draw(self, srcptr, dstptr, angle, originobj ? origin : NULL, flipX, flipY)) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
pg_texture_update(pgTextureObject *self, PyObject *args, PyObject *kw) 
{
    static char *keywords[] = {
        "surface",
        "area",
        NULL
    };

    pgSurfaceObject *surfobj = NULL;
    SDL_Surface *surf = NULL;
    PyObject *area = Py_None;
    SDL_Rect rect;
    SDL_Rect *rectptr = NULL;

    if(!PyArg_ParseTupleAndKeywords(args, kw, "O!|O", keywords, &pgSurface_Type, &surfobj, &area)) {
        return NULL;
    }

    if (!pgSurface_Check(surfobj)) {
        RAISE(PyExc_TypeError, "not a surface");
        return NULL;
    }

    surf = pgSurface_AsSurface(surfobj);

    rectptr = pgRect_FromObject(area, &rect);
    if (rectptr == NULL && area != Py_None) {
        return RAISE(PyExc_TypeError, "area must be a rectangle or None");
    }

    if (SDL_UpdateTexture(self->texture, rectptr, surf->pixels, surf->pitch) < 0) {
        RAISE(pgExc_SDLError, SDL_GetError());
        return -1;       
    }
    Py_RETURN_NONE;
}

static PyMethodDef pg_texture_methods[] = {
    { "from_surface", (PyCFunction)pg_texture_from_surface, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL /* TODO */ },
    { "get_rect", (PyCFunction)pg_texture_get_rect, METH_VARARGS | METH_KEYWORDS, NULL /* TODO */ },
    { "draw", (PyCFunction)pg_texture_draw, METH_VARARGS | METH_KEYWORDS, NULL /* TODO */ },
    { "update", (PyCFunction)pg_texture_update, METH_VARARGS | METH_KEYWORDS, NULL /* TODO */ },
    { NULL }
};

static PyObject *
pg_texture_get_color(pgTextureObject *self, void *closure)
{
    Py_INCREF(self->color);
    return (PyObject*) self->color;
}

static int
pg_texture_set_color(pgTextureObject *self, PyObject *val, void *closure)
{
    Uint8 *colarray = pgColor_AsArray(self->color);
    if (!pg_RGBAFromColorObj(val, colarray)) {
        RAISE(PyExc_TypeError, "expected a color (sequence of color object)");
        return -1;
    }

    if (SDL_SetTextureColorMod(self->texture,
                               colarray[0], colarray[1], colarray[2]) < 0) {
        RAISE(pgExc_SDLError, SDL_GetError());
        return -1;
    }
    return 0;
}

static PyObject *
pg_texture_get_alpha(pgTextureObject *self, void *closure)
{
    return PyLong_FromLong(self->alpha);
}

static int
pg_texture_set_alpha(pgTextureObject *self, PyObject *val, void *closure)
{
    int alpha;
    if ((alpha = PyLong_AsLong(val)) == -1 && PyErr_Occurred()) {
        RAISE(PyExc_TypeError, "alpha should be an integer");
        return -1;
    }
    if (alpha < 0 || alpha > 255) {
        RAISE(PyExc_ValueError, "alpha should be between 0 and 255 (inclusive)");
        return -1;
    }

    self->alpha = (Uint8)alpha;
    if (SDL_SetTextureAlphaMod(self->texture, self->alpha) < 0) {
        RAISE(pgExc_SDLError, SDL_GetError());
        return -1;
    }
    return 0;
}

static PyObject *
pg_texture_get_mode(pgTextureObject *self, void *closure)
{
    SDL_BlendMode mode;
    if (SDL_GetTextureBlendMode(self->texture, &mode) < 0) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    return PyLong_FromLong(mode);
}

static int
pg_texture_set_mode(pgTextureObject *self, PyObject *val, void *closure)
{
    int mode;
    if ((mode = PyLong_AsLong(val)) == -1 && PyErr_Occurred()) {
        RAISE(PyExc_TypeError, "mode should be an integer");
        return -1;
    }
    /* TODO: check values */

    if (SDL_SetTextureBlendMode(self->texture, (SDL_BlendMode)mode) < 0) {
        RAISE(pgExc_SDLError, SDL_GetError());
        return -1;
    }
    return 0;
}

static PyGetSetDef pg_texture_getset[] = {
    { "color", (getter)pg_texture_get_color, (setter)pg_texture_set_color, NULL /*TODO*/, NULL },
    { "alpha", (getter)pg_texture_get_alpha, (setter)pg_texture_set_alpha, NULL /*TODO*/, NULL },
    { "blend_mode", (getter)pg_texture_get_mode, (setter)pg_texture_set_mode, NULL /*TODO*/, NULL },
    { NULL }
};

static Uint32
format_from_depth(int depth)
{
    Uint32 Rmask, Gmask, Bmask, Amask;
    switch (depth) {
    case 16:
        Rmask = 0xF << 8;
        Gmask = 0xF << 4;
        Bmask = 0xF;
        Amask = 0xF << 12;
        break;
    case 0:
    case 32:
        Rmask = 0xF << 16;
        Gmask = 0xF << 8;
        Bmask = 0xF;
        Amask = 0xF << 24;
        break;
    default:
        RAISE(PyExc_ValueError,
              "no standard masks exist for given bitdepth with alpha");
        return 0;
    }
    return SDL_MasksToPixelFormatEnum(depth, Rmask, Gmask, Bmask, Amask);
}

static int
pg_texture_init(pgTextureObject *self, PyObject *args, PyObject *kw)
{
    char* keywords[] = {
        "renderer",
        "size",
        "depth",
        "static",
        "streaming",
        "target",
        NULL
    };
    PyObject *sizeobj;
    PyObject *renderobj;
    int depth = 0;
    int static_ = 0;
    int streaming = 0;
    int target = 0;
    int format;
    int access;

#if PY3
    const char *formatstr = "OO|Ippp";
#else
    const char *formatstr = "OO|Iiii";
#endif

    if (!PyArg_ParseTupleAndKeywords(args, kw, formatstr, keywords,
                                     &renderobj,
                                     &sizeobj,
                                     &depth,
                                     &static_, &streaming, &target)) {
        return -1;
    }

    if (!pgRenderer_Check(renderobj)) {
        RAISE(PyExc_TypeError, "not a renderer object");
        return -1;
    }
    self->renderer = (pgRendererObject*) renderobj;
    Py_INCREF(renderobj);

    if (!pg_TwoIntsFromObj(sizeobj, &self->width, &self->height)) {
        RAISE(PyExc_TypeError,
              "size should be a sequence of two elements");
        return -1;
    }
    if (self->width <= 0 || self->height <= 0) {
        RAISE(PyExc_ValueError,
              "width and height must be positive");
        return -1;
    }

    //implementing the default of "static"
    if (!static_ && !target && !streaming)
        static_ = 1;

    if (static_) {
        if (streaming || target) {
            goto ACCESS_ERROR;
        }
        access = SDL_TEXTUREACCESS_STATIC;
    }
    else if (streaming) {
        if (static_ || target) {
            goto ACCESS_ERROR;
        }
        access = SDL_TEXTUREACCESS_STREAMING;
    }
    else if (target) {
        if (streaming || static_) {
            goto ACCESS_ERROR;
        }
        access = SDL_TEXTUREACCESS_TARGET;
    }
    else {
        access = SDL_TEXTUREACCESS_STATIC;
    }

    format = format_from_depth(depth);
    if (!format && PyErr_Occurred())
        return -1;

    self->texture = SDL_CreateTexture(
        self->renderer->renderer,
        format, access,
        self->width, self->height);

    if (!self->texture) {
        RAISE(pgExc_SDLError, SDL_GetError());
        return -1;
    }

    return 0;

ACCESS_ERROR:

    RAISE(PyExc_ValueError, "only one of static, streaming, "
                            "or target can be true");
    return -1;
}

static PyObject *
pg_texture_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Uint8 rgba[] = { 255, 255, 255 };
    pgColorObject *col = (pgColorObject*) pgColor_NewLength(rgba, 3);
    pgTextureObject *obj;
    if (!col) {
        return NULL;
    }
    obj = (pgTextureObject*) type->tp_alloc(type, 0);
    obj->color = col;
    obj->alpha = 255;
    return (PyObject*)obj;
}

static void
pg_texture_dealloc(pgTextureObject *self)
{
    if (self->texture) {
        SDL_DestroyTexture(self->texture);
        self->texture = NULL;
    }
    Py_XDECREF(self->renderer);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static pgTextureObject *
pg_texture_from_surface(PyObject *self, PyObject *args, PyObject *kw)
{
    char* keywords[] = {
        "renderer",
        "surface",
        NULL
    };
    PyObject *surfaceobj;
    SDL_Surface *surf;
    pgTextureObject *textureobj = (pgTextureObject*) pg_texture_new(&pgTexture_Type, NULL, NULL);

    if (textureobj == NULL)
        return NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO", keywords,
                                     &textureobj->renderer, &surfaceobj)) {
        pg_texture_dealloc(textureobj);
        return NULL;
    }

    Py_INCREF(textureobj->renderer);
    if (!pgRenderer_Check(textureobj->renderer)) {
        RAISE(PyExc_TypeError, "not a renderer object");
        pg_texture_dealloc(textureobj);
        return NULL;
    }

    if (!pgSurface_Check(surfaceobj)) {
        RAISE(PyExc_TypeError, "not a surface");
        pg_texture_dealloc(textureobj);
        return NULL;
    }
    surf = pgSurface_AsSurface((pgSurfaceObject*) surfaceobj);

    textureobj->width = surf->w;
    textureobj->height = surf->h;
    textureobj->texture = SDL_CreateTextureFromSurface(
        textureobj->renderer->renderer, surf);
    if (!textureobj->texture) {
        RAISE(pgExc_SDLError, SDL_GetError());
        pg_texture_dealloc(textureobj);
        return NULL;
    }

    return textureobj;
}

static PyTypeObject pgTexture_Type = {
    PyVarObject_HEAD_INIT(NULL,0)
    "pygame.Texture", /*name*/
    sizeof(pgTextureObject), /*basicsize*/
    0, /*itemsize*/
    (destructor)pg_texture_dealloc, /*dealloc*/
    NULL, /*print*/
    NULL, /*getattr*/
    NULL, /*setattr*/
    NULL, /*compare/reserved*/
    NULL, /*repr*/
    NULL, /*as_number*/
    NULL, /*as_sequence*/
    NULL, /*as_mapping*/
    NULL, /*hash*/
    NULL, /*call*/
    NULL, /*str*/
    0L,
    0L,
    0L,
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    NULL, /* TODO: docstring */
    NULL, /* tp_traverse */
    NULL, /* tp_clear */
    NULL, /* tp_richcompare */
    NULL, /* tp_weaklistoffset */
    NULL, /* tp_iter */
    NULL, /* tp_iternext */
    pg_texture_methods, /* tp_methods */
    pg_texture_members, /* tp_members */
    pg_texture_getset, /* tp_getset */
    NULL, /* tp_base */
    NULL, /* tp_dict */
    NULL, /* tp_descr_get */
    NULL, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc)pg_texture_init, /* tp_init */
    NULL, /* tp_alloc */
    pg_texture_new /* tp_new */
};

/*
 * WINDOW
 */

static PyObject *
pg_window_maximize(pgWindowObject *self)
{
    SDL_MaximizeWindow(self->_win);
}

static PyObject *
pg_window_minimize(pgWindowObject *self)
{
    SDL_MinimizeWindow(self->_win);
}

static PyObject *
pg_window_set_icon(pgWindowObject *self, PyObject *args)
{
    PyObject *surface;
    if (!PyArg_ParseTuple(args, "O!", &pgSurface_Type, &surface))
        return NULL;

    //if (!pgSurface_Check(surface))
    //    raise TypeError('surface must be a Surface object');

    SDL_SetWindowIcon(self->_win, pgSurface_AsSurface(surface));
}

static PyMethodDef pg_window_methods[] = {
    { "maximize", (PyCFunction)pg_window_maximize, METH_NOARGS, "TODO"},
    { "minimize", (PyCFunction)pg_window_minimize, METH_NOARGS, "TODO"},
    { "set_icon", (PyCFunction)pg_window_set_icon, METH_VARARGS, "TODO"},
    { NULL }
};

static PyObject *
pg_window_get_size(pgWindowObject *self) 
{
    int w, h;
    SDL_GetWindowSize(self->_win, &w, &h);
    return Py_BuildValue("(ii)", w, h);
}

static PyObject *
pg_window_set_size(pgWindowObject *self, PyObject *val, void *closure) 
{
    int w, h;
    if (!pg_TwoIntsFromObj(val, &w, &h)) {
        RAISE(PyExc_TypeError,
            "size should be a sequence of two elements");
        return -1;
    }
    SDL_SetWindowSize(self->_win, w, h);
}

static PyGetSetDef pg_window_getset[] = {
    { "size", (getter)pg_window_get_size, (setter)pg_window_set_size, NULL /*TODO*/, NULL },
    { NULL }
};

/*
    char* keywords[] = {
        "renderer",
        "size",
        "depth",
        "static",
        "streaming",
        "target",
        NULL
    };
    PyObject *sizeobj;
    PyObject *renderobj;
    int depth;
    int static_ = 1;
    int streaming = 0;
    int target = 0;
    int format;
    int access;

#if PY3
    const char *formatstr = "OO|Ippp";
#else
    const char *formatstr = "OO|Iiii";
#endif

    if (!PyArg_ParseTupleAndKeywords(args, kw, formatstr, keywords,
                                     &renderobj,
                                     &sizeobj,
                                     &depth,
                                     &static_, &streaming, &target)) {
        return -1;
*/

static int
pg_window_init(pgWindowObject *self, PyObject *args, PyObject *kw) {


    //def __init__(self, title='pygame',
    //             size=DEFAULT_SIZE,
    //             position=WINDOWPOS_UNDEFINED,
    //             bint fullscreen=False,
    //             bint fullscreen_desktop=False, **kwargs):


    // ignoring extensive keyword arguments for now - and fullscreen flags

    char* title = "pygame";
    int x = SDL_WINDOWPOS_UNDEFINED;
    int y = SDL_WINDOWPOS_UNDEFINED;
    int w = 640;
    int h = 480;

    char* keywords[] = {
        "title",
        "size",
        "position",
        NULL
    };

    PyObject *sizeobj = NULL;
    PyObject *positionobj = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "|sOO", keywords, &title, &sizeobj, &positionobj))
        return -1;

    //UNSUPPORTED: previous flags to SIZE and POSITION

    if (sizeobj && !pg_TwoIntsFromObj(sizeobj, &w, &h)) {
        RAISE(PyExc_TypeError,
            "size should be a sequence of two elements");
        return -1;
    }

    if (positionobj && !pg_TwoIntsFromObj(positionobj, &x, &y)) {
        RAISE(PyExc_TypeError,
            "position should be a sequence of two elements");
        return -1;
    }

    //https://wiki.libsdl.org/SDL_CreateWindow
    //https://wiki.libsdl.org/SDL_WindowFlags
    self->_win = SDL_CreateWindow(title, x, y, w, h, 0);
    //self->_is_borrowed=0
    if (!self->_win)
        return -1;
    SDL_SetWindowData(self->_win, "pg_window", self);

    
    //import pygame.pkgdata
    //surf = pygame.image.load(pygame.pkgdata.getResource(
    //                         'pygame_icon.bmp'))
    //surf.set_colorkey(0)
    //self.set_icon(surf)
    
}

static PyObject *
pg_window_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    pgWindowObject *obj;
    obj = (pgWindowObject*) type->tp_alloc(type, 0);
    return (PyObject*)obj;
}

static void
pg_window_dealloc(pgWindowObject *self)
{
    if (self->_win) {
        SDL_DestroyWindow(self->_win);
        self->_win = NULL;
    }
}


static PyTypeObject pgWindow_Type = {
    PyVarObject_HEAD_INIT(NULL,0)
    "pygame.Window", /*name*/
    sizeof(pgWindowObject), /*basicsize*/
    0, /*itemsize*/
    (destructor)pg_window_dealloc, /*dealloc*/
    NULL, /*print*/
    NULL, /*getattr*/
    NULL, /*setattr*/
    NULL, /*compare/reserved*/
    NULL, /*repr*/
    NULL, /*as_number*/
    NULL, /*as_sequence*/
    NULL, /*as_mapping*/
    NULL, /*hash*/
    NULL, /*call*/
    NULL, /*str*/
    0L,
    0L,
    0L,
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    NULL, /* TODO: docstring */
    NULL, /* tp_traverse */
    NULL, /* tp_clear */
    NULL, /* tp_richcompare */
    NULL, /* tp_weaklistoffset */
    NULL, /* tp_iter */
    NULL, /* tp_iternext */
    pg_window_methods, /* tp_methods */
    NULL, /* tp_members */ /*pg_texture_members*/
    pg_window_getset, /* tp_getset */
    NULL, /* tp_base */
    NULL, /* tp_dict */
    NULL, /* tp_descr_get */
    NULL, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc)pg_window_init, /* tp_init */
    NULL, /* tp_alloc */
    pg_window_new /* tp_new */                  /*texture uses this, does window need to?*/
};


MODINIT_DEFINE(video)
{
    PyObject *module;
    PyObject *dict;
#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "video",
                                         NULL /* TODO */,
                                         -1,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};
#endif
    import_pygame_base();
    import_pygame_color();
    import_pygame_surface();
    import_pygame_rect();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
    if (PyType_Ready(&pgRenderer_Type) < 0) {
        MODINIT_ERROR;
    }
    if (PyType_Ready(&pgTexture_Type) < 0) {
        MODINIT_ERROR;
    }
    if (PyType_Ready(&pgWindow_Type) < 0) {
        MODINIT_ERROR;
    }

#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3(MODPREFIX "video", NULL,
                            NULL /* TODO: docstring */);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }

    drawfnc_str = PyUnicode_FromString("draw");
    if (!drawfnc_str) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    /* TODO: cleanup for drawfnc_str */

    dict = PyModule_GetDict(module);

    if (PyDict_SetItemString(dict, "Renderer", (PyObject *)&pgRenderer_Type)) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    if (PyDict_SetItemString(dict, "Texture", (PyObject *)&pgTexture_Type)) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    if (PyDict_SetItemString(dict, "Window", (PyObject *)&pgWindow_Type)) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    MODINIT_RETURN(module);
}
