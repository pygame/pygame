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

#include "../pygame.h"
#include "../pgcompat.h"
#include <structmember.h>


typedef struct pgRendererObject pgRendererObject;

typedef struct {
    PyObject_HEAD
    SDL_Texture *texture;
    pgRendererObject *renderer;
    int width;
    int height;
    pgColorObject *color;
    Uint8 alpha;
} pgTextureObject;

/* FIXME: hack */
typedef struct {
    PyObject_HEAD
    SDL_Window *_win;
} pgWindowObject;


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

static PyTypeObject pgRenderer_Type;

#define pgRenderer_Check(x) (((PyObject*)(x))->ob_type == &pgRenderer_Type)


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
    PyObject *drawstr;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|OOi", keywords,
                                     &source, &dest, &area, &flags))
    {
        return NULL;
    }

    if (!PyObject_HasAttrString(source, "draw")) {
        return RAISE(PyExc_TypeError, "source must be drawable");
    }

    drawstr = PyUnicode_FromString("draw");
    if (!drawstr)
        return NULL;
    if (!PyObject_CallMethodObjArgs(source, drawstr, area, dest, NULL)) {
        Py_DECREF(drawstr);
        return NULL;
    }
    Py_DECREF(drawstr);

    /* TODO: */
    /*if (dest == Py_None) {
        return self.get_viewport();
    }*/
    Py_RETURN_NONE;
}

static PyMethodDef pg_renderer_methods[] = {
    { "clear", (PyCFunction)pg_renderer_clear, METH_NOARGS, NULL /* TODO */ },
    { "present", (PyCFunction)pg_renderer_present, METH_NOARGS, NULL /* TODO */ },
    { "blit", (PyCFunction)pg_renderer_blit, METH_VARARGS | METH_KEYWORDS, NULL /* TODO */ },
    { NULL }
};

static PyObject *
pg_renderer_get_color(pgRendererObject *self, void *closure)
{
    Py_INCREF(self->drawcolor);
    return self->drawcolor;
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

static PyGetSetDef pg_renderer_getset[] = {
    { "draw_color", (getter)pg_renderer_get_color, (setter)pg_renderer_set_color, NULL /*TODO*/, NULL },
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
    TYPE_HEAD(NULL, 0) "pygame._sdl2.Renderer", /*name*/
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


static PyTypeObject pgTexture_Type;

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
               float angle, const int * origin[2],
               int flipX, int flipY)
{
    SDL_RendererFlip flip = SDL_FLIP_NONE;
    SDL_Point pointorigin;

    if (origin) {
        pointorigin.x = (*origin)[0];
        pointorigin.y = (*origin)[1];
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

static PyObject *
pg_texture_draw(pgTextureObject *self, PyObject *args, PyObject *kw)
{
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

    if (pgTexture_Draw(self, srcptr, dstptr, angle, originobj ? &origin : NULL, flipX, flipY)) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyMethodDef pg_texture_methods[] = {
    { "from_surface", (PyCFunction)pg_texture_from_surface, METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL /* TODO */ },
    { "get_rect", (PyCFunction)pg_texture_get_rect, METH_VARARGS | METH_KEYWORDS, NULL /* TODO */ },
    { "draw", (PyCFunction)pg_texture_draw, METH_VARARGS | METH_KEYWORDS, NULL /* TODO */ },
    { NULL }
};

static PyObject *
pg_texture_get_color(pgTextureObject *self, void *closure)
{
    Py_INCREF(self->color);
    return self->color;
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

static PyGetSetDef pg_texture_getset[] = {
    { "color", (getter)pg_texture_get_color, (setter)pg_texture_set_color, NULL /*TODO*/, NULL },
    { "alpha", (getter)pg_texture_get_alpha, (setter)pg_texture_set_alpha, NULL /*TODO*/, NULL },
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
    case 0:
    case 32:
        Rmask = 0xF << 16;
        Gmask = 0xF << 8;
        Bmask = 0xF;
        Amask = 0xF << 24;
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
    /* TODO: use O! for args */
    char* keywords[] = {
        "renderer",
        "surface",
        NULL
    };
    PyObject *surfaceobj;
    SDL_Surface *surf;
    PyObject *renderercapsule;
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
    TYPE_HEAD(NULL, 0) "pygame._sdl2.Texture", /*name*/
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


MODINIT_DEFINE(video_new)
{
    PyObject *module;
    PyObject *dict;
#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "video_new",
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

#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3(MODPREFIX "video_new", NULL,
                            NULL /* TODO: docstring */);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }

    dict = PyModule_GetDict(module);

    Py_INCREF(&pgRenderer_Type);
    if (PyDict_SetItemString(dict, "Renderer", (PyObject *)&pgRenderer_Type)) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    Py_INCREF(&pgTexture_Type);
    if (PyDict_SetItemString(dict, "Texture", (PyObject *)&pgTexture_Type)) {
        Py_DECREF(&pgRenderer_Type);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    MODINIT_RETURN(module);
}
