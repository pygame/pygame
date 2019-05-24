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


typedef struct {
    PyObject_HEAD
    SDL_Texture *_tex;
    PyObject *renderer;
    SDL_Renderer *_renderer; /* FIXME */
    int width;
    int height;
    pgColorObject *_color;
} pgTextureObject;


static PyTypeObject pgTexture_Type;

static PyObject *
pg_texture_from_surface(pgTextureObject *self, PyObject *args, PyObject *kw);


static PyMemberDef pg_texture_members[] = {
    { "renderer", T_OBJECT_EX, offsetof(pgTextureObject, renderer), READONLY, NULL /* TODO */ },
    { "width", T_INT, offsetof(pgTextureObject, width), READONLY, NULL /* TODO */ },
    { "height", T_INT, offsetof(pgTextureObject, height), READONLY, NULL /* TODO */ },
    { NULL }
};

static PyObject *
pg_texture_get_rect(pgTextureObject *self, PyObject *args, PyObject *kw)
{
    /* TODO: kwargs */
    return pgRect_New4(0, 0, self->width, self->height);
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

    if (SDL_RenderCopyEx(self->_renderer, self->_tex,
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
    /* TODO */
}

static int
pg_texture_set_color(pgTextureObject *self, PyObject *val, void *closure)
{
    /* TODO */
}

static PyObject *
pg_texture_get_alpha(pgTextureObject *self, void *closure)
{
    /* TODO */
}

static int
pg_texture_set_alpha(pgTextureObject *self, PyObject *val, void *closure)
{
    /* TODO */
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
        /* TODO: raise value error */
        return 0;
    }
    return SDL_MasksToPixelFormatEnum(depth, Rmask, Gmask, Bmask, Amask);
}

static int
pg_texture_init(pgTextureObject *self, PyObject *args, PyObject *kw)
{
    /* TODO: use O! for args */
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
    int depth;
    int static_ = 1;
    int streaming = 0;
    int target = 0;
    int format;
    int access;
    PyObject* renderercapsule;

#if PY3
    const char *formatstr = "OO|Ippp";
#else
    const char *formatstr = "OO|Iiii";
#endif

    if (!PyArg_ParseTupleAndKeywords(args, kw, formatstr, keywords,
                                     &self->renderer,
                                     &sizeobj,
                                     &depth,
                                     &static_, &streaming, &target)) {
        return -1;
    }

    /* TODO: check renderer */
    Py_INCREF(self->renderer);

    if (!pg_TwoIntsFromObj(sizeobj, &self->width, &self->height)) {
        RAISE(PyExc_TypeError,
              "size should be a sequence of two elements");
        Py_DECREF(self->renderer);
        return -1;
    }
    if (self->width < 0 || self->height < 0) {
        RAISE(PyExc_ValueError,
              "width and height must be positive");
        Py_DECREF(self->renderer);
        return -1;
    }

    /* TODO: check type args */

    /* TODO: get rid of capsule */
    renderercapsule = PyObject_GetAttrString(self->renderer, "get_remove_renderer");
    if (!renderercapsule) {
        Py_DECREF(self->renderer);
        return -1;
    }
    self->_renderer = (SDL_Renderer*) PyCapsule_GetPointer(renderercapsule, NULL);
    Py_DECREF(renderercapsule);

    format = format_from_depth(depth);
    access = SDL_TEXTUREACCESS_STATIC;
    self->_tex = SDL_CreateTexture(
        self->_renderer,
        format, access,
        self->width, self->height);

    /* TODO: check error here */

    return 0;
}

static PyObject *
pg_texture_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Uint8 rgba[] = { 255, 255, 255 };
    pgColorObject *col = (pgColorObject*) pgColor_NewLength(rgba, 3);
    if (!col) {
        return NULL;
    }
    pgTextureObject *obj = (pgTextureObject*) type->tp_alloc(type, 0);
    obj->_color = col;
    return (PyObject*)obj;
}

static void
pg_texture_dealloc(pgTextureObject *self)
{
    /* TODO: destroy texture */
    Py_XDECREF(self->renderer);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
pg_texture_from_surface(PyObject *self, PyObject *args, PyObject *kw)
{
    /* TODO: use O! for args */
    /* FIXME: check types */
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

    /* TODO: get rid of capsule */
    renderercapsule = PyObject_CallMethod(textureobj->renderer, "get_remove_renderer", NULL);
    if (!renderercapsule) {
        Py_DECREF(textureobj->renderer);
        pg_texture_dealloc(textureobj);
        return NULL;
    }
    textureobj->_renderer = (SDL_Renderer*) PyCapsule_GetPointer(renderercapsule, NULL);
    Py_DECREF(renderercapsule);

    surf = pgSurface_AsSurface((pgSurfaceObject*) surfaceobj);

    textureobj->width = surf->w;
    textureobj->height = surf->h;
    textureobj->_tex = SDL_CreateTextureFromSurface(textureobj->_renderer, surf);
    /* TODO: error handling */

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

    Py_INCREF(&pgTexture_Type);
    if (PyDict_SetItemString(dict, "Texture", (PyObject *)&pgTexture_Type)) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    MODINIT_RETURN(module);
}
