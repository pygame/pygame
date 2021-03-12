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