/*
 * RENDERER
 */

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