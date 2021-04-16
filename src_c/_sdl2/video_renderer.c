/*
 * RENDERER
 */

static PyObject *
pg_renderer_from_window(pgRendererObject *cls, PyObject* args, PyObject *kw) {
    // TODO: implement this function properly
    // From reading the code in video.pyx, I don't understand how is supposed to operate
    // Update: I think it's supposed to be the complement to Window.from_display_module() for renderers

    char* keywords[] = {
        "window",
        NULL
    };

    PyObject* window;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O", keywords,
                                     &window)) {
        return NULL;
    }

    if(!pgWindow_Check(window)) {
        RAISE(PyExc_TypeError, "window must be a Window object");
        return NULL;
    }

    // TODO: manage window object lifecycle? Increase refcount, decrease on dealloc?

    pgRendererObject* renderer = pg_renderer_new((PyTypeObject*)cls, NULL, NULL);
    renderer->window = (pgWindowObject *) window;
    renderer->renderer = SDL_GetRenderer(renderer->window->_win);
    if (!renderer->renderer) {
        RAISE(pgExc_SDLError, "I'm confused about what this function is supposed to do");
        return NULL;
    }

    return (PyObject*)renderer;
}

static PyObject *
pg_renderer_clear(pgRendererObject *self)
{
    if (SDL_RenderClear(self->renderer) < 0) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    Py_RETURN_NONE;
}

static PyObject *
pg_renderer_present(pgRendererObject *self)
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

static PyObject *
pg_renderer_draw_line(pgRendererObject *self, PyObject *args, PyObject *kw) {
    //https://wiki.libsdl.org/SDL_RenderDrawLine
    static char *keywords[] = {
        "p1",
        "p2",
        NULL
    };

    PyObject *p1 = NULL, *p2 = NULL;
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;

    if(!PyArg_ParseTupleAndKeywords(args, kw, "OO", keywords, &p1, &p2)) {
        return NULL;
    }
    
    if (!pg_TwoIntsFromObj(p1, &x1, &y1)) {
        return RAISE(PyExc_TypeError, "Point 1 must be a sequence of two numbers.");
    }

    if (!pg_TwoIntsFromObj(p2, &x2, &y2)) {
        return RAISE(PyExc_TypeError, "Point 2 must be a sequence of two numbers.");     
    }

    if (SDL_RenderDrawLine(self->renderer, x1, y1, x2, y2) < 0) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    Py_RETURN_NONE;  
}

static PyObject *
pg_renderer_draw_point(pgRendererObject *self, PyObject *args, PyObject *kw) {
    //https://wiki.libsdl.org/SDL_RenderDrawPoint
    static char *keywords[] = {
        "point",
        NULL
    };

    PyObject *point = NULL;
    int x = 0, y = 0;

    if(!PyArg_ParseTupleAndKeywords(args, kw, "O", keywords, &point)) {
        return NULL;
    }
    
    if (!pg_TwoIntsFromObj(point, &x, &y)) {
        return RAISE(PyExc_TypeError, "Point must be a sequence of two numbers.");     
    }

    if (SDL_RenderDrawPoint(self->renderer, x, y) < 0) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    Py_RETURN_NONE;   
}

static PyObject *
pg_renderer_draw_rect(pgRendererObject *self, PyObject *args, PyObject *kw) {
    //https://wiki.libsdl.org/SDL_RenderDrawRect
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
    if (rectptr == NULL) {
        return RAISE(PyExc_TypeError, "expected a rectangle");
    }

    if (SDL_RenderDrawRect(self->renderer, rectptr) < 0) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    Py_RETURN_NONE;    
}

static PyObject *
pg_renderer_fill_rect(pgRendererObject *self, PyObject *args, PyObject *kw) {
    //https://wiki.libsdl.org/SDL_RenderFillRect
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
    if (rectptr == NULL) {
        return RAISE(PyExc_TypeError, "expected a rectangle");
    }

    if (SDL_RenderFillRect(self->renderer, rectptr) < 0) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    Py_RETURN_NONE;
}

static PyObject *
pg_renderer_to_surface(pgRendererObject *self, PyObject *args, PyObject *kw) {
    //TODO: implement
    Py_RETURN_NONE;
}

static PyMethodDef pg_renderer_methods[] = {
    { "from_window", (PyCFunction)pg_renderer_from_window, METH_VARARGS | METH_KEYWORDS | METH_CLASS, DOC_RENDERERFROMWINDOW},
    { "clear", (PyCFunction)pg_renderer_clear, METH_NOARGS, DOC_RENDERERCLEAR},
    { "present", (PyCFunction)pg_renderer_present, METH_NOARGS, DOC_RENDERERPRESENT},
    { "get_viewport", (PyCFunction)pg_renderer_get_viewport, METH_NOARGS, DOC_RENDERERGETVIEWPORT},
    { "set_viewport", (PyCFunction)pg_renderer_set_viewport, METH_O, DOC_RENDERERSETVIEWPORT},
    { "blit", (PyCFunction)pg_renderer_blit, METH_VARARGS | METH_KEYWORDS, DOC_RENDERERBLIT},
    { "draw_line", (PyCFunction)pg_renderer_draw_line, METH_VARARGS | METH_KEYWORDS, DOC_RENDERERDRAWLINE},
    { "draw_point", (PyCFunction)pg_renderer_draw_point, METH_VARARGS | METH_KEYWORDS, DOC_RENDERERDRAWPOINT},
    { "draw_rect", (PyCFunction)pg_renderer_draw_rect, METH_VARARGS | METH_KEYWORDS, DOC_RENDERERDRAWRECT},
    { "fill_rect", (PyCFunction)pg_renderer_fill_rect, METH_VARARGS | METH_KEYWORDS, DOC_RENDERERFILLRECT},
    { "to_surface", (PyCFunction)pg_renderer_to_surface, METH_VARARGS | METH_KEYWORDS, DOC_RENDERERTOSURFACE},
    { NULL }
};

static PyObject *
pg_renderer_get_logical_size(pgRendererObject *self, void *closure)
{
    //https://wiki.libsdl.org/SDL_RenderGetLogicalSize
    int w, h;
    SDL_RenderGetLogicalSize(self->renderer, &w, &h);
    return Py_BuildValue("(ii)", w, h);
}

static int
pg_renderer_set_logical_size(pgRendererObject *self, PyObject *val, void *closure)
{
    //https://wiki.libsdl.org/SDL_RenderSetLogicalSize
    int w, h;
    if (!pg_TwoIntsFromObj(val, &w, &h)) {
        RAISE(PyExc_TypeError,
            "logical_size should be a sequence of two ints.");
        return -1;
    }

    if(SDL_RenderSetLogicalSize(self->renderer, w, h) < 0) {
        RAISE(pgExc_SDLError, SDL_GetError());
        return -1;
    }
    return 0;
}

static PyObject *
pg_renderer_get_scale(pgRendererObject *self, void *closure)
{
    //https://wiki.libsdl.org/SDL_RenderGetScale
    float x, y;
    SDL_RenderGetScale(self->renderer, &x, &y);
    return Py_BuildValue("(ff)", x, y);
}

static int
pg_renderer_set_scale(pgRendererObject *self, PyObject *val, void *closure)
{
    //https://wiki.libsdl.org/SDL_RenderSetScale
    float x, y;
    if (!pg_TwoFloatsFromObj(val, &x, &y)) {
        RAISE(PyExc_TypeError,
            "scale should be a sequence of two floats.");
        return -1;
    }

    if(SDL_RenderSetScale(self->renderer, x, y) < 0) {
        RAISE(pgExc_SDLError, SDL_GetError());
        return -1;
    }
    return 0;
}

static PyObject *
pg_renderer_get_color(pgRendererObject *self, void *closure)
{
    // Error checking *shouldn't* be necessary because self->color should always be legit
    Uint8 *colarray = pgColor_AsArray(self->drawcolor);
    return pgColor_New(colarray);
}

static int
pg_renderer_set_color(pgRendererObject *self, PyObject *val, void *closure)
{
    Uint8 rgba[4] = {0, 0, 0, 0};
    if (!pg_RGBAFromFuzzyColorObj(val, rgba)) {
        RAISE(PyExc_TypeError, "expected a color (sequence of color object)");
        return -1;
    }

    if (SDL_SetRenderDrawColor(self->renderer,
                               rgba[0], rgba[1], rgba[2], rgba[3]) < 0) {
        RAISE(pgExc_SDLError, SDL_GetError());
        return -1;
    }

    Py_DECREF(self->drawcolor);
    self->drawcolor = (pgColorObject*)pgColor_New(rgba);

    return 0;
}

static PyObject *
pg_renderer_get_target(pgRendererObject *self, void *closure)
{
    //Uses stored value rather than SDL_GetRenderTarget, since we can keep PyObject Target objects this way.
    if(self->target) {
        return Py_BuildValue("O", self->target);
    }
    Py_RETURN_NONE;
}

static PyObject *
pg_renderer_set_target(pgRendererObject *self, PyObject *val, void *closure)
{
    //https://wiki.libsdl.org/SDL_SetRenderTarget
    pgTextureObject* newtarget = NULL;

    if(pgTexture_Check(val)) {
        newtarget = (pgTextureObject*)val;
    }
    else if (val == Py_None) {}
    else {
        return RAISE(PyExc_TypeError, "Target must be a Texture or None.");
    }
        
    self->target = newtarget;
    Py_XINCREF(self->target);

    if(SDL_SetRenderTarget(self->renderer, self->target? self->target->texture: NULL) < 0) {
        return RAISE(pgExc_SDLError, SDL_GetError());          
    }
    return 0;
}

static PyGetSetDef pg_renderer_getset[] = {
    { "draw_color", (getter)pg_renderer_get_color, (setter)pg_renderer_set_color, NULL, NULL },
    { "logical_size", (getter)pg_renderer_get_logical_size, (setter)pg_renderer_set_logical_size, NULL, NULL },
    { "scale", (getter)pg_renderer_get_scale, (setter)pg_renderer_set_scale, NULL, NULL },
    { "target", (getter)pg_renderer_get_target, (setter)pg_renderer_set_target, NULL, NULL },
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

    self->_is_borrowed = 0;

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
    if (!self->_is_borrowed) {
        if (self->renderer) {
            SDL_DestroyRenderer(self->renderer);
            self->renderer = NULL;
        }
    }
    Py_XDECREF(self->window);
    Py_XDECREF(self->target);
    Py_DECREF(self->drawcolor);
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