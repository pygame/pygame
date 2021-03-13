/*
 * WINDOW
 */

static PyObject *
pg_window_from_display_module(pgWindowObject *cls) {
    SDL_Window* window = pg_GetDefaultWindow();
    if (!window) {
        return RAISE(pgExc_SDLError, "The display module has no window to be found! Are you sure you've called set_mode()?");
    }

    pgWindowObject* window_obj = pg_window_new(cls, NULL, NULL);
    window_obj->_win = window;
    window_obj->_is_borrowed = 1;
    return window_obj;
}

static PyObject *
pg_window_set_windowed(pgWindowObject *self)
{
    //Enable windowed mode. (Exit fullscreen)
    //https://wiki.libsdl.org/SDL_SetWindowFullscreen
    if (SDL_SetWindowFullscreen(self->_win, 0) < 0)
        return RAISE(pgExc_SDLError, SDL_GetError());
    Py_RETURN_NONE;
}

static PyObject *
pg_window_set_fullscreen(pgWindowObject *self, PyObject* args, PyObject* kw)
{
    //Enable fullscreen for the window.
    //param bool desktop: If ``True``: use the current desktop resolution. If ``False``: change the fullscreen resolution to the window size.
    //https://wiki.libsdl.org/SDL_SetWindowFullscreen

    char* keywords[] = {
        "desktop",
        NULL
    };

    int desktop = 0;
    int flags = 0;

#if PY3
    const char *formatstr = "|p";
#else
    const char *formatstr = "|i";
#endif

    if (!PyArg_ParseTupleAndKeywords(args, kw, formatstr, keywords, &desktop))
        return NULL;

    if (desktop)
        flags = SDL_WINDOW_FULLSCREEN_DESKTOP;
    else
        flags = SDL_WINDOW_FULLSCREEN;

    if (SDL_SetWindowFullscreen(self->_win, flags) < 0) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    Py_RETURN_NONE;
}

static PyObject *
pg_window_destroy(pgWindowObject *self)
{
    //https://wiki.libsdl.org/SDL_DestroyWindow
    if (self->_win) {
        SDL_DestroyWindow(self->_win);
        self->_win = NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
pg_window_hide(pgWindowObject *self)
{
    //https://wiki.libsdl.org/SDL_HideWindow
    SDL_HideWindow(self->_win);
    Py_RETURN_NONE;
}

static PyObject *
pg_window_show(pgWindowObject *self)
{
    //https://wiki.libsdl.org/SDL_ShowWindow
    SDL_ShowWindow(self->_win);
    Py_RETURN_NONE;
}

static PyObject *
pg_window_focus(pgWindowObject *self, PyObject *args, PyObject *kw)
{
    //https://wiki.libsdl.org/SDL_RaiseWindow
    //https://wiki.libsdl.org/SDL_SetWindowInputFocus (X11 only)

    char* keywords[] = {
        "input_only",
        NULL
    };

    int input_only = 0;

#if PY3
    const char *formatstr = "|p";
#else
    const char *formatstr = "|i";
#endif

    if (!PyArg_ParseTupleAndKeywords(args, kw, formatstr, keywords, &input_only))
        return NULL;

    if (input_only) {
        if (SDL_SetWindowInputFocus(self->_win) < 0) {
            RAISE(pgExc_SDLError, SDL_GetError());
            return -1;
        }
    }
    else {
        SDL_RaiseWindow(self->_win);
    }
    Py_RETURN_NONE;
}

static PyObject *
pg_window_restore(pgWindowObject *self)
{
    SDL_RestoreWindow(self->_win);
    Py_RETURN_NONE;
}

static PyObject *
pg_window_maximize(pgWindowObject *self)
{
    SDL_MaximizeWindow(self->_win);
    Py_RETURN_NONE;
}

static PyObject *
pg_window_minimize(pgWindowObject *self)
{
    SDL_MinimizeWindow(self->_win);
    Py_RETURN_NONE;
}

// TODO: make this function actually work
static PyObject *
pg_window_set_icon(pgWindowObject *self, PyObject *args, PyObject *kw)
{
    //https://wiki.libsdl.org/SDL_SetWindowIcon

    char* keywords[] = {
        "surface",
        NULL
    };

    PyObject *surfaceobj;
    SDL_Surface *surf;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O", keywords, &surfaceobj))
        return NULL;

    if (!pgSurface_Check(surfaceobj)) {
        RAISE(PyExc_TypeError, "surface must be a Surface object");
        return NULL;
    }
    surf = pgSurface_AsSurface(surfaceobj);

    // For some reason I can't compare output of this function for error checking
    SDL_SetWindowIcon(self->_win, surf);
    Py_RETURN_NONE;
}

static PyObject *
pg_window_set_modal_for(pgWindowObject *self, PyObject *args, PyObject *kw)
{
    char* keywords[] = {
        "window",
        NULL
    };

    pgWindowObject* win;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O!", keywords, &pgWindow_Type, &win))
        return NULL;

    //https://wiki.libsdl.org/SDL_SetWindowModalFor
    if (SDL_SetWindowModalFor(self->_win, win->_win) < 0) {
        RAISE(pgExc_SDLError, SDL_GetError());
        return -1;
    }

    // TODO: this should exit gracefully if not supported, why does it segfault?
    Py_RETURN_NONE;
}

static PyMethodDef pg_window_methods[] = {
    { "from_display_module", (PyCFunction)pg_window_from_display_module, METH_CLASS | METH_NOARGS, "TODO"},
    { "set_windowed", (PyCFunction)pg_window_set_windowed, METH_NOARGS, "TODO"},
    { "set_fullscreen", (PyCFunction)pg_window_set_fullscreen, METH_VARARGS | METH_KEYWORDS, "TODO"},
    { "destroy", (PyCFunction)pg_window_destroy, METH_NOARGS, "TODO"},
    { "hide", (PyCFunction)pg_window_hide, METH_NOARGS, "TODO"},
    { "show", (PyCFunction)pg_window_show, METH_NOARGS, "TODO"},
    { "focus", (PyCFunction)pg_window_focus, METH_VARARGS | METH_KEYWORDS, "TODO"},
    { "restore", (PyCFunction)pg_window_restore, METH_NOARGS, "TODO"},
    { "maximize", (PyCFunction)pg_window_maximize, METH_NOARGS, "TODO"},
    { "minimize", (PyCFunction)pg_window_minimize, METH_NOARGS, "TODO"},
    { "set_icon", (PyCFunction)pg_window_set_icon, METH_VARARGS | METH_KEYWORDS, "TODO"},
    { "set_modal_for", (PyCFunction)pg_window_set_modal_for, METH_VARARGS | METH_KEYWORDS, "TODO"},
    { NULL }
};

static PyObject *
pg_window_get_grab(pgWindowObject *self) 
{
    // Window's input grab state (``True`` or ``False``).
    // https://wiki.libsdl.org/SDL_GetWindowGrab
    int grab = SDL_GetWindowGrab(self->_win);
    if (grab) // TODO: does this work on python 2?
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
pg_window_set_grab(pgWindowObject *self, PyObject *val, void *closure) 
{
    // https://wiki.libsdl.org/SDL_SetWindowGrab
    int grab = PyObject_IsTrue(val); // TODO: is this crazy? allows stuff like `window.grab = "wow"`
    SDL_SetWindowGrab(self->_win, grab);
    return 0;
}

static PyObject *
pg_window_get_relative_mouse(pgWindowObject *self) 
{
    // Window's relative mouse motion state (``True`` or ``False``).
    // https://wiki.libsdl.org/SDL_GetRelativeMouseMode
    int relative = SDL_GetRelativeMouseMode();
    if (relative) // TODO: does this work on python 2?
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
pg_window_set_relative_mouse(pgWindowObject *self, PyObject *val, void *closure) 
{
    // https://wiki.libsdl.org/SDL_SetRelativeMouseMode
    int relative = PyObject_IsTrue(val); // TODO: is this crazy? allows stuff like `window.relative_mouse = "wow"`
    SDL_SetRelativeMouseMode(relative);
    return 0;
}

static PyObject *
pg_window_get_title(pgWindowObject *self) 
{
    // The title of the window or u"" if there is none.
    // https://wiki.libsdl.org/SDL_GetWindowTitle
    char* title = SDL_GetWindowTitle(self->_win);
    return Py_BuildValue("s", title);
}

static PyObject *
pg_window_set_title(pgWindowObject *self, PyObject *val, void *closure) 
{
    // Set the window title.
    // https://wiki.libsdl.org/SDL_SetWindowTitle

    // This might be a dumb way of getting a char* from a single PyObject, but it seems to work
    PyObject* args = Py_BuildValue("(O)", val);
    char* title = NULL;
    if(!PyArg_ParseTuple(args, "es", "UTF-8", &title)) {
        return -1;
    }
    SDL_SetWindowTitle(self->_win, title);
    Py_DECREF(args);
    PyMem_Free(title);
    return 0;
}

static PyObject *
pg_window_get_resizable(pgWindowObject *self) 
{
    // Sets whether the window is resizable.
    int resizable = SDL_GetWindowFlags(self->_win) & SDL_WINDOW_RESIZABLE;
    if (resizable) // TODO: does this work on python 2?
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
pg_window_set_resizable(pgWindowObject *self, PyObject *val, void *closure) 
{
    // https://wiki.libsdl.org/SDL_SetWindowResizable
    int resizable = PyObject_IsTrue(val); // TODO: is this crazy? allows stuff like `window.borderless = "wow"`
    SDL_SetWindowResizable(self->_win, resizable);
    return 0;
}

static PyObject *
pg_window_get_borderless(pgWindowObject *self) 
{
    // Add or remove the border from the actual window.
    // .. note:: You can't change the border state of a fullscreen window.

    int borderless = SDL_GetWindowFlags(self->_win) & SDL_WINDOW_BORDERLESS;
    if (borderless) // TODO: does this work on python 2?
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
pg_window_set_borderless(pgWindowObject *self, PyObject *val, void *closure) 
{
    // https://wiki.libsdl.org/SDL_SetWindowBordered
    int borderless = PyObject_IsTrue(val); // TODO: is this crazy? allows stuff like `window.borderless = "wow"`
    SDL_SetWindowBordered(self->_win, 1-borderless);
    return 0;
}

static PyObject *
pg_window_get_id(pgWindowObject *self) 
{
    int id = SDL_GetWindowID(self->_win);
    return Py_BuildValue("i", id);
}

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
    return 0;
}

// TODO: deal with this whole position system (exposing WINDOWPOS_CENTERED, UNDEFINED)
// You could set position with one number and get position as two...

static PyObject *
pg_window_get_position(pgWindowObject *self) 
{
    // Window's screen coordinates, or WINDOWPOS_CENTERED or WINDOWPOS_UNDEFINED
    int x, y;
    SDL_GetWindowPosition(self->_win, &x, &y);
    return Py_BuildValue("(ii)", x, y);
}

static PyObject *
pg_window_set_position(pgWindowObject *self, PyObject *val, void *closure) 
{
    int x, y, parsed = 0;
    if (pg_IntFromObj(val, &x)) {
        if (x == SDL_WINDOWPOS_UNDEFINED) {
            y = SDL_WINDOWPOS_UNDEFINED;
            parsed = 1;
        }
        if (x == SDL_WINDOWPOS_CENTERED) {
            y = SDL_WINDOWPOS_CENTERED;
            parsed = 1;
        }
    }
    if (!parsed && pg_TwoIntsFromObj(val, &x, &y)) {
        parsed = 1;
    }
    if (!parsed) {
        RAISE(PyExc_TypeError, "position should be (x,y) or POS_UNDEFINED or POS_CENTERED");
        return -1;       
    }
    SDL_SetWindowPosition(self->_win, x, y);
    return 0;
}

static PyObject *
pg_window_get_opacity(pgWindowObject *self) 
{
    // Window opacity. It ranges between 0.0 (fully transparent)
    // and 1.0 (fully opaque).
    // https://wiki.libsdl.org/SDL_GetWindowOpacity
    float opacity;
    if (SDL_GetWindowOpacity(self->_win, &opacity) < 0) {
        RAISE(pgExc_SDLError, SDL_GetError());
        return -1;
    }
    return Py_BuildValue("f", opacity);
}

static PyObject *
pg_window_set_opacity(pgWindowObject *self, PyObject *val, void *closure) 
{
    float opacity;
    if(!pg_FloatFromObj(val, &opacity)) {
        RAISE(PyExc_TypeError, "opacity should be a float between 0 and 1.");
        return -1;         
    }
    if (SDL_SetWindowOpacity(self->_win, opacity) < 0) {
        RAISE(pgExc_SDLError, SDL_GetError());
        return -1;        
    }
    return 0;
}

static PyObject *
pg_window_get_brightness(pgWindowObject *self) 
{
    // The brightness (gamma multiplier) for the display that owns a given window.
    // 0.0 is completely dark and 1.0 is normal brightness.
    // https://wiki.libsdl.org/SDL_GetWindowBrightness
    float brightness = SDL_GetWindowBrightness(self->_win);
    return Py_BuildValue("f", brightness);
}

static PyObject *
pg_window_set_brightness(pgWindowObject *self, PyObject *val, void *closure) 
{
    // https://wiki.libsdl.org/SDL_SetWindowBrightness
    float brightness;
    if(!pg_FloatFromObj(val, &brightness)) {
        RAISE(PyExc_TypeError, "brightness should be a float.");
        return -1;         
    }
    if (SDL_SetWindowBrightness(self->_win, brightness) < 0) {
        RAISE(pgExc_SDLError, SDL_GetError());
        return -1;        
    }
    return 0;
}

static PyObject *
pg_window_get_display_index(pgWindowObject *self)
{
    // https://wiki.libsdl.org/SDL_GetWindowDisplayIndex
    int index = SDL_GetWindowDisplayIndex(self->_win);
    if (index < 0) {
        RAISE(pgExc_SDLError, SDL_GetError());
        return -1;           
    }
    return Py_BuildValue("i", index);
}

static PyGetSetDef pg_window_getset[] = {
    { "grab", (getter)pg_window_get_grab, (setter)pg_window_set_grab, NULL /*TODO*/, NULL },
    { "relative_mouse", (getter)pg_window_get_relative_mouse, (setter)pg_window_set_relative_mouse, NULL /*TODO*/, NULL },
    { "title", (getter)pg_window_get_title, (setter)pg_window_set_title, NULL /*TODO*/, NULL },
    { "resizable", (getter)pg_window_get_resizable, (setter)pg_window_set_resizable, NULL /*TODO*/, NULL },
    { "borderless", (getter)pg_window_get_borderless, (setter)pg_window_set_borderless, NULL /*TODO*/, NULL },
    { "id", (getter)pg_window_get_id, NULL, NULL /*TODO*/, NULL },
    { "size", (getter)pg_window_get_size, (setter)pg_window_set_size, NULL /*TODO*/, NULL },
    { "position", (getter)pg_window_get_position, (setter)pg_window_set_position, NULL /*TODO*/, NULL },
    { "opacity", (getter)pg_window_get_opacity, (setter)pg_window_set_opacity, NULL /*TODO*/, NULL },
    { "brightness", (getter)pg_window_get_brightness, (setter)pg_window_set_brightness, NULL /*TODO*/, NULL },
    { "display_index", (getter)pg_window_get_display_index, NULL, NULL /*TODO*/, NULL },
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
    NULL, /* tp_members */
    pg_window_getset, /* tp_getset */
    NULL, /* tp_base */
    NULL, /* tp_dict */
    NULL, /* tp_descr_get */
    NULL, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc)pg_window_init, /* tp_init */
    NULL, /* tp_alloc */
    pg_window_new /* tp_new */
};