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

static PyObject *
pg_window_get_id(pgWindowObject *self) 
{
    int id = SDL_GetWindowID(self->_win);
    return Py_BuildValue("i", id);
}

static PyGetSetDef pg_window_getset[] = {
    { "size", (getter)pg_window_get_size, (setter)pg_window_set_size, NULL /*TODO*/, NULL },
    { "id", (getter)pg_window_get_id, NULL, NULL /*TODO*/, NULL },
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