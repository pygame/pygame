/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners

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

/*
 *  pygame mouse module
 */
#include "pygame.h"

#include "pgcompat.h"

#include "doc/mouse_doc.h"

/* mouse module functions */
static PyObject *
mouse_set_pos(PyObject *self, PyObject *args)
{
    int x, y;

    if (!pg_TwoIntsFromObj(args, &x, &y))
        return RAISE(PyExc_TypeError, "invalid position argument for set_pos");

    VIDEO_INIT_CHECK();

#if IS_SDLv1
    SDL_WarpMouse((Uint16)x, (Uint16)y);

#else  /* IS_SDLv2 */
    {
        SDL_Window *sdlWindow = pg_GetDefaultWindow();
        SDL_Renderer *sdlRenderer = SDL_GetRenderer(sdlWindow);
        if (sdlRenderer!=NULL){
            SDL_Rect vprect;
            float scalex, scaley;

            SDL_RenderGetScale(sdlRenderer, &scalex, &scaley);
            SDL_RenderGetViewport(sdlRenderer, &vprect);

            x += vprect.x;
            y += vprect.y;

            x = (int)(x * scalex);
            y = (int)(y * scaley);
        }
    }

    SDL_WarpMouseInWindow(NULL, (Uint16)x, (Uint16)y);
#endif /* IS_SDLv2 */
    Py_RETURN_NONE;
}

static PyObject *
mouse_get_pos(PyObject *self)
{
    int x, y;

    VIDEO_INIT_CHECK();
    SDL_GetMouseState(&x, &y);

#if IS_SDLv2
    {
        SDL_Window *sdlWindow = pg_GetDefaultWindow();
        SDL_Renderer *sdlRenderer = SDL_GetRenderer(sdlWindow);
        if (sdlRenderer!=NULL){
            SDL_Rect vprect;
            float scalex, scaley;

            SDL_RenderGetScale(sdlRenderer, &scalex, &scaley);
            SDL_RenderGetViewport(sdlRenderer, &vprect);

            x = (int)(x / scalex);
            y = (int)(y / scaley);

            x-=vprect.x;
            y-=vprect.y;

            if (x<0)
                x=0;
            if (x>=vprect.w)
                x=vprect.w-1;
            if (y<0)
                y=0;
            if (y>=vprect.h)
                y=vprect.h-1;
        }
    }
#endif

    return Py_BuildValue("(ii)", x, y);
}

static PyObject *
mouse_get_rel(PyObject *self)
{
    int x, y;

    VIDEO_INIT_CHECK();

    SDL_GetRelativeMouseState(&x, &y);

/*
#if IS_SDLv2
    SDL_Window *sdlWindow = pg_GetDefaultWindow();
    SDL_Renderer *sdlRenderer = SDL_GetRenderer(sdlWindow);
    if (sdlRenderer!=NULL){
        float scalex, scaley;

        SDL_RenderGetScale(sdlRenderer, &scalex, &scaley);

        x/=scalex;
        y/=scaley;
    }
#endif
*/
    return Py_BuildValue("(ii)", x, y);
}

static PyObject *
mouse_get_pressed(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *tuple;
    int state;
    int num_buttons = 3;

    static char *kwids[] = {
        "num_buttons",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", kwids,
                                     &num_buttons))
    return NULL;
    VIDEO_INIT_CHECK();

    if (num_buttons != 3 && num_buttons != 5)
        return RAISE(PyExc_ValueError, "Number of buttons needs to be 3 or 5.");

    state = SDL_GetMouseState(NULL, NULL);
    if (!(tuple = PyTuple_New(num_buttons)))
        return NULL;

    PyTuple_SET_ITEM(tuple, 0, PyBool_FromLong((state & SDL_BUTTON_LMASK) != 0));
    PyTuple_SET_ITEM(tuple, 1, PyBool_FromLong((state & SDL_BUTTON_MMASK) != 0));
    PyTuple_SET_ITEM(tuple, 2, PyBool_FromLong((state & SDL_BUTTON_RMASK) != 0));

    if (num_buttons == 5) {
        PyTuple_SET_ITEM(tuple, 3, PyBool_FromLong((state & SDL_BUTTON_X1MASK) != 0));
        PyTuple_SET_ITEM(tuple, 4, PyBool_FromLong((state & SDL_BUTTON_X2MASK) != 0));
    }

    return tuple;
}

static PyObject *
mouse_set_visible(PyObject *self, PyObject *args)
{
    int toggle;
    #if IS_SDLv2
        int mode;
        SDL_Window *win = NULL;
        Uint32 window_flags = 0;
    #endif

    if (!PyArg_ParseTuple(args, "i", &toggle))
        return NULL;
    VIDEO_INIT_CHECK();

    #if IS_SDLv2
        win = pg_GetDefaultWindow();
        if (win) {
            mode = SDL_GetWindowGrab(win);
            if ((mode == SDL_ENABLE) & !toggle) {
                SDL_SetRelativeMouseMode(1);
            } else {
                SDL_SetRelativeMouseMode(0);
            }
            window_flags = SDL_GetWindowFlags(win);
            if (!toggle && (window_flags & SDL_WINDOW_FULLSCREEN_DESKTOP ||
                            window_flags & SDL_WINDOW_FULLSCREEN))
            {
                SDL_SetHint(SDL_HINT_WINDOW_FRAME_USABLE_WHILE_CURSOR_HIDDEN,
                            "0");
            }
            else
            {
                SDL_SetHint(SDL_HINT_WINDOW_FRAME_USABLE_WHILE_CURSOR_HIDDEN,
                            "1");
            }
        }
    #endif

    toggle = SDL_ShowCursor(toggle);
    return PyInt_FromLong(toggle);
}

static PyObject *
mouse_get_visible(PyObject *self, PyObject *args)
{
    int result;

    VIDEO_INIT_CHECK();

    result = SDL_ShowCursor(SDL_QUERY);

    if (0 > result) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    return PyBool_FromLong(result);
}

static PyObject *
mouse_get_focused(PyObject *self)
{
    VIDEO_INIT_CHECK();
#if IS_SDLv1
    return PyInt_FromLong((SDL_GetAppState() & SDL_APPMOUSEFOCUS) != 0);
#else  /* IS_SDLv2 */
    return PyInt_FromLong(SDL_GetMouseFocus() != NULL);
#endif /* IS_SDLv2 */
}

struct CursorData {
    int w;
    int h;
    int spotx;
    int spoty;
    PyObject *xormask, *andmask;
    pgSurfaceObject *surfobj;
    int constant;
    int type;
} cursor_data;

//possible CursorData.type attributes:
#define COLOR_CURSOR 2
#define BITMAP_CURSOR 1
#define SYSTEM_CURSOR 0 

/* Defining SYSTEM_CURSOR as zero does more than it seems.
Since cursor_data type and constant both initialize to zero, 
When get_cursor() is called before set_cursor() has set something,
it sees a type 0 (system cursor) cursor with the constant 0.
The SDL2 constant SDL_SYSTEM_CURSOR_ARROW is 0, so it wil return the
default cursor.*/

static PyObject *
_set_bitmap_cursor(int w, int h, int spotx, int spoty, PyObject* xormask, PyObject* andmask) {
    Uint8 *xordata = NULL, *anddata = NULL;
    int xorsize, andsize, loop;
    int val;
    SDL_Cursor *lastcursor, *cursor = NULL;

    if (!PySequence_Check(xormask) || !PySequence_Check(andmask))
        return RAISE(PyExc_TypeError, "xormask and andmask must be sequences");

    if (w % 8)
        return RAISE(PyExc_ValueError, "Cursor width must be divisible by 8.");

    xorsize = PySequence_Length(xormask);
    andsize = PySequence_Length(andmask);

    if (xorsize != w * h / 8 || andsize != w * h / 8)
        return RAISE(PyExc_ValueError,
                     "bitmasks must be sized width*height/8");

    xordata = (Uint8 *)malloc(xorsize);
    anddata = (Uint8 *)malloc(andsize);

    if ((NULL == xordata) || (NULL == anddata)) {
        free(xordata);
        free(anddata);
        return PyErr_NoMemory();
    }

    for (loop = 0; loop < xorsize; ++loop) {
        if (!pg_IntFromObjIndex(xormask, loop, &val))
            goto interror;
        xordata[loop] = (Uint8)val;
        if (!pg_IntFromObjIndex(andmask, loop, &val))
            goto interror;
        anddata[loop] = (Uint8)val;
    }

    cursor = SDL_CreateCursor(xordata, anddata, w, h, spotx, spoty);
    free(xordata);
    free(anddata);
    xordata = NULL;
    anddata = NULL;

    if (!cursor)
        return RAISE(pgExc_SDLError, SDL_GetError());

    lastcursor = SDL_GetCursor();
    SDL_SetCursor(cursor);
    SDL_FreeCursor(lastcursor);

    //To make sure that the mask data sticks around, it has to have its ref count increased
    //Conversely, the old data stored in cursor_data (if it is there) doesn't need to be around anymore
    Py_XDECREF(cursor_data.xormask);
    Py_XDECREF(cursor_data.andmask);
    Py_INCREF(xormask);
    Py_INCREF(andmask);

    cursor_data.type = BITMAP_CURSOR;
    cursor_data.xormask = xormask;
    cursor_data.andmask = andmask;
    cursor_data.w = w;
    cursor_data.h = h;
    cursor_data.spotx = spotx;
    cursor_data.spoty = spoty;
    Py_RETURN_NONE;

interror:
    if (xordata)
        free(xordata);
    if (anddata)
        free(anddata);
    return RAISE(PyExc_TypeError, "Invalid number in mask array");
}

static PyObject *
_set_system_cursor(int constant) {
#if IS_SDLv2
    SDL_Cursor *lastcursor, *cursor = NULL;

    cursor = SDL_CreateSystemCursor(constant);
    if (!cursor){
        //SDL_GetError() wasn't returning relevant stuff when this function fails
        return RAISE(pgExc_SDLError, "Error while creating system cursor");
    }

    lastcursor = SDL_GetCursor();
    SDL_SetCursor(cursor);
    SDL_FreeCursor(lastcursor);

    cursor_data.type = SYSTEM_CURSOR;
    cursor_data.constant = constant;   
    Py_RETURN_NONE;
#endif
return RAISE(PyExc_TypeError, "System cursors from constant are unavailable in SDL1");
}

static PyObject *
_set_color_cursor(int spotx, int spoty, pgSurfaceObject *surfobj) {
#if IS_SDLv2
    SDL_Cursor *lastcursor, *cursor = NULL;
    SDL_Surface *surf = NULL;
    surf = pgSurface_AsSurface(surfobj);

    cursor = SDL_CreateColorCursor(surf, spotx, spoty);
    if (!cursor)
        return RAISE(pgExc_SDLError, SDL_GetError());

    lastcursor = SDL_GetCursor();
    SDL_SetCursor(cursor);
    SDL_FreeCursor(lastcursor);

    //To make sure that the surface is stored properly, it has to have its ref count increased
    //Conversely, the old data stored in cursor_data (if it is there) doesn't need to be around anymore
    Py_XDECREF(cursor_data.surfobj);
    Py_INCREF(surfobj);

    cursor_data.type = COLOR_CURSOR;
    cursor_data.spotx = spotx;
    cursor_data.spoty = spoty;
    cursor_data.surfobj = surfobj;
    Py_RETURN_NONE;

#endif
return RAISE(PyExc_TypeError, "Cursors from a surface are unavailable in SDL1");
}

static PyObject *
mouse_set_system_cursor(PyObject *self, PyObject *args)
{
    int constant;

    if (PyErr_WarnEx(PyExc_DeprecationWarning, "set_cursor() now has all the functionality of set_system_cursor(), so set_system_cursor() will be deprecated in pygame 2.2", 1) == -1) {
        return NULL;
    }
    
    VIDEO_INIT_CHECK();

    if (!PyArg_ParseTuple(args, "i", &constant)) {
        return NULL;
    }

    return _set_system_cursor(constant);
}

//mouse.set_cursor goes through a python layer first, see cursors.py
static PyObject *
mouse_set_cursor(PyObject *self, PyObject *args, PyObject *kwds)
{
    //normal_cursor stuff
    int w=0, h=0, spotx, spoty;
    PyObject *xormask, *andmask;

    //system_cursor stuff
    int constant = -1;

    //color_cursor stuff
    pgSurfaceObject *surfobj = NULL;

    static char *keywords[] = {"system", "bitmap", "color", NULL};

    VIDEO_INIT_CHECK();

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|(i)((ii)(ii)OO)((ii)O!)", keywords, &constant,
                                     &w, &h, &spotx, &spoty, &xormask, &andmask, &spotx, &spoty,
                                     &pgSurface_Type, &surfobj)) {
        return NULL;
    }

    if (constant > -1) {
        return _set_system_cursor(constant);
    }
    else if (w && h) {
        return _set_bitmap_cursor(w, h, spotx, spoty, xormask, andmask);
    }
    else if (surfobj) {
        return _set_color_cursor(spotx, spoty, surfobj);
    }
    return RAISE(PyExc_ValueError, "Invalid cursor format: no valid template found");  
}

//mouse.get_cursor goes through a python layer first, see cursors.py
static PyObject*
mouse_get_cursor(PyObject *self)
{
    VIDEO_INIT_CHECK();

    if (cursor_data.type == SYSTEM_CURSOR) {
        return Py_BuildValue("(i)", cursor_data.constant);
    }
    if (cursor_data.type == BITMAP_CURSOR) {
        return Py_BuildValue("(ii)(ii)OO", cursor_data.w, cursor_data.h, cursor_data.spotx, cursor_data.spoty, cursor_data.xormask, cursor_data.andmask); 
    }
    if (cursor_data.type == COLOR_CURSOR) {
        return Py_BuildValue("(ii)O", cursor_data.spotx, cursor_data.spoty, cursor_data.surfobj);
    }
    return RAISE(pgExc_SDLError, "Cursor not found");
}

static PyMethodDef _mouse_methods[] = {
    {"set_pos", mouse_set_pos, METH_VARARGS, DOC_PYGAMEMOUSESETPOS},
    {"get_pos", (PyCFunction)mouse_get_pos, METH_VARARGS,
     DOC_PYGAMEMOUSEGETPOS},
    {"get_rel", (PyCFunction)mouse_get_rel, METH_VARARGS,
     DOC_PYGAMEMOUSEGETREL},
    {"get_pressed", (PyCFunction)mouse_get_pressed,
     METH_VARARGS | METH_KEYWORDS, DOC_PYGAMEMOUSEGETPRESSED},
    {"set_visible", mouse_set_visible, METH_VARARGS,
     DOC_PYGAMEMOUSESETVISIBLE},
    {"get_visible", mouse_get_visible, METH_NOARGS, DOC_PYGAMEMOUSEGETVISIBLE},
    {"get_focused", (PyCFunction)mouse_get_focused, METH_VARARGS,
     DOC_PYGAMEMOUSEGETFOCUSED},
    {"set_system_cursor", mouse_set_system_cursor, METH_VARARGS, "set_system_cursor(constant) -> None\nset the mouse cursor to a system variant"},
    {"_set_cursor", (PyCFunction)mouse_set_cursor, METH_VARARGS | METH_KEYWORDS, "Internal API for mouse.set_cursor"},
    {"_get_cursor", (PyCFunction)mouse_get_cursor, METH_NOARGS, "Internal API for mouse.get_cursor"},
    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(mouse)
{
    PyObject *module;

#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "mouse",
                                         DOC_PYGAMEMOUSE,
                                         -1,
                                         _mouse_methods,
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
    import_pygame_surface();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    /* create the module */
#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3("mouse", _mouse_methods, DOC_PYGAMEMOUSE);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    MODINIT_RETURN(module);
}
