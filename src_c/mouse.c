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
mouse_get_pressed(PyObject *self)
{
    PyObject *tuple;
    int state;

    VIDEO_INIT_CHECK();

    state = SDL_GetMouseState(NULL, NULL);
    if (!(tuple = PyTuple_New(3)))
        return NULL;

    PyTuple_SET_ITEM(tuple, 0, PyInt_FromLong((state & SDL_BUTTON(1)) != 0));
    PyTuple_SET_ITEM(tuple, 1, PyInt_FromLong((state & SDL_BUTTON(2)) != 0));
    PyTuple_SET_ITEM(tuple, 2, PyInt_FromLong((state & SDL_BUTTON(3)) != 0));

    return tuple;
}

static PyObject *
mouse_set_visible(PyObject *self, PyObject *args)
{
    int toggle;
    #if IS_SDLv2
        int mode;
        SDL_Window *win = NULL;
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

static PyObject *
mouse_set_cursor(PyObject *self, PyObject *args)
{
    int w, h, spotx, spoty;
    PyObject *xormask, *andmask;
    Uint8 *xordata = NULL, *anddata = NULL;
    int xorsize, andsize, loop;
    int val;
    SDL_Cursor *lastcursor, *cursor = NULL;

    if (!PyArg_ParseTuple(args, "(ii)(ii)OO", &w, &h, &spotx, &spoty, &xormask,
                          &andmask))
        return NULL;

    VIDEO_INIT_CHECK();

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

    Py_RETURN_NONE;

interror:
    if (xordata)
        free(xordata);
    if (anddata)
        free(anddata);
    return RAISE(PyExc_TypeError, "Invalid number in mask array");
}

#if IS_SDLv1
static PyObject *
mouse_get_cursor(PyObject *self)
{
    SDL_Cursor *cursor = NULL;
    PyObject *xordata, *anddata;
    int size, loop, w, h, spotx, spoty;

    VIDEO_INIT_CHECK();

    cursor = SDL_GetCursor();
    if (!cursor)
        return RAISE(pgExc_SDLError, SDL_GetError());

    w = cursor->area.w;
    h = cursor->area.h;
    spotx = cursor->hot_x;
    spoty = cursor->hot_y;

    size = cursor->area.w * cursor->area.h / 8;
    xordata = PyTuple_New(size);
    if (!xordata)
        return NULL;
    anddata = PyTuple_New(size);
    if (!anddata) {
        Py_DECREF(xordata);
        return NULL;
    }

    for (loop = 0; loop < size; ++loop) {
        PyTuple_SET_ITEM(xordata, loop, PyInt_FromLong(cursor->data[loop]));
        PyTuple_SET_ITEM(anddata, loop, PyInt_FromLong(cursor->mask[loop]));
    }

    return Py_BuildValue("((ii)(ii)NN)", w, h, spotx, spoty, xordata, anddata);
}
#endif /* IS_SDLv1 */

static PyMethodDef _mouse_methods[] = {
    {"set_pos", mouse_set_pos, METH_VARARGS, DOC_PYGAMEMOUSESETPOS},
    {"get_pos", (PyCFunction)mouse_get_pos, METH_VARARGS,
     DOC_PYGAMEMOUSEGETPOS},
    {"get_rel", (PyCFunction)mouse_get_rel, METH_VARARGS,
     DOC_PYGAMEMOUSEGETREL},
    {"get_pressed", (PyCFunction)mouse_get_pressed, METH_VARARGS,
     DOC_PYGAMEMOUSEGETPRESSED},
    {"set_visible", mouse_set_visible, METH_VARARGS,
     DOC_PYGAMEMOUSESETVISIBLE},
    {"get_visible", mouse_get_visible, METH_NOARGS, DOC_PYGAMEMOUSEGETVISIBLE},
    {"get_focused", (PyCFunction)mouse_get_focused, METH_VARARGS,
     DOC_PYGAMEMOUSEGETFOCUSED},
    {"set_cursor", mouse_set_cursor, METH_VARARGS, DOC_PYGAMEMOUSESETCURSOR},
#if IS_SDLv1
    {"get_cursor", (PyCFunction)mouse_get_cursor, METH_VARARGS,
     DOC_PYGAMEMOUSEGETCURSOR},
#endif /* IS_SDLv1 */

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
