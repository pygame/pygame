/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners, 2008 Marcus von Appen

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
#define PYGAME_SDLWM_INTERNAL

#include <SDL_syswm.h>
#include "pgsdl.h"
#include "sdlwm_doc.h"

static PyObject* _sdl_wmsetcaption (PyObject *self, PyObject *args);
static PyObject* _sdl_wmgetcaption (PyObject *self);
static PyObject* _sdl_wmseticon (PyObject *self, PyObject *args);
static PyObject* _sdl_wmiconify (PyObject *self);
static PyObject* _sdl_wmtogglefullscreen (PyObject *self);
static PyObject* _sdl_wmgetinfo (PyObject *self);
static PyObject* _sdl_wmgrabinput (PyObject *self, PyObject *args);

static PyMethodDef _wm_methods[] = {
    { "get_caption", (PyCFunction)_sdl_wmgetcaption, METH_NOARGS,
      DOC_WM_GET_CAPTION },
    { "set_caption", _sdl_wmsetcaption, METH_VARARGS, DOC_WM_SET_CAPTION },
    { "set_icon", _sdl_wmseticon, METH_VARARGS, DOC_WM_SET_ICON },
    { "iconify_window", (PyCFunction)_sdl_wmiconify, METH_NOARGS,
      DOC_WM_ICONIFY_WINDOW },
    { "toggle_fullscreen", (PyCFunction)_sdl_wmtogglefullscreen, METH_NOARGS,
      DOC_WM_TOGGLE_FULLSCREEN },
    { "get_info", (PyCFunction) _sdl_wmgetinfo, METH_NOARGS, DOC_WM_GET_INFO },
    { "grab_input", _sdl_wmgrabinput, METH_O, DOC_WM_GRAB_INPUT },
    { NULL, NULL, 0, NULL }
};

static PyObject*
_sdl_wmsetcaption (PyObject *s, PyObject *args)
{
    char *title, *icon = NULL;

    ASSERT_VIDEO_SURFACE_SET(NULL);

    if (!PyArg_ParseTuple (args, "s|s:set_caption", &title, &icon))
        return NULL;

    if (!icon)
        icon = title;

    SDL_WM_SetCaption (title, icon);
    Py_RETURN_NONE;
}

static PyObject*
_sdl_wmgetcaption (PyObject *self)
{
    char *title, *icon;

    ASSERT_VIDEO_SURFACE_SET(NULL);
    SDL_WM_GetCaption (&title, &icon);
    return Py_BuildValue ("(ss)", title, icon);
}

static PyObject*
_sdl_wmseticon (PyObject *self, PyObject *args)
{
    PyObject *surface;
    PyObject *mask = NULL;
    SDL_Surface *sfmask = NULL;

    ASSERT_VIDEO_INIT(NULL);

    if (SDL_GetVideoSurface ())
    {
        PyErr_SetString (PyExc_PyGameError, "video surface already exists");
        return NULL;
    }

    if (!PyArg_ParseTuple (args, "O|O:set_icon", &surface, &mask))
        return NULL;

    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }

    if (mask)
    {
        PyErr_SetString (PyExc_NotImplementedError,
            "icon masks are not supported yet");
    }
    /* TODO: support the mask */
    SDL_WM_SetIcon (((PySDLSurface*)surface)->surface, NULL);
    Py_RETURN_NONE;
}

static PyObject*
_sdl_wmiconify (PyObject *self)
{
    ASSERT_VIDEO_SURFACE_SET(NULL);

    if (!SDL_WM_IconifyWindow ())
        Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}

static PyObject*
_sdl_wmtogglefullscreen (PyObject *self)
{
    ASSERT_VIDEO_SURFACE_SET(NULL);

    if (!SDL_WM_ToggleFullScreen (SDL_GetVideoSurface ()))
        Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}

static PyObject*
_sdl_wmgetinfo (PyObject *self)
{
    PyObject *dict;
    PyObject *tmp;
    SDL_SysWMinfo info;

    ASSERT_VIDEO_SURFACE_SET(NULL);

    dict = PyDict_New ();
    if (!dict)
        return NULL;

    SDL_VERSION (&(info.version));
    if (SDL_GetWMInfo (&info) <= 0)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }

#ifdef SDL_VIDEO_DRIVER_X11
    tmp = PyLong_FromUnsignedLong (info.info.x11.window);
    PyDict_SetItemString (dict, "window", tmp);
    Py_DECREF (tmp);

    tmp = PyCObject_FromVoidPtr (info.info.x11.display, NULL);
    PyDict_SetItemString (dict, "display", tmp);
    Py_DECREF (tmp);

    tmp = PyCObject_FromVoidPtr (info.info.x11.lock_func, NULL);
    PyDict_SetItemString (dict, "lock_func", tmp);
    Py_DECREF (tmp);

    tmp = PyCObject_FromVoidPtr (info.info.x11.unlock_func, NULL);
    PyDict_SetItemString (dict, "unlock_func", tmp);
    Py_DECREF (tmp);

    tmp = PyLong_FromUnsignedLong (info.info.x11.fswindow);
    PyDict_SetItemString (dict, "fswindow", tmp);
    Py_DECREF (tmp);

    tmp = PyLong_FromUnsignedLong (info.info.x11.wmwindow);
    PyDict_SetItemString (dict, "wmwindow", tmp);
    Py_DECREF (tmp);

#elif defined(SDL_VIDEO_DRIVER_NANOX)
    tmp = PyInt_FromLong (info.window);
    PyDict_SetItemString (dict, "window", tmp);
    Py_DECREF (tmp);

#elif defined(SDL_VIDEO_DRIVER_WINDIB) || defined(SDL_VIDEO_DRIVER_DDRAW) || defined(SDL_VIDEO_DRIVER_GAPI)
    tmp = PyInt_FromLong ((long)info.window);
    PyDict_SetItemString (dict, "window", tmp);
    Py_DECREF (tmp);

    tmp = PyInt_FromLong ((long)info.hglrc);
    PyDict_SetItemString (dict, "hglrc", tmp);
    Py_DECREF (tmp);

#elif defined(SDL_VIDEO_DRIVER_RISCOS)
    tmp = PyInt_FromLong (info.window);
    PyDict_SetItemString (dict, "window", tmp);
    Py_DECREF (tmp);

    tmp = PyInt_FromLong (info.wimpVersion);
    PyDict_SetItemString (dict, "wimpVersion", tmp);
    Py_DECREF (tmp);

    tmp = PyInt_FromLong (info.taskHandle);
    PyDict_SetItemString (dict, "taskHandle", tmp);
    Py_DECREF (tmp);

#else
    tmp = PyInt_FromLong (info.data);
    PyDict_SetItemString (dict, "data", tmp);
    Py_DECREF (tmp);
#endif

    return dict;
}

static PyObject*
_sdl_wmgrabinput (PyObject *self, PyObject *args)
{
    int mode;

    ASSERT_VIDEO_SURFACE_SET(NULL);

    if (PyBool_Check (args))
    {
        if (args == Py_True)
            mode = SDL_WM_GrabInput (SDL_GRAB_ON);
        else
            mode = SDL_WM_GrabInput (SDL_GRAB_OFF);
    }
    else if (IntFromObj (args, &mode))
        mode = SDL_WM_GrabInput (mode);
    else
    {
        PyErr_SetString (PyExc_TypeError, "argument must be bool or int");
        return NULL;
    }
    return PyInt_FromLong (mode);
}

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_wm (void)
#else
PyMODINIT_FUNC initwm (void)
#endif
{
    PyObject *mod;

#ifdef IS_PYTHON_3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "wm",
        DOC_WM,
        -1,
        _wm_methods,
        NULL, NULL, NULL, NULL
    };
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("wm", _wm_methods, DOC_WM);
#endif
    if (!mod)
        goto fail;

    if (import_pygame2_base () < 0)
        goto fail;
    if (import_pygame2_sdl_base () < 0)
        goto fail;
    if (import_pygame2_sdl_video () < 0)
        goto fail;
    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
