/*
    pygame - Python Game Library
    Copyright (C) 2006, 2007 Rene Dudfield, Marcus von Appen

    Originally written and put in the public domain by Sam Lantinga.

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

/* Handle clipboard text and data in arbitrary formats */

#include <limits.h>
#include <stdio.h>

#include "SDL.h"

#include "SDL_syswm.h"

#include "scrap.h"

#include "pygame.h"

#include "doc/scrap_doc.h"

#include "pgcompat.h"

/**
 * Indicates, whether pygame.scrap was initialized or not.
 */
static int _scrapinitialized = 0;

/**
 * Currently active Clipboard object.
 */
static ScrapClipType _currentmode;
static PyObject *_selectiondata = NULL;
static PyObject *_clipdata = NULL;

/* Forward declarations. */
static PyObject *
_scrap_get_types(PyObject *self, PyObject *args);
static PyObject *
_scrap_contains(PyObject *self, PyObject *args);
static PyObject *
_scrap_get_scrap(PyObject *self, PyObject *args);
static PyObject *
_scrap_put_scrap(PyObject *self, PyObject *args);
static PyObject *
_scrap_lost_scrap(PyObject *self, PyObject *args);
static PyObject *
_scrap_set_mode(PyObject *self, PyObject *args);

/* Determine what type of clipboard we are using */
#if defined(__unix__) && defined(SDL_VIDEO_DRIVER_X11)
/*!defined(__QNXNTO__) &&*/
#define X11_SCRAP
#include <time.h> /* Needed for clipboard timeouts. */
#include "scrap_x11.c"
#elif defined(__WIN32__)
#define WIN_SCRAP
#include "scrap_win.c"
/*
#elif defined(__QNXNTO__)
    #define QNX_SCRAP
static uint32_t _cliptype = 0;
    #include "scrap_qnx.c"
*/
#elif defined(__APPLE__)
#define MAC_SCRAP
#include "scrap_mac.c"
#else
#error Unknown window manager for clipboard handling
#endif /* scrap type */

/**
 * \brief Indicates whether the scrap module is already initialized.
 *
 * \return 0 if the module is not initialized, 1, if it is.
 */
int
pygame_scrap_initialized(void)
{
    return _scrapinitialized;
}

#if !defined(MAC_SCRAP)
/*
 * Initializes the pygame scrap module.
 */
static PyObject *
_scrap_init(PyObject *self, PyObject *args)
{
    VIDEO_INIT_CHECK();

    if (!pygame_scrap_initialized()) {
        Py_XDECREF(_clipdata);
        Py_XDECREF(_selectiondata);
        _clipdata = PyDict_New();
        _selectiondata = PyDict_New();
    }

    /* In case we've got not video surface, we won't initialize
     * anything.
     */
    if (!SDL_GetVideoSurface())
        return RAISE(pgExc_SDLError, "No display mode is set");
    if (!pygame_scrap_init())
        return RAISE(pgExc_SDLError, SDL_GetError());

    Py_RETURN_NONE;
}
#endif

/*
 * Indicates whether the scrap module is currently initialized.
 *
 * Note: All platforms supported here.
 */
static PyObject *
_scrap_get_init(PyObject *self, PyObject *args)
{
    return PyBool_FromLong(pygame_scrap_initialized());
}

#if !defined(MAC_SCRAP)
/*
 * Gets the currently available types from the active clipboard.
 */
static PyObject *
_scrap_get_types(PyObject *self, PyObject *args)
{
    int i = 0;
    char **types;
    char *type;
    PyObject *list;
    PyObject *tmp;

    PYGAME_SCRAP_INIT_CHECK();
    if (!pygame_scrap_lost()) {
        switch (_currentmode) {
            case SCRAP_SELECTION:
                return PyDict_Keys(_selectiondata);
            case SCRAP_CLIPBOARD:
            default:
                return PyDict_Keys(_clipdata);
        }
    }

    list = PyList_New(0);
    types = pygame_scrap_get_types();
    if (!types)
        return list;
    while (types[i] != NULL) {
        type = types[i];
        tmp = PyUnicode_DecodeASCII(type, strlen(type), 0);
        if (!tmp) {
            Py_DECREF(list);
            return 0;
        }
        if (PyList_Append(list, tmp)) {
            Py_DECREF(list);
            Py_DECREF(tmp);
            return 0;
        }
        Py_DECREF(tmp);
        i++;
    }
    return list;
}
#endif

#if !defined(MAC_SCRAP)
/*
 * Checks whether the active clipboard contains a certain type.
 */
static PyObject *
_scrap_contains(PyObject *self, PyObject *args)
{
    char *type = NULL;

    if (!PyArg_ParseTuple(args, "s", &type))
        return NULL;
    if (pygame_scrap_contains(type))
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}
#endif

#if !defined(MAC_SCRAP)
/*
 * Gets the content for a certain type from the active clipboard.
 */
static PyObject *
_scrap_get_scrap(PyObject *self, PyObject *args)
{
    char *scrap = NULL;
    PyObject *retval;
    char *scrap_type;
    unsigned long count;

    PYGAME_SCRAP_INIT_CHECK();

    if (!PyArg_ParseTuple(args, "s", &scrap_type))
        return NULL;

    if (!pygame_scrap_lost()) {
        /* Still own the clipboard. */
        PyObject *scrap_dict = NULL;
        PyObject *key = NULL;
        PyObject *val = NULL;

        switch (_currentmode) {
            case SCRAP_SELECTION:
                scrap_dict = _selectiondata;
                break;

            case SCRAP_CLIPBOARD:
            default:
                scrap_dict = _clipdata;
                break;
        }

#if PY3
        key = PyUnicode_FromString(scrap_type);
        if (NULL == key) {
            return PyErr_Format(PyExc_ValueError,
                                "invalid scrap data type identifier (%s)",
                                scrap_type);
        }

        val = PyDict_GetItemWithError(scrap_dict, key);
        Py_DECREF(key);

        if (NULL == val) {
            if (PyErr_Occurred()) {
                return PyErr_Format(PyExc_SystemError,
                                    "pygame.scrap internal error (key=%s)",
                                    scrap_type);
            }

            Py_RETURN_NONE;
        }
#else  /* !PY3 */
        val = PyDict_GetItemString(scrap_dict, scrap_type);
        if (NULL == val) {
            Py_RETURN_NONE;
        }
#endif /* !PY3 */

        Py_INCREF(val);
        return val;
    }

    /* pygame_get_scrap() only returns NULL or !NULL, but won't set any
     * errors. */
    scrap = pygame_scrap_get(scrap_type, &count);
    if (!scrap)
        Py_RETURN_NONE;

    retval = Bytes_FromStringAndSize(scrap, count);
    return retval;
}
#endif

#if !defined(MAC_SCRAP)
/*
 * This will put a python string into the clipboard.
 */
static PyObject *
_scrap_put_scrap(PyObject *self, PyObject *args)
{
    int scraplen;
    char *scrap = NULL;
    char *scrap_type;
    PyObject *tmp;
#if PY3
    static const char argfmt[] = "sy#";
#else
    static char argfmt[] = "st#";
#endif

    PYGAME_SCRAP_INIT_CHECK();

    if (!PyArg_ParseTuple(args, argfmt, &scrap_type, &scrap, &scraplen)) {
        return NULL;
    }

    /* Set it in the clipboard. */
    if (!pygame_scrap_put(scrap_type, scraplen, scrap))
        return RAISE(pgExc_SDLError,
                     "content could not be placed in clipboard.");

    /* Add or replace the set value. */
    switch (_currentmode) {
        case SCRAP_SELECTION: {
            tmp = Bytes_FromStringAndSize(scrap, scraplen);
            PyDict_SetItemString(_selectiondata, scrap_type, tmp);
            Py_DECREF(tmp);
            break;
        }
        case SCRAP_CLIPBOARD:
        default: {
            tmp = Bytes_FromStringAndSize(scrap, scraplen);
            PyDict_SetItemString(_clipdata, scrap_type, tmp);
            Py_DECREF(tmp);
            break;
        }
    }

    Py_RETURN_NONE;
}
#endif

#if !defined(MAC_SCRAP)
/*
 * Checks whether the pygame window has lost the clipboard.
 */
static PyObject *
_scrap_lost_scrap(PyObject *self, PyObject *args)
{
    PYGAME_SCRAP_INIT_CHECK();

    if (pygame_scrap_lost())
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}
#endif

#if !defined(MAC_SCRAP)
/*
 * Sets the clipboard mode. This only works for the X11 environment, which
 * diverses between mouse selections and the clipboard.
 */
static PyObject *
_scrap_set_mode(PyObject *self, PyObject *args)
{
    PYGAME_SCRAP_INIT_CHECK();
    if (!PyArg_ParseTuple(args, "i", &_currentmode))
        return NULL;

    if (_currentmode != SCRAP_CLIPBOARD && _currentmode != SCRAP_SELECTION)
        return RAISE(PyExc_ValueError, "invalid clipboard mode");

#ifndef X11_SCRAP
    /* Force the clipboard, if not in a X11 environment. */
    _currentmode = SCRAP_CLIPBOARD;
#endif
    Py_RETURN_NONE;
}

#endif /* !defined(MAC_SCRAP) */

static PyMethodDef scrap_builtins[] = {
/*
 * Only initialise these functions for ones we know about.
 *
 * Note, the macosx stuff is done in sdlosx_main.m
 */
#if (defined(X11_SCRAP) || defined(WIN_SCRAP) || defined(QNX_SCRAP) || \
     defined(MAC_SCRAP))

    {"init", _scrap_init, 1, DOC_PYGAMESCRAPINIT},
    {"get_init", _scrap_get_init, METH_NOARGS, DOC_PYGAMESCRAPGETINIT},
    {"contains", _scrap_contains, METH_VARARGS, DOC_PYGAMESCRAPCONTAINS},
    {"get", _scrap_get_scrap, METH_VARARGS, DOC_PYGAMESCRAPGET},
    {"get_types", _scrap_get_types, METH_NOARGS, DOC_PYGAMESCRAPGETTYPES},
    {"put", _scrap_put_scrap, METH_VARARGS, DOC_PYGAMESCRAPPUT},
    {"lost", _scrap_lost_scrap, METH_NOARGS, DOC_PYGAMESCRAPLOST},
    {"set_mode", _scrap_set_mode, METH_VARARGS, DOC_PYGAMESCRAPSETMODE},

#endif
    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(scrap)
{
#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "scrap",
                                         "",
                                         -1,
                                         scrap_builtins,
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
    return PyModule_Create(&_module);
#else
    Py_InitModule3(MODPREFIX "scrap", scrap_builtins, NULL);
#endif
}
