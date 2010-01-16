/*
  NET2 is a threaded, event based, network IO library for SDL.
  Copyright (C) 2002 Bob Pendleton

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public License
  as published by the Free Software Foundation; either version 2.1
  of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free
  Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
  02111-1307 USA

  If you do not wish to comply with the terms of the LGPL please
  contact the author as other terms are available for a fee.

  Bob Pendleton
  Bob@Pendleton.com
*/

#define PYGAME_SDLFASTEVENT_INTERNAL

#include "pgsdl.h"
#include "fastevents.h"
#include "sdlextfastevent_doc.h"

typedef struct {
    int fewasinit;
} _FastEventState;

#ifdef IS_PYTHON_3
struct PyModuleDef _feventmodule; /* Forward declaration */
#define FASTEVENT_MOD_STATE(mod) ((_FastEventState*)PyModule_GetState(mod))
#define FASTEVENT_STATE FASTEVENT_MOD_STATE(PyState_FindModule(&_feventmodule))
#else
static _FastEventState _modstate;
#define FASTEVENT_MOD_STATE(mod) (&_modstate)
#define FASTEVENT_STATE FASTEVENT_MOD_STATE(NULL)
#endif
static int _fastevent_clear (PyObject *mod);

#define ASSERT_FASTEVENT_INIT(x)                                        \
    ASSERT_VIDEO_INIT(x);                                               \
    if (!FASTEVENT_STATE->fewasinit)                                    \
    {                                                                   \
        PyErr_SetString(PyExc_PyGameError,                              \
            "fastevent subsystem not initialized");                     \
        return (x);                                                     \
    }

static PyObject* _sdl_feventinit (PyObject *self);
static PyObject* _sdl_feventquit (PyObject *self);
static PyObject* _sdl_feventpump (PyObject *self);
static PyObject* _sdl_feventpoll (PyObject *self);
static PyObject* _sdl_feventwait (PyObject *self);
static PyObject* _sdl_feventpush (PyObject *self, PyObject *args);
static PyObject* _sdl_feventget (PyObject *self);

static PyMethodDef _fevent_methods[] = {
    { "init", (PyCFunction)_sdl_feventinit, METH_NOARGS, DOC_FASTEVENT_INIT },
    { "quit", (PyCFunction)_sdl_feventquit, METH_NOARGS, DOC_FASTEVENT_QUIT },
    { "poll", (PyCFunction)_sdl_feventpoll, METH_NOARGS, DOC_FASTEVENT_POLL },
    { "wait", (PyCFunction)_sdl_feventwait, METH_NOARGS, DOC_FASTEVENT_WAIT },
    { "push", _sdl_feventpush, METH_VARARGS, DOC_FASTEVENT_PUSH },
    { "get", (PyCFunction)_sdl_feventget, METH_NOARGS, DOC_FASTEVENT_GET },
    { NULL, NULL, 0, NULL }
};

static PyObject*
_sdl_feventinit (PyObject *self)
{
    _FastEventState *state = FASTEVENT_MOD_STATE (self);
    ASSERT_VIDEO_INIT (NULL);

    if (state->fewasinit)
        Py_RETURN_NONE;

#ifndef WITH_THREAD
    PyErr_SetString (PyExc_PyGameError, "fastevent requires a threaded Python");
    return NULL;
#else
    if (FE_Init () != 0)
    {
        PyErr_SetString (PyExc_PyGameError, FE_GetError ());
        return NULL;
    }
    state->fewasinit = 1;
    Py_RETURN_NONE;
#endif
}

static PyObject*
_sdl_feventquit (PyObject *self)
{
    _FastEventState *state = FASTEVENT_MOD_STATE (self);
    if (!state->fewasinit)
        Py_RETURN_NONE;
    FE_Quit ();
    state->fewasinit = 0;
    Py_RETURN_NONE;
}

static PyObject*
_sdl_feventpump (PyObject *self)
{
    ASSERT_FASTEVENT_INIT (NULL);
    FE_PumpEvents ();
    Py_RETURN_NONE;
}

static PyObject*
_sdl_feventpoll (PyObject *self)
{
    SDL_Event event;
    int status;

    ASSERT_FASTEVENT_INIT (NULL);

    status = FE_PollEvent (&event);
    if (status == 1)
        return PyEvent_New (&event);
    else
    {
        /* Check for -1 */
        Py_RETURN_NONE;
    }

}
static PyObject*
_sdl_feventwait (PyObject *self)
{
    SDL_Event event;
    int status;

    ASSERT_FASTEVENT_INIT (NULL);

    Py_BEGIN_ALLOW_THREADS;
    status = FE_WaitEvent (&event);
    Py_END_ALLOW_THREADS;

    /* FE_WaitEvent will block forever on error */
    if (!status)
    {
        PyErr_SetString (PyExc_PyGameError,
            "unexpected error in FE_WaitEvent");
        return NULL;
    }
    return PyEvent_New (&event);
}

static PyObject*
_sdl_feventpush (PyObject *self, PyObject *args)
{
    PyObject *ev;
    SDL_Event event;
    int status;

    ASSERT_FASTEVENT_INIT (NULL);

    if (!PyArg_ParseTuple (args, "O:push", &ev))
        return NULL;
    if (!PyEvent_Check (ev))
    {
        PyErr_SetString (PyExc_TypeError, "event must be an Event");
        return NULL;
    }

    if (!PyEvent_SDLEventFromEvent (ev, &event))
        return NULL;

    Py_BEGIN_ALLOW_THREADS;
    status = FE_PushEvent (&event);
    Py_END_ALLOW_THREADS;

    if (status != 1)
    {
        PyErr_SetString (PyExc_PyGameError, "unexpected error in FE_PushEvent");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject*
_sdl_feventget (PyObject *self)
{
    SDL_Event event;
    PyObject *list, *ev;
    int status;

    ASSERT_FASTEVENT_INIT (NULL);

    list = PyList_New (0);
    if (!list)
        return NULL;

    FE_PumpEvents ();

    while (1)
    {
        status = FE_PollEvent (&event);
        if (status != 1)
            break;
        ev = PyEvent_New (&event);
        if (!ev)
        {
            Py_DECREF (list);
            return NULL;
        }

        if (PyList_Append (list, ev) == -1)
        {
            Py_DECREF (list);
            Py_DECREF (ev);
            return NULL;
        }
        Py_DECREF (ev);
    }

    return list;
}

static int
_fastevent_clear (PyObject *mod)
{
    _FastEventState *state = FASTEVENT_MOD_STATE (mod);
    if (!state->fewasinit)
        return 0;
    FE_Quit ();
    state->fewasinit = 0;
    return 0;
}

#ifdef IS_PYTHON_3
struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    "fastevent",
    DOC_FASTEVENT,
    sizeof (_FastEventState),
    _fevent_methods,
    NULL,
    NULL,
    _fastevent_clear,
    NULL
};
#endif

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_fastevent (void)
#else
PyMODINIT_FUNC initfastevent (void)
#endif
{
    PyObject *mod, *eventmod, *dict;
    _FastEventState *state;

#ifdef IS_PYTHON_3
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("fastevent", _fevent_methods, DOC_FASTEVENT);
#endif
    if (!mod)
        goto fail;
    state = FASTEVENT_MOD_STATE (mod);
    state->fewasinit = 0;

    dict = PyModule_GetDict (mod);

    if (import_pygame2_base () < 0)
        goto fail;
    if (import_pygame2_sdl_base () < 0)
        goto fail;
    if (import_pygame2_sdl_event () < 0)
        goto fail;
    if (import_pygame2_sdl_video () < 0)
        goto fail;
    
    eventmod = PyImport_ImportModule ("pygame2.sdl.event");
    if (eventmod)
    {
        char *NAMES[] = { "Event", NULL };
        int  i;

        for (i = 0; NAMES[i]; i++)
        {
            PyObject *ref = PyObject_GetAttrString (eventmod, NAMES[i]);
            if (ref)
            {
                PyDict_SetItemString (dict, NAMES[i], ref);
                Py_DECREF (ref);
            }
            else
                PyErr_Clear ();
        }
    }

    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
