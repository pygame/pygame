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
 *  pygame fastevent module
 */
#define PYGAMEAPI_FASTEVENT_INTERNAL
#include "pygame.h"

#include "pgcompat.h"

#include "doc/fastevent_doc.h"

#include "fastevents.h"

static int FE_WasInit = 0;

#define FE_INIT_CHECK()                                                       \
    do {                                                                      \
        if (!FE_WasInit)                                                      \
            return RAISE(pgExc_SDLError, "fastevent system not initialized"); \
    } while (0)

static void
fastevent_cleanup(void)
{
    if (FE_WasInit) {
        FE_Quit();
        FE_WasInit = 0;
    }
}

/* fastevent module functions */
static PyObject *
fastevent_init(PyObject *self, PyObject *args)
{
    VIDEO_INIT_CHECK();

#ifndef WITH_THREAD
    return RAISE(pgExc_SDLError,
                 "pygame.fastevent requires a threaded Python");
#else
    if (!FE_WasInit) {
        if (FE_Init() == -1)
            return RAISE(pgExc_SDLError, FE_GetError());

        pg_RegisterQuit(fastevent_cleanup);
        FE_WasInit = 1;
    }

    Py_RETURN_NONE;
#endif /* WITH_THREAD */
}

static PyObject *
fastevent_get_init(PyObject *self, PyObject *args)
{
    return PyBool_FromLong(FE_WasInit);
}

static PyObject *
fastevent_pump(PyObject *self, PyObject *args)
{
    FE_INIT_CHECK();
    FE_PumpEvents();
    Py_RETURN_NONE;
}

static PyObject *
fastevent_wait(PyObject *self, PyObject *args)
{
    SDL_Event event;
    int status;

    FE_INIT_CHECK();

    Py_BEGIN_ALLOW_THREADS;
    status = FE_WaitEvent(&event);
    Py_END_ALLOW_THREADS;

    /* FE_WaitEvent will block forever on error */
    if (!status)
        return RAISE(pgExc_SDLError, "unexpected error in FE_WaitEvent!");

    return pgEvent_New(&event);
}

static PyObject *
fastevent_poll(PyObject *self, PyObject *args)
{
    SDL_Event event;
    int status;

    FE_INIT_CHECK();

    status = FE_PollEvent(&event);
    if (status == 1)
        return pgEvent_New(&event);
    else {
        /* Check for -1 */
        return pgEvent_New(NULL);
    }
}

static PyObject *
fastevent_get(PyObject *self, PyObject *args)
{
    SDL_Event event;
    PyObject *list, *e;
    int status;

    FE_INIT_CHECK();

    list = PyList_New(0);
    if (!list)
        return NULL;

    FE_PumpEvents();

    while (1) {
        status = FE_PollEvent(&event);
        if (status != 1)
            break;
        e = pgEvent_New(&event);
        if (!e) {
            Py_DECREF(list);
            return NULL;
        }

        if (0 != PyList_Append(list, e)) {
            Py_DECREF(list);
            Py_DECREF(e);
            return NULL; /* Exception already set. */
        }
        Py_DECREF(e);
    }

    return list;
}

static PyObject *
fastevent_post(PyObject *self, PyObject *arg)
{
    SDL_Event event;
    int status;

    if (!PyObject_IsInstance(arg, (PyObject *)&pgEvent_Type)) {
        PyErr_Format(PyExc_TypeError, "argument 1 must be %s, not %s",
                     pgEvent_Type.tp_name, Py_TYPE(arg)->tp_name);
        return NULL;
    }

    FE_INIT_CHECK();

    if (pgEvent_FillUserEvent((pgEventObject *)arg, &event))
        return NULL;

    Py_BEGIN_ALLOW_THREADS;
    status = FE_PushEvent(&event);
    Py_END_ALLOW_THREADS;

    if (status != 1)
        return RAISE(pgExc_SDLError, "Unexpected error in FE_PushEvent");

    Py_RETURN_NONE;
}

static PyMethodDef _fastevent_methods[] = {
    {"init", fastevent_init, METH_NOARGS, DOC_PYGAMEFASTEVENTINIT},
    {"get_init", fastevent_get_init, METH_NOARGS, DOC_PYGAMEFASTEVENTGETINIT},
    {"get", fastevent_get, METH_NOARGS, DOC_PYGAMEFASTEVENTGET},
    {"pump", fastevent_pump, METH_NOARGS, DOC_PYGAMEFASTEVENTPUMP},
    {"wait", fastevent_wait, METH_NOARGS, DOC_PYGAMEFASTEVENTWAIT},
    {"poll", fastevent_poll, METH_NOARGS, DOC_PYGAMEFASTEVENTPOLL},
    {"post", fastevent_post, METH_O, DOC_PYGAMEFASTEVENTPOST},

    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(fastevent)
{
    PyObject *module, *eventmodule, *dict;
    int ecode;

#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "fastevent",
                                         DOC_PYGAMEFASTEVENT,
                                         -1,
                                         _fastevent_methods,
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
    import_pygame_event();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    /* create the module */
#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3(MODPREFIX "fastevent", _fastevent_methods,
                            DOC_PYGAMEFASTEVENT);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    dict = PyModule_GetDict(module);

    /* add the event module functions if available */
    eventmodule = PyImport_ImportModule(IMPPREFIX "event");
    if (eventmodule) {
        char *NAMES[] = {"Event", "event_name", NULL};
        int i;

        for (i = 0; NAMES[i]; i++) {
            PyObject *ref = PyObject_GetAttrString(eventmodule, NAMES[i]);
            if (ref) {
                ecode = PyDict_SetItemString(dict, NAMES[i], ref);
                Py_DECREF(ref);
                if (ecode == -1) {
                    DECREF_MOD(module);
                    MODINIT_ERROR;
                }
            }
            else
                PyErr_Clear();
        }
    }
    else {
        PyErr_Clear();
    }
    MODINIT_RETURN(module);
}
