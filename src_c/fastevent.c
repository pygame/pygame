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

/* DOC */ static char doc_init[] =
    /* DOC */
    "pygame.fastevent.init() -> None\n"
    /* DOC */ "initialize pygame.fastevent.\n"
    /* DOC */;
static PyObject *
fastevent_init(PyObject *self)
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

/* DOC */ static char doc_pump[] =
    /* DOC */
    "pygame.fastevent.pump() -> None\n"
    /* DOC */
    "update the internal messages\n"
    /* DOC */
    "\n"
    /* DOC */
    "For each frame of your game, you will need to make some sort\n"
    /* DOC */
    "of call to the event queue. This ensures your program can internally\n"
    /* DOC */
    "interact with the rest of the operating system. If you are not using\n"
    /* DOC */
    "other event functions in your game, you should call pump() to allow\n"
    /* DOC */
    "pygame to handle internal actions.\n"
    /* DOC */
    "\n"
    /* DOC */
    "There are important things that must be dealt with internally in the\n"
    /* DOC */
    "event queue. The main window may need to be repainted. Certain "
    "joysticks\n"
    /* DOC */
    "must be polled for their values. If you fail to make a call to the "
    "event\n"
    /* DOC */
    "queue for too long, the system may decide your program has locked up.\n"
    /* DOC */;
static PyObject *
fastevent_pump(PyObject *self)
{
    FE_INIT_CHECK();
    FE_PumpEvents();
    Py_RETURN_NONE;
}

/* DOC */ static char doc_wait[] =
    /* DOC */
    "pygame.fastevent.wait() -> Event\n"
    /* DOC */
    "wait for an event\n"
    /* DOC */
    "\n"
    /* DOC */
    "Returns the current event on the queue. If there are no messages\n"
    /* DOC */
    "waiting on the queue, this will not return until one is\n"
    /* DOC */
    "available. Sometimes it is important to use this wait to get\n"
    /* DOC */
    "events from the queue, it will allow your application to idle\n"
    /* DOC */ "when the user isn't doing anything with it.\n"
    /* DOC */;
static PyObject *
fastevent_wait(PyObject *self)
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

/* DOC */ static char doc_poll[] =
    /* DOC */
    "pygame.fastevent.poll() -> Event\n"
    /* DOC */
    "get an available event\n"
    /* DOC */
    "\n"
    /* DOC */
    "Returns next event on queue. If there is no event waiting on the\n"
    /* DOC */ "queue, this will return an event with type NOEVENT.\n"
    /* DOC */;
static PyObject *
fastevent_poll(PyObject *self)
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

/* DOC */ static char doc_get[] =
    /* DOC */
    "pygame.fastevent.get() -> list of Events\n"
    /* DOC */ "get all events from the queue\n"
    /* DOC */;
static PyObject *
fastevent_get(PyObject *self)
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

        PyList_Append(list, e);
        Py_DECREF(e);
    }

    return list;
}

/*DOC*/ static char doc_post[] =
    /*DOC*/
    "pygame.fastevent.post(Event) -> None\n"
    /*DOC*/
    "place an event on the queue\n"
    /*DOC*/
    "\n"
    /*DOC*/
    "This will post your own event objects onto the event queue.\n"
    /*DOC*/
    "You can past any event type you want, but some care must be\n"
    /*DOC*/
    "taken. For example, if you post a MOUSEBUTTONDOWN event to the\n"
    /*DOC*/
    "queue, it is likely any code receiving the event will expect\n"
    /*DOC*/
    "the standard MOUSEBUTTONDOWN attributes to be available, like\n"
    /*DOC*/
    "'pos' and 'button'.\n"
    /*DOC*/
    "\n"
    /*DOC*/
    "Because pygame.fastevent.post() may have to wait for the queue\n"
    /*DOC*/
    "to empty, you can get into a dead lock if you try to append an\n"
    /*DOC*/
    "event on to a full queue from the thread that processes events.\n"
    /*DOC*/
    "For that reason I do not recommend using this function in the\n"
    /*DOC*/ "main thread of an SDL program.\n"
    /*DOC*/;
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
    {"init", (PyCFunction)fastevent_init, METH_NOARGS, doc_init},
    {"get", (PyCFunction)fastevent_get, METH_NOARGS, doc_get},
    {"pump", (PyCFunction)fastevent_pump, METH_NOARGS, doc_pump},
    {"wait", (PyCFunction)fastevent_wait, METH_NOARGS, doc_wait},
    {"poll", (PyCFunction)fastevent_poll, METH_NOARGS, doc_poll},
    {"post", fastevent_post, METH_O, doc_post},

    {NULL, NULL, 0, NULL}};

/*DOC*/ static char doc_fastevent_MODULE[] =
    /*DOC*/
    "pygame.fastevent is a wrapper for Bob Pendleton's fastevent\n"
    /*DOC*/
    "library.  It provides fast events for use in multithreaded\n"
    /*DOC*/
    "environments.  When using pygame.fastevent, you can not use\n"
    /*DOC*/
    "any of the pump, wait, poll, post, get, peek, etc. functions\n"
    /*DOC*/ "from pygame.event, but you should use the Event objects.\n"
    /*DOC*/;
MODINIT_DEFINE(fastevent)
{
    PyObject *module, *eventmodule, *dict;
    int ecode;

#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "fastevent",
                                         doc_fastevent_MODULE,
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
                            doc_fastevent_MODULE);
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
