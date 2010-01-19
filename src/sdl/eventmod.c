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
#define PYGAME_SDLEVENT_INTERNAL

#include "eventmod.h"
#include "pgsdl.h"
#include "sdlevent_doc.h"

static int _event_traverse (PyObject *mod, visitproc visit, void *arg);
static int _event_clear (PyObject *mod);

static int _sdl_filter_events (const SDL_Event *event);

static PyObject* _sdl_eventpump (PyObject *self);
static PyObject* _sdl_eventpoll (PyObject *self);
static PyObject* _sdl_eventwait (PyObject *self);
static PyObject* _sdl_eventpush (PyObject *self, PyObject *args);
static PyObject* _sdl_eventstate (PyObject *self, PyObject *args);
static PyObject* _sdl_eventpeep (PyObject *self, PyObject *args);
static PyObject* _sdl_eventclear (PyObject *self, PyObject *args);
static PyObject* _sdl_eventget (PyObject *self, PyObject *args);
static PyObject* _sdl_eventpeek (PyObject *self, PyObject *args);
static PyObject* _sdl_eventsetblocked (PyObject *self, PyObject *args);
static PyObject* _sdl_eventgetblocked (PyObject *self);
static PyObject* _sdl_eventsetfilter (PyObject *self, PyObject *args);
static PyObject* _sdl_eventgetfilter (PyObject *self);
static PyObject* _sdl_eventgetappstate (PyObject *self);

static PyMethodDef _event_methods[] = {
    { "pump", (PyCFunction)_sdl_eventpump, METH_NOARGS, DOC_EVENT_PUMP },
    { "poll", (PyCFunction)_sdl_eventpoll, METH_NOARGS, DOC_EVENT_POLL },
    { "wait", (PyCFunction)_sdl_eventwait, METH_NOARGS, DOC_EVENT_WAIT },
    { "push", _sdl_eventpush, METH_O, DOC_EVENT_PUSH },
    { "state", _sdl_eventstate, METH_VARARGS, DOC_EVENT_STATE },
    { "peep", _sdl_eventpeep, METH_VARARGS, DOC_EVENT_PEEP },
    { "clear", _sdl_eventclear, METH_VARARGS, DOC_EVENT_CLEAR },
    { "get", _sdl_eventget, METH_VARARGS, DOC_EVENT_GET },
    { "peek", _sdl_eventpeek, METH_VARARGS, DOC_EVENT_PEEK },
    { "set_blocked", _sdl_eventsetblocked, METH_O, DOC_EVENT_SET_BLOCKED },
    { "get_blocked", (PyCFunction)_sdl_eventgetblocked, METH_NOARGS,
      DOC_EVENT_GET_BLOCKED },
    { "set_filter", _sdl_eventsetfilter, METH_O, DOC_EVENT_SET_FILTER },
    { "get_filter", (PyCFunction)_sdl_eventgetfilter, METH_NOARGS,
      DOC_EVENT_GET_FILTER },
    { "get_app_state", (PyCFunction) _sdl_eventgetappstate, METH_NOARGS,
      DOC_EVENT_GET_APP_STATE },
    { NULL, NULL, 0, NULL }
};

static int
_sdl_filter_events (const SDL_Event *event)
{
    PyObject *result, *ev;
    int retval;

    if (!SDLEVENT_STATE->filterhook)
        return 1;
    
    ev = PyEvent_New ((SDL_Event*)event);
    if (!ev)
        return 0;

    result = PyObject_CallObject (SDLEVENT_STATE->filterhook, ev);
    Py_DECREF (ev);

    retval = PyObject_IsTrue (result);
    if (retval == -1)
    {
        /* Errors are considered false */
        PyErr_Clear ();
        retval = 0;
    }

    if (!retval)
    {
        if (event->type >= SDL_USEREVENT && event->type < SDL_NUMEVENTS &&
            event->user.code == PYGAME_USEREVENT_CODE &&
            event->user.data1 == (void*)PYGAME_USEREVENT)
        {
            Py_DECREF ((PyObject*) event->user.data2);
        }
    }

    Py_XDECREF (result);
    return retval;
}

static PyObject*
_sdl_eventpump (PyObject *self)
{
    ASSERT_VIDEO_INIT(NULL);
    SDL_PumpEvents ();
    Py_RETURN_NONE;
}

static PyObject*
_sdl_eventpoll (PyObject *self)
{
    SDL_Event event;
    
    ASSERT_VIDEO_INIT(NULL);
    if (SDL_PollEvent (&event))
        return PyEvent_NewInternal (&event, 1);
    Py_RETURN_NONE;
}

static PyObject*
_sdl_eventwait (PyObject *self)
{
    SDL_Event event;
    int _stat;
    
    ASSERT_VIDEO_INIT(NULL);
    
    Py_BEGIN_ALLOW_THREADS;
    _stat = SDL_WaitEvent (&event);
    Py_END_ALLOW_THREADS;
    
    if (_stat)
        return PyEvent_NewInternal (&event, 1);
    
    PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
    return NULL;
}

static PyObject*
_sdl_eventpush (PyObject *self, PyObject *args)
{
    SDL_Event event;
    
    ASSERT_VIDEO_INIT(NULL);

    if (!PyEvent_Check (args))
    {
        PyErr_SetString (PyExc_TypeError, "event must be an Event");
        return NULL;
    }
    
    if (!PyEvent_SDLEventFromEvent (args, &event))
        return NULL;

    if (SDL_EventState (event.type, SDL_QUERY) == SDL_IGNORE)
        Py_RETURN_NONE; /* Event is blocked, do not post it. */

    if (SDL_PushEvent (&event) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_sdl_eventstate (PyObject *self, PyObject *args)
{
    Uint8 type;
    int state;

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple (args, "ii:state", &type, &state))
        return NULL;
    return PyInt_FromLong (SDL_EventState (type, state));
}

static PyObject*
_sdl_eventpeep (PyObject *self, PyObject *args)
{
    int count;
    SDL_eventaction action;
    Uint32 mask;
    SDL_Event *events;
    PyObject *list = NULL;

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple (args, "iil|O:peep", &count, &action, &mask, &list))
        return NULL;

    if (list && !PySequence_Check (list))
    {
        PyErr_SetString (PyExc_TypeError, "events must be a sequence");
        return NULL;
    }
    
    if (count < 0)
    {
        PyErr_SetString (PyExc_ValueError, "count must not be negative");
        return NULL;
    }
    if (count == 0)
        return PyInt_FromLong (0);


    switch (action)
    {
    case SDL_ADDEVENT:
    {
        Py_ssize_t i, itemcount;
        PyObject *ev;
        if (!list)
        {
            PyErr_SetString (PyExc_PyGameError,"event sequence must be passed");
            return NULL;
        }
        itemcount = PySequence_Size (list);
        if (itemcount == 0)
            return PyInt_FromLong (0);
        itemcount = MIN (itemcount, (Py_ssize_t) count);

        events = PyMem_New (SDL_Event, (size_t) itemcount);
        if (!events)
            return NULL;

        for (i = 0; i < itemcount; i++)
        {
            ev = PySequence_ITEM (list, i);
            if (!ev)
            {
                PyMem_Free (events);
                return NULL;
            }
            if (!PyEvent_Check (ev))
            {
                PyMem_Free (events);
                Py_DECREF (ev);
                PyErr_SetString (PyExc_TypeError,
                    "event sequence must contain Event objects only");
                return NULL;
            }
            if (!PyEvent_SDLEventFromEvent (ev, &(events[i])))
            {
                PyMem_Free (events);
                Py_DECREF (ev);
                return NULL;
            }
            Py_DECREF (ev);
        }
        count = SDL_PeepEvents (events, (int)itemcount, action, mask);
        PyMem_Free (events);
        if (count == -1)
        {
            PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
            return NULL;
        }
        return PyInt_FromLong (count);
    }
    case SDL_GETEVENT:
    case SDL_PEEKEVENT:
    {
        int i;
        int release = (action == SDL_PEEKEVENT) ? 0 : 1;
        PyObject *ev;

        events = PyMem_New (SDL_Event, (size_t) count);
        if (!events)
            return NULL;

        count = SDL_PeepEvents (events, count, action, mask);
        if (count == -1)
        {
            PyMem_Free (events);
            PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
            return NULL;
        }
        
        list = PyList_New (count);
        if (!list)
        {
            PyMem_Free (events);
            return NULL;
        }

        for (i = 0; i < count; i++)
        {
            ev = PyEvent_NewInternal (&(events[i]), release);
            if (!ev)
            {
                Py_DECREF (list);
                PyMem_Free (events);
                return NULL;
            }
            PyList_SetItem (list, (Py_ssize_t)i, ev);
        }
        PyMem_Free (events);
        return list;
    }
    default:
        PyErr_SetString (PyExc_ValueError, "invalid action");
        return NULL;
    }
}

static PyObject*
_sdl_eventclear (PyObject *self, PyObject *args)
{
    PyObject *events = NULL;
    SDL_Event event;
    Uint32 mask = 0, mm;
    Py_ssize_t count, i;

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple (args, "|O:clear", &events))
        return NULL;
    if (!events)
    {
        /* Anything should be cleared */
        mask = SDL_ALLEVENTS;
    }
    else if (Uint32FromObj (events, &mm))
    {
        /* Single event */
        mask = SDL_EVENTMASK (mm);
    }
    else if (PySequence_Check (events))
    {
        /* List of events */
        count = PySequence_Size (events);
        if (count == 0)
            Py_RETURN_NONE;
        for (i = 0; i < count; i++)
        {
            if (!Uint32FromSeqIndex (events, i, &mm))
            {
                PyErr_SetString (PyExc_TypeError,
                    "mask sequence must consist of valid event types");
                return NULL;
            }
            mask |= SDL_EVENTMASK (mm);
        }
    }
    else
    {
        PyErr_SetString (PyExc_TypeError, "invalid events argument");
        return NULL;
    }

    if (mask)
    {
        SDL_PumpEvents ();
        while (SDL_PeepEvents (&event, 1, SDL_GETEVENT, mask) == 1)
        {
            /* Release any memory hold by the Python system. */
            if (event.type >= SDL_USEREVENT && event.type < SDL_NUMEVENTS &&
                event.user.code == PYGAME_USEREVENT_CODE &&
                event.user.data1 == (void*)PYGAME_USEREVENT)
            {
                Py_DECREF ((PyObject*) event.user.data2);
            }
        }
    }
    Py_RETURN_NONE;
}

static PyObject*
_sdl_eventget (PyObject *self, PyObject *args)
{
    PyObject *events = NULL;
    PyObject *list;
    SDL_Event event;
    Uint32 mask = 0, mm;
    Py_ssize_t count, i;

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple (args, "|O:get", &events))
        return NULL;
    if (!events)
    {
        /* Anything should be fetched. */
        mask = SDL_ALLEVENTS;
    }
    else if (Uint32FromObj (events, &mm))
    {
        /* Single event */
        mask = SDL_EVENTMASK (mm);
    }
    else if (PySequence_Check (events))
    {
        /* List of events */
        count = PySequence_Size (events);
        if (count == 0)
            Py_RETURN_NONE;
        for (i = 0; i < count; i++)
        {
            if (!Uint32FromSeqIndex (events, i, &mm))
            {
                PyErr_SetString (PyExc_TypeError,
                    "mask sequence must consist of valid event types");
                return NULL;
            }
            mask |= SDL_EVENTMASK (mm);
        }
    }
    else
    {
        PyErr_SetString (PyExc_TypeError, "invalid events argument");
        return NULL;
    }

    if (mask)
    {
        PyObject *e;

        list = PyList_New (0);
        if (!list)
            return NULL;

        SDL_PumpEvents ();
        while (SDL_PeepEvents (&event, 1, SDL_GETEVENT, mask) == 1)
        {
            e = PyEvent_NewInternal (&event, 1);
            if (!e)
            {
                Py_DECREF (list);
                return NULL;
            }
            if (PyList_Append (list, e) == -1)
            {
                Py_DECREF (e);
                Py_DECREF (list);
                return NULL;
            }
            Py_DECREF (e);
        }
        return list;
    }
    Py_RETURN_NONE;
}

static PyObject*
_sdl_eventpeek (PyObject *self, PyObject *args)
{
    PyObject *events = NULL;
    SDL_Event event;
    Uint32 mask = 0, mm;
    Py_ssize_t count, i;

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple (args, "|O:peek", &events))
        return NULL;
    if (!events)
    {
        /* Anything should be peeked. */
        mask = SDL_ALLEVENTS;
    }
    else if (Uint32FromObj (events, &mm))
    {
        /* Single event */
        mask = SDL_EVENTMASK (mm);
    }
    else if (PySequence_Check (events))
    {
        /* List of events */
        count = PySequence_Size (events);
        if (count == 0)
            Py_RETURN_NONE;
        for (i = 0; i < count; i++)
        {
            if (!Uint32FromSeqIndex (events, i, &mm))
            {
                PyErr_SetString (PyExc_TypeError,
                    "mask sequence must consist of valid event types");
                return NULL;
            }
            mask |= SDL_EVENTMASK (mm);
        }
    }
    else
    {
        PyErr_SetString (PyExc_TypeError, "invalid events argument");
        return NULL;
    }

    if (mask)
    {
        SDL_PumpEvents ();
        if (SDL_PeepEvents (&event, 1, SDL_PEEKEVENT, mask) == 1)
            Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject*
_sdl_eventsetblocked (PyObject *self, PyObject *args)
{
    Uint8 types[SDL_NUMEVENTS] = { 0 };
    Uint8 type;
    int i;
 
    if (args == Py_None)
    {
        /* Reset the previous state, allow all events. */
        for (i = 0; i < SDL_NUMEVENTS; i++)
            SDL_EventState ((Uint8)i, SDL_ENABLE);
    }
    else if (Uint8FromObj (args, &type))
    {
        /* Reset the previous state, allow all events. */
        for (i = 0; i < SDL_NUMEVENTS; i++)
            SDL_EventState ((Uint8)i, SDL_ENABLE);

        /* Single event to be blocked */
        SDL_EventState (type, SDL_IGNORE);
    }
    else if (PySequence_Check (args))
    {
        Py_ssize_t count, j;

        PyErr_Clear (); /* from Uint8FromObj */

        /* List of events */
        count = PySequence_Size (args);
        if (count > SDL_NUMEVENTS)
        {
            PyErr_SetString (PyExc_ValueError,
                "event sequence exceeds amount of available events");
            return NULL;
        }
        if (count == 0)
            Py_RETURN_NONE;
        
        for (j = 0, i = 0; j < count; j++, i++)
        {
            if (!Uint8FromSeqIndex (args, j, &(types[i])))
            {
                PyErr_SetString (PyExc_TypeError,
                    "event sequence must consist of valid event types");
                return NULL;
            }
        }

        /* Reset the previous state, allow all events. */
        for (i = 0; i < SDL_NUMEVENTS; i++)
            SDL_EventState ((Uint8)i , SDL_ENABLE);
        for (i = 0; i < (int)count; i++)
            SDL_EventState (types[i], SDL_IGNORE);
    }
    else
    {
        PyErr_SetString (PyExc_TypeError, "invalid events argument");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_sdl_eventgetblocked (PyObject *self)
{
    PyObject *list, *val;
    int i;
    
    list = PyList_New (0);
    if (!list)
        return NULL;

    for (i = 0; i < SDL_NUMEVENTS; i++)
    {
        if (SDL_EventState ((Uint8)i, SDL_QUERY) != SDL_IGNORE)
            continue;
        val = PyInt_FromLong (i);
        if (PyList_Append (list, val) == -1)
        {
            Py_DECREF (list);
            Py_DECREF (val);
            return NULL;
        }
        Py_DECREF (val);
    }
    return list;
}

static PyObject*
_sdl_eventsetfilter (PyObject *self, PyObject *args)
{
    _SDLEventState *state = SDLEVENT_MOD_STATE(self);

    ASSERT_VIDEO_INIT(NULL);

    if (args == Py_None)
    {
        /* Reset the filter hook */
        Py_XDECREF (state->filterhook);
        SDL_SetEventFilter (NULL);
        state->filterhook = NULL;
        Py_RETURN_NONE;
    }

    if (!PyCallable_Check (args))
    {
        PyErr_SetString (PyExc_TypeError, "hook must be callable");
        return NULL;
    }

    Py_INCREF (args);
    state->filterhook = args;
    SDL_SetEventFilter (_sdl_filter_events);

    Py_RETURN_NONE;
}

static PyObject*
_sdl_eventgetfilter (PyObject *self)
{
    _SDLEventState *state = SDLEVENT_MOD_STATE(self);

    ASSERT_VIDEO_INIT(NULL);

    if (!state->filterhook)
        Py_RETURN_NONE;
    Py_INCREF (state->filterhook);
    return state->filterhook;
}

static PyObject*
_sdl_eventgetappstate (PyObject *self)
{
    ASSERT_VIDEO_INIT(NULL);
    return PyInt_FromLong (SDL_GetAppState ());
}

static int
_event_traverse (PyObject *mod, visitproc visit, void *arg)
{
    Py_VISIT (SDLEVENT_MOD_STATE(mod)->filterhook);
    return 0;
}

static int
_event_clear (PyObject *mod)
{
    Py_CLEAR (SDLEVENT_MOD_STATE(mod)->filterhook);
    return 0;
}

#ifdef IS_PYTHON_3
struct PyModuleDef _eventmodule = {
    PyModuleDef_HEAD_INIT,
    "event",
    DOC_EVENT,
    sizeof(_SDLEventState),
    _event_methods,
    NULL,
    _event_traverse,
    _event_clear,
    NULL
};
#else
_SDLEventState _modstate;
#endif

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_event (void)
#else
PyMODINIT_FUNC initevent (void)
#endif
{
    PyObject *mod = NULL;
    PyObject *c_api_obj;
    _SDLEventState *state;
    static void *c_api[PYGAME_SDLEVENT_SLOTS];

    /* Complete types */
    if (PyType_Ready (&PyEvent_Type) < 0)
        goto fail;
    Py_INCREF (&PyEvent_Type);

#ifdef IS_PYTHON_3
    mod = PyModule_Create (&_eventmodule);
#else
    mod = Py_InitModule3 ("event", _event_methods, DOC_EVENT);
#endif
    if (!mod)
        goto fail;
    state = SDLEVENT_MOD_STATE(mod);
    state->filterhook = NULL;

    PyModule_AddObject (mod, "Event", (PyObject *) &PyEvent_Type);

    event_export_capi (c_api);

    c_api_obj = PyCObject_FromVoidPtr ((void *) c_api, NULL);
    if (c_api_obj)
        PyModule_AddObject (mod, PYGAME_SDLEVENT_ENTRY, c_api_obj);    

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
