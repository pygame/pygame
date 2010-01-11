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
#define PYGAME_SDLTIME_INTERNAL

#include "pgsdl.h"
#include "sdltime_doc.h"

typedef struct
{
    SDL_TimerID    id;
    PyObject      *callable;
    PyObject      *param;
#ifdef WITH_THREAD
    PyThreadState *thread;
#endif
} _TimerData;

typedef struct {
    PyObject *timerhook;
    PyObject *timerlist;
} _SDLTimerState;

#ifdef IS_PYTHON_3
struct PyModuleDef _timermodule; /* Forward declaration */
#define SDLTIMER_MOD_STATE(mod) ((_SDLTimerState*)PyModule_GetState(mod))
#define SDLTIMER_STATE SDLTIMER_MOD_STATE(PyState_FindModule(&_timermodule))
#else
_SDLTimerState _modstate;
#define SDLTIMER_MOD_STATE(mod) (&_modstate)
#define SDLTIMER_STATE SDLTIMER_MOD_STATE(NULL)
#endif

static int _timer_traverse (PyObject *mod, visitproc visit, void *arg);
static int _timer_clear (PyObject *mod);

static Uint32 _sdl_timercallback (Uint32 interval);
static Uint32 _sdl_timerfunc (Uint32 interval, void *param);
static void _free_timerdata (_TimerData *data);
static void _remove_alltimers (_SDLTimerState *mod);

static PyObject* _sdl_timeinit (PyObject *self);
static PyObject* _sdl_timewasinit (PyObject *self);
static PyObject* _sdl_timequit (PyObject *self);
static PyObject* _sdl_timegetticks (PyObject *self);
static PyObject* _sdl_timedelay (PyObject *self, PyObject *args);
static PyObject* _sdl_settimer (PyObject *self, PyObject *args);
static PyObject* _sdl_addtimer (PyObject *self, PyObject *args);
static PyObject* _sdl_removetimer (PyObject *self, PyObject *args);

static PyMethodDef _time_methods[] = {
    { "init", (PyCFunction) _sdl_timeinit, METH_NOARGS, DOC_TIME_INIT },
    { "was_init", (PyCFunction) _sdl_timewasinit, METH_NOARGS,
      DOC_TIME_WAS_INIT },
    { "quit", (PyCFunction) _sdl_timequit, METH_NOARGS, DOC_TIME_QUIT },
    { "get_ticks", (PyCFunction) _sdl_timegetticks, METH_NOARGS,
      DOC_TIME_GET_TICKS },
    { "delay", _sdl_timedelay, METH_VARARGS, DOC_TIME_DELAY },
    { "set_timer", _sdl_settimer, METH_VARARGS, DOC_TIME_SET_TIMER },
    { "add_timer", _sdl_addtimer, METH_VARARGS, DOC_TIME_ADD_TIMER },
    { "remove_timer", _sdl_removetimer, METH_VARARGS, DOC_TIME_REMOVE_TIMER },
    { NULL, NULL, 0, NULL }
};

static Uint32
_sdl_timercallback (Uint32 interval)
{
    PyObject *result, *val;
    Uint32 retval;
    _SDLTimerState *state = SDLTIMER_STATE;

    if (!state->timerhook)
        return 1;
    
    val = PyLong_FromUnsignedLong (interval);
    result = PyObject_CallObject (state->timerhook, val);
    Py_DECREF (val);

    if (!Uint32FromObj (result, &retval))
    {
        /* Wrong signature, remove the callback */
        PyErr_SetString (PyExc_ValueError,
            "callback must return a positive integer");
        Py_XDECREF (result);
        Py_XDECREF (state->timerhook);
        state->timerhook = NULL;
        SDL_SetTimer (0, NULL);
        return 0;
    }

    Py_XDECREF (result);
    return retval;
}

static Uint32
_sdl_timerfunc (Uint32 interval, void *param)
{
    _TimerData *timerdata = (_TimerData*) param;
    PyObject *result, *val;
    Uint32 retval = 0;
    _SDLTimerState *state;

#ifdef WITH_THREAD
    PyThreadState* oldstate;
    PyEval_AcquireLock ();
    oldstate = PyThreadState_Swap (timerdata->thread);
#endif
    state = SDLTIMER_STATE;
    
    val = PyLong_FromUnsignedLong (interval);
    if (timerdata->param)
        result = PyObject_CallObject (timerdata->callable, timerdata->param);
    else
        result = PyObject_CallObject (timerdata->callable, NULL);
    
    Py_DECREF (val);
    if (!result)
    {
        PyErr_Warn (PyExc_RuntimeError, "timer callback failed:"); 
        PyErr_Print ();
        PyErr_Clear ();
        return 0;
    }
    
    if (!Uint32FromObj (result, &retval))
    {
        /* Wrong signature, ignore the callback */
        PyErr_Clear ();
        Py_XDECREF (result);
        PyErr_Warn (PyExc_ValueError,
            "timer callback return value must be a positive integer"); 
        retval = 0;
    }
    else
    {
        Py_XDECREF (result);
    }

#ifdef WITH_THREAD
    PyThreadState_Swap (oldstate);
    PyEval_ReleaseLock ();
#endif
    return retval;
}

static void
_free_timerdata (_TimerData *data)
{
    if (!data)
        return;
    
    if (data->id)
        SDL_RemoveTimer (data->id);
    data->id = NULL;

    Py_XDECREF (data->callable);
    Py_XDECREF (data->param);
#ifdef WITH_THREAD
    PyThreadState_Clear (data->thread);
    PyThreadState_Delete (data->thread);
#endif
    PyMem_Free (data);
}

static void
_remove_alltimers (_SDLTimerState *state)
{
    Py_ssize_t pos, count;
    PyObject *val;
    _TimerData *timerdata;
    
    if (!state || state->timerlist == NULL)
        return;
    
    /* Clean up all timers */
    count = PyList_GET_SIZE (state->timerlist);
    for (pos = 0; pos < count; pos++)
    {
        val = PyList_GET_ITEM (state->timerlist, pos);
        timerdata = (_TimerData*) PyCObject_AsVoidPtr (val);
        _free_timerdata (timerdata);
    }
    Py_XDECREF (state->timerlist);
    state->timerlist = PyList_New (0);
}

static PyObject*
_sdl_timeinit (PyObject *self)
{
    if (SDL_WasInit (SDL_INIT_TIMER))
        Py_RETURN_NONE;
        
    if (SDL_InitSubSystem (SDL_INIT_TIMER) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_sdl_timewasinit (PyObject *self)
{
    if (SDL_WasInit (SDL_INIT_TIMER))
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject*
_sdl_timequit (PyObject *self)
{
    _SDLTimerState *state = SDLTIMER_MOD_STATE (self);
    if (state->timerhook)
    {
        /* Reset the single timer hook */
        Py_XDECREF (state->timerhook);
        SDL_SetTimer (0, NULL);
        state->timerhook = NULL;
    }
    _remove_alltimers (state);
    Py_XDECREF (state->timerlist);
    state->timerlist = NULL;

    if (SDL_WasInit (SDL_INIT_TIMER))
        SDL_QuitSubSystem (SDL_INIT_TIMER);
    Py_RETURN_NONE;
}

static PyObject*
_sdl_timegetticks (PyObject *self)
{
    ASSERT_TIME_INIT(NULL);
    return PyLong_FromUnsignedLong (SDL_GetTicks ());
}

static PyObject*
_sdl_timedelay (PyObject *self, PyObject *args)
{
    Uint32 ms;

    if (!PyArg_ParseTuple (args, "l:delay", &ms))
        return NULL;
    Py_BEGIN_ALLOW_THREADS;
    SDL_Delay (ms);
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyObject*
_sdl_settimer (PyObject *self, PyObject *args)
{
    Uint32 interval;
    PyObject *hook;
    _SDLTimerState *state = SDLTIMER_MOD_STATE (self);

    ASSERT_TIME_INIT(NULL);

    if (!PyArg_ParseTuple (args, "lO:set_timer", &interval, &hook))
        return NULL;

    if (hook == Py_None)
    {
        /* Reset the timer hook */
        Py_XDECREF (state->timerhook);
        SDL_SetTimer (0, NULL);
        state->timerhook = NULL;
        Py_RETURN_NONE;
    }

    if (!PyCallable_Check (hook))
    {
        PyErr_SetString (PyExc_TypeError, "timer callback must be callable");
        return NULL;
    }

    Py_INCREF (hook);
    state->timerhook = hook;
    SDL_SetTimer (interval, _sdl_timercallback);

    Py_RETURN_NONE;
}

static PyObject*
_sdl_addtimer (PyObject *self, PyObject *args)
{
    SDL_TimerID id;
    Uint32 interval;
    _TimerData *timerdata;
    PyObject *retval, *func, *data = NULL;
    _SDLTimerState *state = SDLTIMER_MOD_STATE (self);

    ASSERT_TIME_INIT (NULL);
    
    if (!state->timerlist)
    {
        state->timerlist = PyList_New (0);
        if (!state->timerlist)
            return NULL;
    }

    if (!PyArg_ParseTuple (args, "iO|O:add_timer", &interval, &func, &data))
        return NULL;

    if (!PyCallable_Check (func))
    {
        PyErr_SetString (PyExc_TypeError, "timer callback must be callable");
        return NULL;
    }
    
    timerdata = PyMem_New (_TimerData, 1);
    if (!timerdata)
        return NULL;
    
    if (data)
    {
        if (!PySequence_Check (data))
        {
            /* Pack data */
            PyObject *tuple = PyTuple_New (1);
            if (!tuple)
                return NULL;
            PyTuple_SET_ITEM (tuple, 0, data);
            Py_INCREF (data);
            data = tuple;
        }
        else
        {
            Py_XINCREF (data);
        }
    }
    Py_INCREF (func);

    timerdata->callable = func;
    timerdata->param = data;
    timerdata->id = NULL;

#ifdef WITH_THREAD
    PyEval_InitThreads ();
    timerdata->thread = PyThreadState_New (PyThreadState_Get ()->interp);
#endif

    retval = PyCObject_FromVoidPtr (timerdata, NULL);
    id = SDL_AddTimer (interval, _sdl_timerfunc, timerdata);
    if (!id)
    {
        Py_DECREF (retval);
        _free_timerdata (timerdata);
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    timerdata->id = id;

    if (PyList_Append (state->timerlist, retval) == -1)
    {
        Py_DECREF (retval);
        _free_timerdata (timerdata);
        SDL_RemoveTimer (id);
        return NULL;
    }
    return retval;
}

static PyObject*
_sdl_removetimer (PyObject *self, PyObject *args)
{
    _TimerData *timerdata = NULL, *idobj;
    int found = 0;
    Py_ssize_t pos, count;
    PyObject *val, *cobj;
    _SDLTimerState *state = SDLTIMER_MOD_STATE (self);
    
    ASSERT_TIME_INIT (NULL);
    
    if (!PyArg_ParseTuple (args, "O:remove_timer", &cobj))
        return NULL;

    if (!state->timerlist  || !PyCObject_Check (cobj))
    {
        PyErr_SetString (PyExc_TypeError, "invalid timer id");
        return NULL;
    }

    idobj = (_TimerData*) PyCObject_AsVoidPtr (cobj);
    if (!idobj || idobj->id == NULL)
    {
        PyErr_SetString (PyExc_ValueError, "timer already removed");
        return NULL;
    }
    
    count = PyList_GET_SIZE (state->timerlist);
    for (pos = 0; pos < count; pos++)
    {
        val = PyList_GET_ITEM (state->timerlist, pos);
        timerdata = (_TimerData*) PyCObject_AsVoidPtr (val);
        if (timerdata->id != idobj->id)
            continue;
        found = 1;
        break;
    }
    if (!found)
    {
        PyErr_SetString (PyExc_TypeError, "invalid timer id");
        return NULL;
    }
    
    if (PySequence_DelItem (state->timerlist, pos) == -1)
        return NULL;
    _free_timerdata (timerdata);
    Py_RETURN_NONE;
}

static int
_timer_traverse (PyObject *mod, visitproc visit, void *arg)
{
    _SDLTimerState *state = SDLTIMER_MOD_STATE (mod);
    Py_VISIT (state->timerhook);
    Py_VISIT (state->timerlist);
    return 0;
}
static int
_timer_clear (PyObject *mod)
{
    _SDLTimerState *state = SDLTIMER_MOD_STATE (mod);
    
    _remove_alltimers (state);
    Py_CLEAR (state->timerhook);
    Py_CLEAR (state->timerlist);
    return 0;
}

#ifdef IS_PYTHON_3
struct PyModuleDef _timermodule = {
    PyModuleDef_HEAD_INIT,
    "time",
    DOC_TIME,
    sizeof (_SDLTimerState),
    _time_methods,
    NULL,
    _timer_traverse,
    _timer_clear,
    NULL
};
#endif

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_time (void)
#else
PyMODINIT_FUNC inittime (void)
#endif
{
    PyObject *mod;
    _SDLTimerState *state;

#ifdef IS_PYTHON_3
    mod = PyModule_Create (&_timermodule);
#else
    mod = Py_InitModule3 ("time", _time_methods, DOC_TIME);
#endif
    if (!mod)
        goto fail;
    state = SDLTIMER_MOD_STATE(mod);
    state->timerhook = NULL;
    state->timerlist = NULL;

    if (import_pygame2_base () < 0)
        goto fail;
    if (import_pygame2_sdl_base () < 0)
        goto fail;
    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
