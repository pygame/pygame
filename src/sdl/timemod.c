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

typedef struct
{
    SDL_TimerID  id;
    PyObject    *callable;
    PyObject    *param;
} _TimerData;

static PyObject *_timerhook = NULL;
static Uint32 _sdl_timercallback (Uint32 interval);

static PyObject *_timerlist = NULL;
static Uint32 _sdl_timerfunc (Uint32 interval, void *param);
static void _free_timerdata (void *data);

static PyObject* _sdl_timeinit (PyObject *self);
static PyObject* _sdl_timewasinit (PyObject *self);
static PyObject* _sdl_timequit (PyObject *self);
static PyObject* _sdl_timegetticks (PyObject *self);
static PyObject* _sdl_timedelay (PyObject *self, PyObject *args);
static PyObject* _sdl_settimer (PyObject *self, PyObject *args);
static PyObject* _sdl_addtimer (PyObject *self, PyObject *args);
static PyObject* _sdl_removetimer (PyObject *self, PyObject *args);

static PyMethodDef _time_methods[] = {
    { "init", (PyCFunction) _sdl_timeinit, METH_NOARGS, "" },
    { "was_init", (PyCFunction) _sdl_timewasinit, METH_NOARGS, "" },
    { "quit", (PyCFunction) _sdl_timequit, METH_NOARGS, "" },
    { "get_ticks", (PyCFunction) _sdl_timegetticks, METH_NOARGS, "" },
    { "delay", _sdl_timedelay, METH_VARARGS, "" },
    { "set_timer", _sdl_settimer, METH_VARARGS, "" },
    { "add_timer", _sdl_addtimer, METH_VARARGS, "" },
    { "remove_timer", _sdl_removetimer, METH_VARARGS, "" },
    { NULL, NULL, 0, NULL }
};

static Uint32
_sdl_timercallback (Uint32 interval)
{
    PyObject *result, *val;
    Uint32 retval;

    if (!_timerhook)
        return 1;
    
    val = PyLong_FromUnsignedLong (interval);
    result = PyObject_CallObject (_timerhook, val);
    Py_DECREF (val);

    if (!Uint32FromObj (result, &retval))
    {
        /* Wrong signature, remove the callback */
        PyErr_SetString (PyExc_ValueError,
            "callback must return a positive integer");
        Py_XDECREF (result);
        Py_XDECREF (_timerhook);
        _timerhook = NULL;
        SDL_SetTimer (0, NULL);
        return 0;
    }

    Py_XDECREF (result);
    return retval;
}

static Uint32
_sdl_timerfunc (Uint32 interval, void *param)
{
    _TimerData *timerdata;
    PyObject *result, *val, *timer;
    Uint32 retval;
    
    timer = (PyObject*) param;
    timerdata = (_TimerData*) PyCObject_AsVoidPtr (timer);

    val = PyLong_FromUnsignedLong (interval);
    if (timerdata->param)
        result = PyObject_CallFunctionObjArgs (timerdata->callable, val,
            timerdata->param);
    else
        result = PyObject_CallObject (timerdata->callable, val);

    Py_DECREF (val);

    if (!Uint32FromObj (result, &retval))
    {
        Py_ssize_t pos;

        Py_XDECREF (result);
        pos = PySequence_Index (_timerlist, timer);
        PySequence_DelItem (_timerlist, pos);

        /* Wrong signature, remove the callback */
        PyErr_SetString (PyExc_ValueError,
            "callback must return a positive integer");
        return 0;
    }

    Py_XDECREF (result);
    return retval;
}

static void
_free_timerdata (void *data)
{
    _TimerData *t = (_TimerData*) data;
    if (!data)
        return;

    if (t->id)
        SDL_RemoveTimer(t->id);

    Py_XDECREF (t->callable);
    Py_XDECREF (t->param);
    PyMem_Free (t);
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
    if (_timerhook)
    {
        /* Reset the single timer hook */
        Py_XDECREF (_timerhook);
        SDL_SetTimer (0, NULL);
        _timerhook = NULL;
    }
    Py_XDECREF (_timerlist);

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

    ASSERT_TIME_INIT(NULL);
    
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

    ASSERT_TIME_INIT(NULL);

    if (!PyArg_ParseTuple (args, "lO:set_timer", &interval, &hook))
        return NULL;

    if (hook == Py_None)
    {
        /* Reset the timer hook */
        Py_XDECREF (_timerhook);
        SDL_SetTimer (0, NULL);
        _timerhook = NULL;
        Py_RETURN_NONE;
    }

    if (!PyCallable_Check (hook))
    {
        PyErr_SetString (PyExc_TypeError, "timer callback must be callable");
        return NULL;
    }

    Py_INCREF (hook);
    _timerhook = hook;
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

    if (!_timerlist)
    {
        _timerlist = PyList_New (0);
        if (!_timerlist)
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

    Py_INCREF (func);
    Py_XINCREF (data);
    timerdata->callable = func;
    timerdata->param = data;

    retval = PyCObject_FromVoidPtr (timerdata, _free_timerdata);
    id = SDL_AddTimer (interval, _sdl_timerfunc, retval);
    if (!id)
    {
        Py_DECREF (retval);
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    timerdata->id = id;

    if (PyList_Append (_timerlist, retval) == -1)
    {
        Py_DECREF (retval); /* Takes care of freeing. */
        return NULL;
    }
    Py_DECREF (retval); /* Decrease incremented refcount  */
    return PyInt_FromSsize_t (PyList_Size (_timerlist));
}

static PyObject*
_sdl_removetimer (PyObject *self, PyObject *args)
{
    _TimerData *timerdata;
    Py_ssize_t pos;
    PyObject *val;
    
    if (!PyArg_ParseTuple (args, "n:remove_timer", &pos))
        return NULL;

    if (!_timerlist || pos < 0 || pos >= PyList_Size (_timerlist))
    {
        PyErr_SetString (PyExc_TypeError, "invalid timer id");
        return NULL;
    }

    val = PyList_GET_ITEM (_timerlist, pos);
    timerdata = (_TimerData*) PyCObject_AsVoidPtr (val);
    if (!SDL_RemoveTimer (timerdata->id))
        Py_RETURN_FALSE;

    timerdata->id = NULL;
    PySequence_DelItem (_timerlist, pos);
    Py_RETURN_TRUE;
}

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_time (void)
#else
PyMODINIT_FUNC inittime (void)
#endif
{
    PyObject *mod;

#ifdef IS_PYTHON_3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "time",
        "",
        -1,
        _time_methods,
        NULL, NULL, NULL, NULL
    };
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("time", _time_methods, "");
#endif
    if (!mod)
        goto fail;

    if (import_pygame2_base () < 0)
        goto fail;
    if (import_pygame2_sdl_base () < 0)
        goto fail;
    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
