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

#include "pygame.h"

#include "pgcompat.h"

#include "doc/time_doc.h"

#define WORST_CLOCK_ACCURACY 12

typedef struct pgEventTimer {
    struct pgEventTimer *next;
    intptr_t timer_id;
    pgEventObject *event;
    int repeat;
} pgEventTimer;

static pgEventTimer *pg_event_timer = NULL;
static SDL_mutex *timermutex = NULL;
static intptr_t pg_timer_id = 0;

static PyObject *
pg_time_autoquit(PyObject *self)
{
    pgEventTimer *hunt, *todel;
    /* We can let errors silently pass in this function, because this
     * needs to run */
    SDL_LockMutex(timermutex);
    if (pg_event_timer) {
        hunt = pg_event_timer;
        while (hunt) {
            todel = hunt;
            hunt = hunt->next;
            Py_DECREF(todel->event);
            PyMem_Free(todel);
        }
        pg_event_timer = NULL;
        pg_timer_id = 0;
    }
    SDL_UnlockMutex(timermutex);
    /* After we are done, we can destroy the mutex as well */
    SDL_DestroyMutex(timermutex);
    timermutex = NULL;
    Py_RETURN_NONE;
}

static PyObject *
pg_time_autoinit(PyObject *self)
{
    /* allocate a mutex for timer data holding struct*/
    if (!timermutex) {
        timermutex = SDL_CreateMutex();
        if (!timermutex)
            return RAISE(pgExc_SDLError, SDL_GetError());
    }
    Py_RETURN_NONE;
}

static intptr_t
_pg_add_event_timer(pgEventObject *ev, int repeat)
{
    pgEventTimer *new;

    new = PyMem_New(pgEventTimer, 1);
    if (!new) {
        PyErr_NoMemory();
        return 0;
    }

    if (SDL_LockMutex(timermutex) < 0) {
        /* this case will almost never happen, but still handle it */
        PyMem_Free(new);
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return 0;
    }

    pg_timer_id++;

    new->next = pg_event_timer;
    new->timer_id = pg_timer_id;
    new->event = ev;
    new->repeat = repeat;
    pg_event_timer = new;

    /* Chances of it failing here are next to zero, dont do anything */
    SDL_UnlockMutex(timermutex);
    return new->timer_id;
}

static void
_pg_remove_event_timer(pgEventObject *ev)
{
    pgEventTimer *hunt, *prev = NULL;

    SDL_LockMutex(timermutex);
    if (pg_event_timer) {
        hunt = pg_event_timer;
        while (hunt->event->type != ev->type) {
            prev = hunt;
            hunt = hunt->next;
            if (!hunt) {
                /* Reached end without finding a match, quit early */
                SDL_UnlockMutex(timermutex);
                return;
            }
        }
        if (prev)
            prev->next = hunt->next;
        else
            pg_event_timer = hunt->next;
        Py_DECREF(hunt->event);
        PyMem_Del(hunt);
    }
    /* Chances of it failing here are next to zero, dont do anything */
    SDL_UnlockMutex(timermutex);
}

static pgEventTimer *
_pg_get_event_on_timer(intptr_t timer_id)
{
    pgEventTimer *hunt;

    if (SDL_LockMutex(timermutex) < 0)
        return NULL;

    hunt = pg_event_timer;
    while (hunt) {
        if (hunt->timer_id == timer_id) {
            if (hunt->repeat >= 0) {
                hunt->repeat--;
            }
            break;
        }
        hunt = hunt->next;
    }

    /* Chances of it failing here are next to zero, dont do anything */
    SDL_UnlockMutex(timermutex);
    return hunt;
}

static Uint32
timer_callback(Uint32 interval, void *param)
{
    pgEventTimer *evtimer;
    SDL_Event event;
    PyGILState_STATE gstate;

    evtimer = _pg_get_event_on_timer((intptr_t)param);
    if (!evtimer)
        return 0;

    /* This function runs in a seperate thread, so we acquire the GIL,
     * pgEvent_FillUserEvent and _pg_remove_event_timer do python API calls */
    gstate = PyGILState_Ensure();

    if (SDL_WasInit(SDL_INIT_VIDEO)) {
        pgEvent_FillUserEvent(evtimer->event, &event);
        if (SDL_PushEvent(&event) <= 0)
            Py_DECREF(evtimer->event->dict);
    }
    else
        evtimer->repeat = 0;

    if (!evtimer->repeat) {
        /* This does memory cleanup */
        _pg_remove_event_timer(evtimer->event);
        interval = 0;
    }

    PyGILState_Release(gstate);
    return interval;
}

static int
accurate_delay(int ticks)
{
    int funcstart, delay;
    if (ticks <= 0)
        return 0;

    if (!SDL_WasInit(SDL_INIT_TIMER)) {
        if (SDL_InitSubSystem(SDL_INIT_TIMER)) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            return -1;
        }
    }

    funcstart = SDL_GetTicks();
    if (ticks >= WORST_CLOCK_ACCURACY) {
        delay = (ticks - 2) - (ticks % WORST_CLOCK_ACCURACY);
        if (delay >= WORST_CLOCK_ACCURACY) {
            Py_BEGIN_ALLOW_THREADS;
            SDL_Delay(delay);
            Py_END_ALLOW_THREADS;
        }
    }
    do {
        delay = ticks - (SDL_GetTicks() - funcstart);
    } while (delay > 0);

    return SDL_GetTicks() - funcstart;
}

static PyObject *
time_get_ticks(PyObject *self)
{
    if (!SDL_WasInit(SDL_INIT_TIMER))
        return PyLong_FromLong(0);
    return PyLong_FromLong(SDL_GetTicks());
}

static PyObject *
time_delay(PyObject *self, PyObject *arg)
{
    int ticks;
    PyObject *arg0;

    /*for some reason PyArg_ParseTuple is puking on -1's! BLARG!*/
    if (PyTuple_Size(arg) != 1)
        return RAISE(PyExc_ValueError, "delay requires one integer argument");
    arg0 = PyTuple_GET_ITEM(arg, 0);
    if (!PyLong_Check(arg0))
        return RAISE(PyExc_TypeError, "delay requires one integer argument");

    ticks = PyLong_AsLong(arg0);
    if (ticks < 0)
        ticks = 0;

    ticks = accurate_delay(ticks);
    if (ticks == -1)
        return NULL;
    return PyLong_FromLong(ticks);
}

static PyObject *
time_wait(PyObject *self, PyObject *arg)
{
    int ticks, start;
    PyObject *arg0;

    /*for some reason PyArg_ParseTuple is puking on -1's! BLARG!*/
    if (PyTuple_Size(arg) != 1)
        return RAISE(PyExc_ValueError, "delay requires one integer argument");
    arg0 = PyTuple_GET_ITEM(arg, 0);
    if (!PyLong_Check(arg0))
        return RAISE(PyExc_TypeError, "delay requires one integer argument");

    if (!SDL_WasInit(SDL_INIT_TIMER)) {
        if (SDL_InitSubSystem(SDL_INIT_TIMER)) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
    }

    ticks = PyLong_AsLong(arg0);
    if (ticks < 0)
        ticks = 0;

    start = SDL_GetTicks();
    Py_BEGIN_ALLOW_THREADS;
    SDL_Delay(ticks);
    Py_END_ALLOW_THREADS;

    return PyLong_FromLong(SDL_GetTicks() - start);
}

static PyObject *
time_set_timer(PyObject *self, PyObject *args, PyObject *kwargs)
{
    int ticks, loops = 0;
    intptr_t timer_id;
    PyObject *obj;
    pgEventObject *e;

    static char *kwids[] = {"event", "millis", "loops", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|i", kwids, &obj, &ticks,
                                     &loops))
        return NULL;

    if (!timermutex)
        return RAISE(pgExc_SDLError, "pygame is not initialized");

    if (PyLong_Check(obj)) {
        e = (pgEventObject *)pgEvent_New2(PyLong_AsLong(obj), NULL);
        if (!e)
            return NULL;
    }
    else if (pgEvent_Check(obj)) {
        Py_INCREF(obj);
        e = (pgEventObject *)obj;
    }
    else
        return RAISE(PyExc_TypeError,
                     "first argument must be an event type or event object");

    /* stop original timer, if it exists */
    _pg_remove_event_timer(e);

    if (ticks <= 0) {
        Py_DECREF(e);
        Py_RETURN_NONE;
    }

    /* just doublecheck that timer is initialized */
    if (!SDL_WasInit(SDL_INIT_TIMER)) {
        if (SDL_InitSubSystem(SDL_INIT_TIMER)) {
            Py_DECREF(e);
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
    }

    timer_id = _pg_add_event_timer(e, loops);
    if (!timer_id) {
        Py_DECREF(e);
        return NULL;
    }

    if (!SDL_AddTimer(ticks, timer_callback, (void *)timer_id)) {
        _pg_remove_event_timer(e); /* Does cleanup */
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    Py_RETURN_NONE;
}

/*clock object interface*/
typedef struct {
    PyObject_HEAD int last_tick;
    int fps_count, fps_tick;
    float fps;
    int timepassed, rawpassed;
    PyObject *rendered;
} PyClockObject;

// to be called by the other tick functions.
static PyObject *
clock_tick_base(PyObject *self, PyObject *arg, int use_accurate_delay)
{
    PyClockObject *_clock = (PyClockObject *)self;
    float framerate = 0.0f;
    int nowtime;

    if (!PyArg_ParseTuple(arg, "|f", &framerate))
        return NULL;

    if (framerate) {
        int delay, endtime = (int)((1.0f / framerate) * 1000.0f);
        _clock->rawpassed = SDL_GetTicks() - _clock->last_tick;
        delay = endtime - _clock->rawpassed;

        /*just doublecheck that timer is initialized*/
        if (!SDL_WasInit(SDL_INIT_TIMER)) {
            if (SDL_InitSubSystem(SDL_INIT_TIMER)) {
                return RAISE(pgExc_SDLError, SDL_GetError());
            }
        }

        if (use_accurate_delay)
            delay = accurate_delay(delay);
        else {
            // this uses sdls delay, which can be inaccurate.
            if (delay < 0)
                delay = 0;

            Py_BEGIN_ALLOW_THREADS;
            SDL_Delay((Uint32)delay);
            Py_END_ALLOW_THREADS;
        }

        if (delay == -1)
            return NULL;
    }

    nowtime = SDL_GetTicks();
    _clock->timepassed = nowtime - _clock->last_tick;
    _clock->fps_count += 1;
    _clock->last_tick = nowtime;
    if (!framerate)
        _clock->rawpassed = _clock->timepassed;

    if (!_clock->fps_tick) {
        _clock->fps_count = 0;
        _clock->fps_tick = nowtime;
    }
    else if (_clock->fps_count >= 10) {
        _clock->fps =
            _clock->fps_count / ((nowtime - _clock->fps_tick) / 1000.0f);
        _clock->fps_count = 0;
        _clock->fps_tick = nowtime;
        Py_XDECREF(_clock->rendered);
    }
    return PyLong_FromLong(_clock->timepassed);
}

static PyObject *
clock_tick(PyObject *self, PyObject *arg)
{
    return clock_tick_base(self, arg, 0);
}

static PyObject *
clock_tick_busy_loop(PyObject *self, PyObject *arg)
{
    return clock_tick_base(self, arg, 1);
}

static PyObject *
clock_get_fps(PyObject *self, PyObject *args)
{
    PyClockObject *_clock = (PyClockObject *)self;
    return PyFloat_FromDouble(_clock->fps);
}

static PyObject *
clock_get_time(PyObject *self, PyObject *args)
{
    PyClockObject *_clock = (PyClockObject *)self;
    return PyLong_FromLong(_clock->timepassed);
}

static PyObject *
clock_get_rawtime(PyObject *self, PyObject *args)
{
    PyClockObject *_clock = (PyClockObject *)self;
    return PyLong_FromLong(_clock->rawpassed);
}

/* clock object internals */

static struct PyMethodDef clock_methods[] = {
    {"tick", clock_tick, METH_VARARGS, DOC_CLOCKTICK},
    {"get_fps", clock_get_fps, METH_NOARGS, DOC_CLOCKGETFPS},
    {"get_time", clock_get_time, METH_NOARGS, DOC_CLOCKGETTIME},
    {"get_rawtime", clock_get_rawtime, METH_NOARGS, DOC_CLOCKGETRAWTIME},
    {"tick_busy_loop", clock_tick_busy_loop, METH_VARARGS,
     DOC_CLOCKTICKBUSYLOOP},
    {NULL, NULL, 0, NULL}};

static void
clock_dealloc(PyObject *self)
{
    PyClockObject *_clock = (PyClockObject *)self;
    Py_XDECREF(_clock->rendered);
    PyObject_Free(self);
}

PyObject *
clock_str(PyObject *self)
{
    char str[1024];
    PyClockObject *_clock = (PyClockObject *)self;

    sprintf(str, "<Clock(fps=%.2f)>", (float)_clock->fps);

    return PyUnicode_FromString(str);
}

static int
clock_init(PyClockObject *self, PyObject *args, PyObject *kwargs)
{
    if (!SDL_WasInit(SDL_INIT_TIMER)) {
        if (SDL_InitSubSystem(SDL_INIT_TIMER)) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            return -1;
        }
    }

    self->fps_tick = 0;
    self->timepassed = 0;
    self->rawpassed = 0;
    self->last_tick = SDL_GetTicks();
    self->fps = 0.0f;
    self->fps_count = 0;
    self->rendered = NULL;

    return 0;
}

static PyTypeObject PyClock_Type = {
    PyVarObject_HEAD_INIT(NULL, 0) "Clock", /* name */
    sizeof(PyClockObject),                  /* basic size */
    0,                                      /* itemsize */
    clock_dealloc,                          /* dealloc */
    0,                                      /* print */
    0,                                      /* getattr */
    0,                                      /* setattr */
    0,                                      /* compare */
    clock_str,                              /* repr */
    0,                                      /* as_number */
    0,                                      /* as_sequence */
    0,                                      /* as_mapping */
    (hashfunc)0,                            /* hash */
    (ternaryfunc)0,                         /* call */
    clock_str,                              /* str */
    0,                                      /* tp_getattro */
    0,                                      /* tp_setattro */
    0,                                      /* tp_as_buffer */
    0,                                      /* flags */
    DOC_PYGAMETIMECLOCK,                    /* Documentation string */
    0,                                      /* tp_traverse */
    0,                                      /* tp_clear */
    0,                                      /* tp_richcompare */
    0,                                      /* tp_weaklistoffset */
    0,                                      /* tp_iter */
    0,                                      /* tp_iternext */
    clock_methods,                          /* tp_methods */
    0,                                      /* tp_members */
    0,                                      /* tp_getset */
    0,                                      /* tp_base */
    0,                                      /* tp_dict */
    0,                                      /* tp_descr_get */
    0,                                      /* tp_descr_set */
    0,                                      /* tp_dictoffset */
    (initproc)clock_init,                   /* tp_init */
    0,                                      /* tp_alloc */
    PyType_GenericNew,                      /* tp_new */
};

static PyMethodDef _time_methods[] = {
    {"__PYGAMEinit__", (PyCFunction)pg_time_autoinit, METH_NOARGS,
     "auto initialize function for time"},
    {"__PYGAMEquit__", (PyCFunction)pg_time_autoquit, METH_NOARGS,
     "auto quit function for time"},
    {"get_ticks", (PyCFunction)time_get_ticks, METH_NOARGS,
     DOC_PYGAMETIMEGETTICKS},
    {"delay", time_delay, METH_VARARGS, DOC_PYGAMETIMEDELAY},
    {"wait", time_wait, METH_VARARGS, DOC_PYGAMETIMEWAIT},
    {"set_timer", (PyCFunction)time_set_timer, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMETIMESETTIMER},

    {NULL, NULL, 0, NULL}};

#ifdef __SYMBIAN32__
PYGAME_EXPORT
void
initpygame_time(void)
#else
MODINIT_DEFINE(time)
#endif
{
    PyObject *module;
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "time",
                                         DOC_PYGAMETIME,
                                         -1,
                                         _time_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};

    /* need to import base module, just so SDL is happy. Do this first so if
       the module is there is an error the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        return NULL;
    }

    import_pygame_event();
    if (PyErr_Occurred()) {
        return NULL;
    }

    /* type preparation */
    if (PyType_Ready(&PyClock_Type) < 0) {
        return NULL;
    }

    /* create the module */
    module = PyModule_Create(&_module);
    if (!module) {
        return NULL;
    }

    Py_INCREF(&PyClock_Type);
    if (PyModule_AddObject(module, "Clock", (PyObject *)&PyClock_Type)) {
        Py_DECREF(&PyClock_Type);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
