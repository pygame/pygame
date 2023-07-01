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

/**
 * Linked list structure to hold timer data.
 */
typedef struct pgEventTimer {
    struct pgEventTimer
        *next;            /** Pointer to the next timer in the linked list. */
    intptr_t timer_id;    /** The ID of the timer. */
    pgEventObject *event; /** The event object associated with the timer. */
    int repeat;           /** The number of times the timer should repeat. */
} pgEventTimer;

/**
 * pg_event_timer is a pointer to the first timer in a linked list of
 * pgEventTimer structures. Each pgEventTimer structure represents a timer that
 * has been created using pygame.time.set_timer. The linked list is used to
 * keep track of all the timers that have been created. If no timers have been
 * created, pg_event_timer is NULL.
 */
static pgEventTimer *pg_event_timer = NULL;

/**
 * timermutex is a mutex used to protect the pg_event_timer linked list.
 * It is created when pygame.time.autoinit is called, and destroyed when
 * pygame.time.autoquit is called.
 */
static SDL_mutex *timermutex = NULL;

/**
 * pg_timer_id is an integer that is incremented each time a timer is created.
 */
static intptr_t pg_timer_id = 0;

static PyObject *
pg_time_autoquit(PyObject *self, PyObject *_null)
{
    pgEventTimer *hunt, *todel;
    /* We can let errors silently pass in this function, because this
     * needs to run */
    if (timermutex) {
        SDL_LockMutex(timermutex);
    }
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
    if (timermutex) {
        SDL_UnlockMutex(timermutex);
        /* After we are done, we can destroy the mutex as well */
        SDL_DestroyMutex(timermutex);
        timermutex = NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
pg_time_autoinit(PyObject *self, PyObject *_null)
{
#ifndef __EMSCRIPTEN__
    /* allocate a mutex for timer data holding struct*/
    if (!timermutex) {
        timermutex = SDL_CreateMutex();
        if (!timermutex)
            return RAISE(pgExc_SDLError, SDL_GetError());
    }
#endif
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

    Py_XINCREF(ev);

    new->next = pg_event_timer;
    new->timer_id = pg_timer_id;
    new->event = ev;
    new->repeat = repeat;
    pg_event_timer = new;

    /* Chances of it failing here are next to zero, dont do anything */
    SDL_UnlockMutex(timermutex);
    return new->timer_id;
}

/**
 * \brief Removes the pgEventTimer struct associated with the given
 * pgEventObject from the linked list of timers.
 *
 * \param ev The pgEventObject to remove from the linked list of timers.
 *
 * \note Requires GIL.
 * \note Requires locking of timermutex.
 */
static void
_pg_remove_event_timer(pgEventObject *ev)
{
    pgEventTimer *hunt, *prev = NULL;
    if (!pg_event_timer) {
        /* No timers, quit early */
        return;
    }

    if (SDL_LockMutex(timermutex) < 0)
        return;

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
    /* Chances of it failing here are next to zero, dont do anything */
    SDL_UnlockMutex(timermutex);
}

/**
 * Returns the pgEventTimer struct associated with the given timer_id.
 * Decrements the repeat counter of the timer by 1 if it is greater than or
 * equal to 0.
 *
 * \param timer_id The ID of the timer to retrieve.
 * \returns A pointer to the pgEventTimer struct associated with the given
 * timer_id, or NULL if not found.
 *
 * \note No GIL required
 * \note Requires timermutex lock
 */
static pgEventTimer *
_pg_get_event_on_timer(intptr_t timer_id)
{
    pgEventTimer *hunt, *found;

    if (SDL_LockMutex(timermutex) < 0)
        return NULL;

    hunt = pg_event_timer;
    found = NULL;
    while (hunt) {
        if (hunt->timer_id == timer_id) {
            if (hunt->repeat >= 0) {
                hunt->repeat--;
            }
            found = hunt;
            break;
        }
        hunt = hunt->next;
    }

    /* Chances of it failing here are next to zero, dont do anything */
    SDL_UnlockMutex(timermutex);
    return found;
}

static Uint32
timer_callback(Uint32 interval, void *param)
{
    pgEventTimer *evtimer;
    SDL_Event event;
    PyGILState_STATE gstate;

    /* This function runs in a separate thread, so we acquire the GIL,
     * pgEvent_FillUserEvent and _pg_remove_event_timer do python API calls */
    gstate = PyGILState_Ensure();

    evtimer = _pg_get_event_on_timer((intptr_t)param);
    if (evtimer == NULL) {
        PyGILState_Release(gstate);
        return 0;
    }

    if (SDL_WasInit(SDL_INIT_VIDEO)) {
        if (evtimer->event && evtimer->event->dict) {
            pgEvent_FillUserEvent(evtimer->event, &event);
            if (SDL_PushEvent(&event) <= 0)
                Py_DECREF(evtimer->event->dict);
        }
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
time_get_ticks(PyObject *self, PyObject *_null)
{
    if (!SDL_WasInit(SDL_INIT_TIMER))
        return PyLong_FromLong(0);
    return PyLong_FromLong(SDL_GetTicks());
}

static PyObject *
time_delay(PyObject *self, PyObject *arg)
{
    int ticks;
    if (!PyLong_Check(arg))
        return RAISE(PyExc_TypeError, "delay requires one integer argument");

    ticks = PyLong_AsLong(arg);
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
    if (!PyLong_Check(arg))
        return RAISE(PyExc_TypeError, "wait requires one integer argument");

    if (!SDL_WasInit(SDL_INIT_TIMER)) {
        if (SDL_InitSubSystem(SDL_INIT_TIMER)) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
    }

    ticks = PyLong_AsLong(arg);
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

#ifdef __EMSCRIPTEN__
    return RAISE(PyExc_NotImplementedError,
                 "set_timer is not implemented on WASM yet");
#endif

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
clock_get_fps(PyObject *self, PyObject *_null)
{
    PyClockObject *_clock = (PyClockObject *)self;
    return PyFloat_FromDouble(_clock->fps);
}

static PyObject *
clock_get_time(PyObject *self, PyObject *_null)
{
    PyClockObject *_clock = (PyClockObject *)self;
    return PyLong_FromLong(_clock->timepassed);
}

static PyObject *
clock_get_rawtime(PyObject *self, PyObject *_null)
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
    char str[64];
    PyClockObject *_clock = (PyClockObject *)self;

    int ret = PyOS_snprintf(str, 64, "<Clock(fps=%.2f)>", _clock->fps);
    if (ret < 0 || ret >= 64) {
        return RAISE(PyExc_RuntimeError,
                     "Internal PyOS_snprintf call failed!");
    }

    return PyUnicode_FromString(str);
}

static PyObject *
clock_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    char *kwids[] = {NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", kwids)) {
        /* This function does not actually take in any arguments, but this
         * argparse function is used to generate pythonic error messages if
         * any args are passed */
        return NULL;
    }

    if (!SDL_WasInit(SDL_INIT_TIMER)) {
        if (SDL_InitSubSystem(SDL_INIT_TIMER)) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
    }

    PyClockObject *self = (PyClockObject *)(type->tp_alloc(type, 0));
    self->fps_tick = 0;
    self->timepassed = 0;
    self->rawpassed = 0;
    self->last_tick = SDL_GetTicks();
    self->fps = 0.0f;
    self->fps_count = 0;
    self->rendered = NULL;

    return (PyObject *)self;
}

static PyTypeObject PyClock_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pygame.time.Clock",
    .tp_basicsize = sizeof(PyClockObject),
    .tp_dealloc = clock_dealloc,
    .tp_repr = clock_str,
    .tp_str = clock_str,
    .tp_doc = DOC_PYGAMETIMECLOCK,
    .tp_methods = clock_methods,
    .tp_new = clock_new,
};

static PyMethodDef _time_methods[] = {
    {"_internal_mod_init", (PyCFunction)pg_time_autoinit, METH_NOARGS,
     "auto initialize function for time"},
    {"_internal_mod_quit", (PyCFunction)pg_time_autoquit, METH_NOARGS,
     "auto quit function for time"},
    {"get_ticks", (PyCFunction)time_get_ticks, METH_NOARGS,
     DOC_PYGAMETIMEGETTICKS},
    {"delay", time_delay, METH_O, DOC_PYGAMETIMEDELAY},
    {"wait", time_wait, METH_O, DOC_PYGAMETIMEWAIT},
    {"set_timer", (PyCFunction)time_set_timer, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMETIMESETTIMER},

    {NULL, NULL, 0, NULL}};

#if defined(BUILD_STATIC)
// avoid PyInit_time conflict with static builtin
MODINIT_DEFINE(pg_time)
#else
MODINIT_DEFINE(time)
#endif  // BUILD_STATIC
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
