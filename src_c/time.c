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

/* This is a doubly linked list of event timers. Each timer has a python dict,
 * timer ID, SDL event type and a repeat member. Access to this linked list
 * must be protected with an SDL mutex.
 * An SDL mutex would be redundant if python GIL was unconditionally held in
 * the C callback function. But acquiring GIL can be costlier than acquiring a
 * dedictated mutex for event timers, because the GIL is usually held more
 * often.
 */
typedef struct pgEventTimer {
    struct pgEventTimer *prev;
    struct pgEventTimer *next;
    PyObject *event_dict;
    intptr_t timer_id;
    int event_type;
    int repeat;
} pgEventTimer;

/* Pointer to head of timers linked list */
static pgEventTimer *pg_event_timer = NULL;
/* SDL mutex to be held during access to above pointer */
static SDL_mutex *pg_timer_mutex = NULL;
/* Timer ID incrementer */
static intptr_t pg_timer_id = 0;

/* Free a timer instance from linked list. Needs timer mutex held for data
 * safety. Might need GIL for a timer with a python dict. The caller must
 * ensure that the GIL is held, in this case */
static void
_pg_timer_free(pgEventTimer *timer)
{
    if (!timer) {
        return;
    }

    /* remove reference to current timer from linked list */
    if (timer->prev) {
        timer->prev->next = timer->next;
        if (timer->next) {
            timer->next->prev = timer->prev;
        }
    }
    else {
        /* current timer is head of the queue, replace head */
        pg_event_timer = timer->next;
        if (pg_event_timer) {
            pg_event_timer->prev = NULL;
        }
    }

    Py_XDECREF(timer->event_dict);
    free(timer);
}

static PyObject *
pg_time_autoquit(PyObject *self, PyObject *_null)
{
    /* We can let errors silently pass in this function, because this
     * needs to run */
    if (pg_timer_mutex) {
        SDL_LockMutex(pg_timer_mutex);
    }
    while (pg_event_timer) {
        /* Keep freeing till all are freed */
        _pg_timer_free(pg_event_timer);
    }
    if (pg_timer_mutex) {
        SDL_UnlockMutex(pg_timer_mutex);
        /* After we are done, we can destroy the mutex as well */
        SDL_DestroyMutex(pg_timer_mutex);
        pg_timer_mutex = NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
pg_time_autoinit(PyObject *self, PyObject *_null)
{
#ifndef __EMSCRIPTEN__
    /* allocate a mutex for timer data holding struct */
    if (!pg_timer_mutex) {
        pg_timer_mutex = SDL_CreateMutex();
        if (!pg_timer_mutex)
            return RAISE(pgExc_SDLError, SDL_GetError());
    }
#endif
    Py_RETURN_NONE;
}

/* Add a new timer in the timer linked list at the first index. Needs timer
 * mutex held for data safety. Needs GIL if 'ev_dict' is non-NULL. The caller
 * must ensure that the GIL is held, in this case.
 * Returns -1 on memoryerror without python error set */
static int
_pg_add_event_timer(int ev_type, PyObject *ev_dict, int repeat)
{
    pgEventTimer *new = (pgEventTimer *)malloc(sizeof(pgEventTimer));
    if (!new) {
        return -1;
    }
    Py_XINCREF(ev_dict);

    pg_timer_id++;

    /* insert the timer into the doubly linked list at the first index */
    new->prev = NULL;
    new->next = pg_event_timer;
    new->timer_id = pg_timer_id;
    new->event_type = ev_type;
    new->event_dict = ev_dict;
    new->repeat = repeat;
    if (pg_event_timer) {
        pg_event_timer->prev = new;
    }
    pg_event_timer = new;

    return 0;
}

/* Get event on timer queue. Needs timer mutex held for data safety. Does not
 * need GIL held. Decrements repeat count of timer being returned.
 * Returns NULL if timer is not found. */
static pgEventTimer *
_pg_get_event_on_timer(intptr_t timer_id)
{
    pgEventTimer *hunt = pg_event_timer;
    while (hunt) {
        if (hunt->timer_id == timer_id) {
            if (hunt->repeat >= 0) {
                hunt->repeat--;
            }
            break;
        }
        hunt = hunt->next;
    }
    return hunt;
}

/* Clear event timer by type. Needs timer mutex and GIL held for data safety.
 */
static void
_pg_clear_event_timer_type(int ev_type)
{
    pgEventTimer *hunt = pg_event_timer;
    while (hunt) {
        if (hunt->event_type == ev_type) {
            _pg_timer_free(hunt);
            return;
        }
        hunt = hunt->next;
    }
}

/* Timer callback function
 * TODO: This needs better error handling and a way to report to the user */
static Uint32
timer_callback(Uint32 interval, void *param)
{
    pgEventTimer *evtimer;
    PyGILState_STATE gstate;
    int gil_held = 0;
    if (SDL_LockMutex(pg_timer_mutex) < 0) {
        return 0;
    }

    evtimer = _pg_get_event_on_timer((intptr_t)param);
    if (!evtimer) {
        interval = 0;
        goto end;
    }

    /* Acquire GIL only if python API calls will be made, and that will only
     * happen only if dict is a non-null python dict */
    if (evtimer->event_dict) {
        gstate = PyGILState_Ensure();
        gil_held = 1;
    }

    if (SDL_WasInit(SDL_INIT_VIDEO)) {
        pg_post_event((Uint32)evtimer->event_type, evtimer->event_dict);
    }
    else {
        evtimer->repeat = 0;
    }

    if (!evtimer->repeat) {
        /* This does memory cleanup */
        _pg_timer_free(evtimer);
        interval = 0;
    }

end:
    if (gil_held) {
        PyGILState_Release(gstate);
    }

    SDL_UnlockMutex(pg_timer_mutex);
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
    PyObject *obj, *ev_dict = NULL;
    int ev_type;
    pgEventObject *e;

    static char *kwids[] = {"event", "millis", "loops", NULL};

    /* do not allow set_timer to work on WASM for now... this needs some more
     * testing and fixes that are WIP on other PRs */
#ifdef __EMSCRIPTEN__
    return RAISE(PyExc_NotImplementedError,
                 "set_timer is not implemented on WASM yet");
#endif

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|i", kwids, &obj, &ticks,
                                     &loops))
        return NULL;

    if (!pg_timer_mutex)
        return RAISE(pgExc_SDLError, "pygame is not initialized");

    if (PyLong_Check(obj)) {
        ev_type = (int)PyLong_AsLong(obj);
        if (ev_type == -1 && PyErr_Occurred()) {
            return NULL;
        }
    }
    else if (pgEvent_Check(obj)) {
        e = (pgEventObject *)obj;
        ev_type = e->type;
        ev_dict = e->dict;
    }
    else
        return RAISE(PyExc_TypeError,
                     "first argument must be an event type or event object");

    if (SDL_LockMutex(pg_timer_mutex) < 0) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    /* get and clear original timer, if it exists */
    _pg_clear_event_timer_type(ev_type);

    /* This means that the user only wanted to remove an existing timer */
    if (ticks <= 0) {
        goto end;
    }

    /* just doublecheck that timer is initialized */
    if (!SDL_WasInit(SDL_INIT_TIMER)) {
        if (SDL_InitSubSystem(SDL_INIT_TIMER)) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            goto end;
        }
    }

    if (_pg_add_event_timer(ev_type, ev_dict, loops)) {
        PyErr_NoMemory();
        goto end;
    }

    if (!SDL_AddTimer(ticks, timer_callback, (void *)pg_timer_id)) {
        _pg_timer_free(pg_event_timer); /* Does cleanup */
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
    }

end:
    if (SDL_UnlockMutex(pg_timer_mutex)) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
    }

    if (PyErr_Occurred()) {
        return NULL;
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
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "Clock",
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
#if defined(BUILD_STATIC)
// avoid PyInit_time conflict with static builtin
MODINIT_DEFINE(pg_time)
#else
MODINIT_DEFINE(time)
#endif  // BUILD_STATIC
#endif  //__SYMBIAN32__
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
