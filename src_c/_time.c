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

typedef struct pgEventTimer {
    struct pgEventTimer *next;
    pgEventObject *event;
    int repeat;
} pgEventTimer;

static pgEventTimer *pg_event_timer = NULL;
static SDL_mutex *timermutex = NULL;

static void
_pg_event_timer_cleanup(void)
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
            PyMem_Del(todel);
        }
        pg_event_timer = NULL;
    }
    SDL_UnlockMutex(timermutex);
    /* After we are done, we can destroy the mutex as well */
    SDL_DestroyMutex(timermutex);
    timermutex = NULL;
}

static PyObject *
pg_time_autoinit(PyObject *self)
{
    /* register cleanup function for event timer holding structure,
     * allocate a mutex for this structure too */
    if (!timermutex && !pg_event_timer) {
        timermutex = SDL_CreateMutex();
        if (!timermutex) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            return PyInt_FromLong(0);
        }
        pg_RegisterQuit(_pg_event_timer_cleanup);
    }
    return PyInt_FromLong(1);
}

static int
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
        PyMem_Del(new);
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return 0;
    }

    new->next = pg_event_timer;
    new->event = ev;
    new->repeat = repeat;
    pg_event_timer = new;

    /* Chances of it failing here are next to zero, dont do anything */
    SDL_UnlockMutex(timermutex);
    return 1;
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
            if (!hunt)
                break;
        }
        if (hunt) {
            if (prev)
                prev->next = hunt->next;
            else
                pg_event_timer = hunt->next;
            Py_DECREF(hunt->event);
            PyMem_Del(hunt);
        }
    }
    /* Chances of it failing here are next to zero, dont do anything */
    SDL_UnlockMutex(timermutex);
}

static pgEventTimer *
_pg_get_event_on_timer(pgEventObject *ev)
{
    pgEventTimer *hunt;

    if (SDL_LockMutex(timermutex) < 0)
        return NULL;

    hunt = pg_event_timer;
    while (hunt) {
        if (hunt->event->type == ev->type) {
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
pg_timer_callback(Uint32 interval, void *param)
{
    pgEventTimer *evtimer;
    SDL_Event event;
    PyGILState_STATE gstate;

    evtimer = _pg_get_event_on_timer((pgEventObject *)param);
    if (!evtimer)
        return 0;

    /* This function runs in a seperate thread, so we acquire the GIL,
     * pgEvent_FillUserEvent and _pg_remove_event_timer do python API calls */
    gstate = PyGILState_Ensure();

    if (SDL_WasInit(SDL_INIT_VIDEO)) {
        pgEvent_FillUserEvent(evtimer->event, &event);
#if IS_SDLv1
        if (SDL_PushEvent(&event) < 0)
#else
        if (SDL_PushEvent(&event) <= 0)
#endif
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

/* check if SDL timer is initialised, if not try to initialize. Return 0
on failure and also set python error */
static int
_pg_is_sdl_time_init(void) {
    if (!SDL_WasInit(SDL_INIT_TIMER)) {
        if (SDL_InitSubSystem(SDL_INIT_TIMER)) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            return 0;
        }
    }
    return 1;
}

#if IS_SDLv2
static Uint64
_pg_get_clock(void) {
    return SDL_GetPerformanceCounter();
}

static Uint64
_pg_get_clock_precision(void) {
    /* Get the number of ticks of the precision clock per second.
     * This clock uses the most accurate clock available on the system
     * On most windows platforms, this has microsecond precision and on
     * most Unix-based platforms (Linux, Mac, etc) this gives nanosecond
     * precision (fallback to microsecond precision in rare cases)
     * In worst cases (extremely rare) it may fall back to using
     * millisecond accuracy.
     * Here, we divide by 1000 because we want ticks per millisecond, not
     * ticks per second */
    return SDL_GetPerformanceFrequency() / 1000;
}

#else /* IS_SDLv1 */
static Uint64
_pg_get_clock(void) {
    return SDL_GetTicks();
}

static Uint64
_pg_get_clock_precision(void) {
    return 1;
}
#endif /* IS_SDLv1 */

static double
_pg_get_delta_millis(Uint64 start)
{
    return (double)(_pg_get_clock() - start) / _pg_get_clock_precision();
}

static Uint64
_pg_get_ticks_from_millis(double delta_millis)
{
    return (Uint64)(delta_millis * _pg_get_clock_precision());
}

static double
_pg_accurate_delay(double millis)
{
    int sleeptime;
    Uint64 endtime, starttime = _pg_get_clock();

    if (millis <= 0)
        return 0.0;

    endtime = starttime + _pg_get_ticks_from_millis(millis);

    /* We want to strike a good balance between sleeping the processor
     * and doing a busy loop. This way, our delay is accurate, and we
     * do not burn the CPU either. The next block of code does just that,
     * it determines the time this function needs to sleep. Note that
     * these are just arbitrary values, chosen while trying to keep a
     * good balance */
    if (millis >= 12)
        sleeptime = (int)millis - 3;
    else if (millis >= 5)
        sleeptime = (int)millis - 2;
    else if (millis >= 3)
        sleeptime = (int)millis - 1;
    else
        sleeptime = 0;

    if (sleeptime) {
        Py_BEGIN_ALLOW_THREADS;
        SDL_Delay(sleeptime);
        Py_END_ALLOW_THREADS;
    }

    while (_pg_get_clock() < endtime); // wait here
    return _pg_get_delta_millis(starttime);
}

static PyObject *
pg_time_get_ticks(PyObject *self)
{
    if (!SDL_WasInit(SDL_INIT_TIMER))
        return PyInt_FromLong(0);
    return PyInt_FromLong(SDL_GetTicks());
}

static PyObject *
pg_time_wait(PyObject *self, PyObject *arg)
{
    double delay;
    if (!PyArg_ParseTuple(arg, "d", &delay))
        return NULL;

    if (!_pg_is_sdl_time_init())
        return NULL;

    return PyFloat_FromDouble(_pg_accurate_delay(delay));
}

static PyObject *
pg_time_set_timer(PyObject *self, PyObject *args, PyObject *kwargs)
{
    int ticks, loops = 0;
    PyObject *obj;
    pgEventObject *e;

    static char *kwids[] = {
        "event",
        "millis",
        "loops",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|i", kwids,
                                     &obj, &ticks, &loops))
        return NULL;

    if (!timermutex)
        return RAISE(pgExc_SDLError, "pygame is not initialized");

    if (PyInt_Check(obj)) {
        e = (pgEventObject *)pgEvent_New2(PyInt_AsLong(obj), NULL);
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

    /*just doublecheck that timer is initialized*/
    if (!_pg_is_sdl_time_init()) {
        Py_DECREF(e);
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    if (!_pg_add_event_timer(e, loops)) {
        Py_DECREF(e);
        return NULL;
    }

    if (!SDL_AddTimer(ticks, pg_timer_callback, (void *)e)) {
        _pg_remove_event_timer(e); /* Does cleanup */
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    Py_RETURN_NONE;
}

/*clock object interface*/
typedef struct {
    PyObject_HEAD
    int fps_count;
    Uint64 last_tick;
    double fps, fps_sum, timepassed, rawpassed;
} pgClockObject;

static PyObject *
pg_clock_tick(PyObject *self, PyObject *arg)
{
    pgClockObject *_clock = (pgClockObject *)self;
    double framerate = 0.0, delay = 0.0;

    if (!PyArg_ParseTuple(arg, "|d", &framerate))
        return NULL;

    /*just doublecheck that timer is initialized*/
    if (!_pg_is_sdl_time_init())
        return NULL;

    _clock->rawpassed = _pg_get_delta_millis(_clock->last_tick);
    if (framerate)
        delay = _pg_accurate_delay((1000.0 / framerate) - _clock->rawpassed);

    _clock->timepassed = _clock->rawpassed + delay;
    _clock->last_tick = _pg_get_clock();

    _clock->fps_count += 1;
    _clock->fps_sum += _clock->timepassed;

    if (_clock->fps_count >= 10) {
        _clock->fps = 1000.0 * (_clock->fps_count / _clock->fps_sum);
        _clock->fps_count = 0;
        _clock->fps_sum = 0.0;
    }
    return PyFloat_FromDouble(_clock->timepassed);
}

static PyObject *
pg_clock_get_fps(PyObject *self)
{
    pgClockObject *_clock = (pgClockObject *)self;
    return PyFloat_FromDouble(_clock->fps);
}

static PyObject *
pg_clock_get_time(PyObject *self)
{
    pgClockObject *_clock = (pgClockObject *)self;
    return PyFloat_FromDouble(_clock->timepassed);
}

static PyObject *
pg_clock_get_rawtime(PyObject *self)
{
    pgClockObject *_clock = (pgClockObject *)self;
    return PyFloat_FromDouble(_clock->rawpassed);
}

/* clock object internals */
static struct PyMethodDef _clock_methods[] = {
    {"tick", pg_clock_tick, METH_VARARGS, DOC_CLOCKTICK},
    {"get_fps", (PyCFunction)pg_clock_get_fps, METH_NOARGS, DOC_CLOCKGETFPS},
    {"get_time", (PyCFunction)pg_clock_get_time, METH_NOARGS, DOC_CLOCKGETTIME},
    {"get_rawtime", (PyCFunction)pg_clock_get_rawtime, METH_NOARGS,
     DOC_CLOCKGETRAWTIME},
    {"tick_busy_loop", pg_clock_tick, METH_VARARGS,
     DOC_CLOCKTICKBUSYLOOP},
    {NULL, NULL, 0, NULL}};

static void
pg_clock_dealloc(PyObject *self)
{
    PyObject_Del(self);
}

PyObject *
pg_clock_str(PyObject *self)
{
    char str[1024];
    pgClockObject *_clock = (pgClockObject *)self;

    sprintf(str, "<Clock(fps=%.2f)>", (float)_clock->fps);

    return Text_FromUTF8(str);
}

static PyTypeObject pgClock_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "Clock",                    /* name */
    sizeof(pgClockObject),      /* basic size */
    0,                          /* itemsize */
    pg_clock_dealloc,           /* dealloc */
    0,                          /* print */
    0,                          /* getattr */
    0,                          /* setattr */
    0,                          /* compare */
    pg_clock_str,               /* repr */
    0,                          /* as_number */
    0,                          /* as_sequence */
    0,                          /* as_mapping */
    (hashfunc)0,                /* hash */
    (ternaryfunc)0,             /* call */
    pg_clock_str,               /* str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    0,                          /* flags */
    DOC_PYGAMETIMECLOCK,        /* Documentation string */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _clock_methods,             /* tp_methods */
};

PyObject *
pg_ClockInit(PyObject *self)
{
    pgClockObject *_clock = PyObject_New(pgClockObject, &pgClock_Type);

    if (!_clock) {
        return NULL;
    }

    _clock->timepassed = 0.0;
    _clock->rawpassed = 0.0;
    _clock->fps_sum = 0.0;
    _clock->fps = 0.0;
    _clock->fps_count = 0;

    /*just doublecheck that timer is initialized*/
    if (!_pg_is_sdl_time_init())
        return NULL;

    _clock->last_tick = _pg_get_clock();
    return (PyObject *)_clock;
}

static PyMethodDef _time_methods[] = {
    {"__PYGAMEinit__", (PyCFunction)pg_time_autoinit, METH_NOARGS,
        "auto initialize function for time"},
    {"get_ticks", (PyCFunction)pg_time_get_ticks, METH_NOARGS,
     DOC_PYGAMETIMEGETTICKS},
    {"delay", pg_time_wait, METH_VARARGS, DOC_PYGAMETIMEDELAY},
    {"wait", pg_time_wait, METH_VARARGS, DOC_PYGAMETIMEWAIT},
    {"set_timer", (PyCFunction)pg_time_set_timer,
        METH_VARARGS | METH_KEYWORDS, DOC_PYGAMETIMESETTIMER},

    {"Clock", (PyCFunction)pg_ClockInit, METH_NOARGS, DOC_PYGAMETIMECLOCK},

    {NULL, NULL, 0, NULL}};

#ifdef __SYMBIAN32__
PYGAME_EXPORT
void
initpygame__time(void)
#else
MODINIT_DEFINE(_time)
#endif
{
#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "_time",
                                         DOC_PYGAMETIME,
                                         -1,
                                         _time_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};
#endif

    /* need to import base module, just so SDL is happy. Do this first so if
       the module is there is an error the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    import_pygame_event();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    /* type preparation */
    if (PyType_Ready(&pgClock_Type) < 0) {
        MODINIT_ERROR;
    }

    /* create the module */
#if PY3
    return PyModule_Create(&_module);
#else
    Py_InitModule3(MODPREFIX "_time", _time_methods, DOC_PYGAMETIME);
#endif
}
