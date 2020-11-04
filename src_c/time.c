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

#if IS_SDLv2
#define pgNUMEVENTS (16 + (PG_NUMEVENTS - PGE_USEREVENT))
#else /* IS_SDLv1 */
#define pgNUMEVENTS PG_NUMEVENTS
#endif /* IS_SDLv1 */

static SDL_TimerID event_timers[pgNUMEVENTS] = {0};

#if IS_SDLv2
static size_t
_pg_enumerate_event(Uint32 type)
{
    assert(pgNUMEVENTS == 1 + 15 + (PG_NUMEVENTS - PGE_USEREVENT));
    switch (type) {
        case SDL_ACTIVEEVENT:
            return 1;
        case SDL_KEYDOWN:
            return 2;
        case SDL_KEYUP:
            return 3;
        case SDL_MOUSEMOTION:
            return 4;
        case SDL_MOUSEBUTTONDOWN:
            return 5;
        case SDL_MOUSEBUTTONUP:
            return 6;
        case SDL_JOYAXISMOTION:
            return 7;
        case SDL_JOYBALLMOTION:
            return 8;
        case SDL_JOYHATMOTION:
            return 9;
        case SDL_JOYBUTTONDOWN:
            return 10;
        case SDL_JOYBUTTONUP:
            return 11;
        case SDL_VIDEORESIZE:
            return 12;
        case SDL_VIDEOEXPOSE:
            return 13;
        case SDL_QUIT:
            return 14;
        case SDL_SYSWMEVENT:
            return 15;
    }
    if (type >= PGE_USEREVENT && type < PG_NUMEVENTS)
        return type - PGE_USEREVENT + 16;
    return 0;
}
#endif /* IS_SDLv2 */

static Uint32
_pg_timer_callback(Uint32 interval, void *param)
{
    if (SDL_WasInit(SDL_INIT_VIDEO)) {
        SDL_Event event;
        memset(&event, 0, sizeof(event));
        event.type = (intptr_t)param;
        SDL_PushEvent(&event);
    }
    return interval;
}

static Uint32
_pg_timer_callback_once(Uint32 interval, void *param)
{
    return _pg_timer_callback(0, param);
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
     * millisecond accuracy */
    return SDL_GetPerformanceFrequency();
}

#else /* IS_SDLv1 */
static Uint64
_pg_get_clock(void) {
    return SDL_GetTicks();
}

static Uint64
_pg_get_clock_precision(void) {
    return 1000;
}
#endif /* IS_SDLv1 */

static double
_pg_get_delta_millis(Uint64 start)
{
    return 1000.0 * ((double)(_pg_get_clock() - start) / _pg_get_clock_precision());
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
    Uint64 starttime = _pg_get_clock();
    Uint64 endtime = starttime + _pg_get_ticks_from_millis(millis);
    
    if (millis >= 20)
        sleeptime = (int)millis - 5;
    else if (millis >= 15)
        sleeptime = (int)millis - 4;
    else if (millis >= 10)
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

#if IS_SDLv2
static PyObject *
pg_time_set_timer(PyObject *self, PyObject *arg)
{
    SDL_TimerID newtimer;
    int ticks = 0;
    int once = 0;
    SDL_EventType event;
    size_t index;
    if (!PyArg_ParseTuple(arg, "ii|i", &event, &ticks, &once))
        return NULL;

    index = _pg_enumerate_event(event);
    if (index == 0)
        return RAISE(PyExc_ValueError, "Unrecognized event type");

    /*stop original timer*/
    if (event_timers[index]) {
        SDL_RemoveTimer(event_timers[index]);
        event_timers[index] = 0;
    }

    if (ticks <= 0)
        Py_RETURN_NONE;

    /*just doublecheck that timer is initialized*/
    if (!_pg_is_sdl_time_init())
        return NULL;

    if (once) {
        newtimer = SDL_AddTimer(ticks, _pg_timer_callback_once, (void *)event);
    } else {
        newtimer = SDL_AddTimer(ticks, _pg_timer_callback, (void *)event);
    }
    if (!newtimer)
        return RAISE(pgExc_SDLError, SDL_GetError());
    event_timers[index] = newtimer;

    Py_RETURN_NONE;
}
#else  /* IS_SDLv1 */
static PyObject *
pg_time_set_timer(PyObject *self, PyObject *arg)
{
    SDL_TimerID newtimer;
    int ticks = 0;
    int once = 0;
    intptr_t event = SDL_NOEVENT;
    if (!PyArg_ParseTuple(arg, "ii|i", &event, &ticks, &once))
        return NULL;

    if (event <= SDL_NOEVENT || event >= PG_NUMEVENTS)
        return RAISE(PyExc_ValueError,
                     "Event id must be between NOEVENT(0) and NUMEVENTS(32)");

    /*stop original timer*/
    if (event_timers[event]) {
        SDL_RemoveTimer(event_timers[event]);
        event_timers[event] = NULL;
    }

    if (ticks <= 0)
        Py_RETURN_NONE;

    /*just doublecheck that timer is initialized*/
    if (!_pg_is_sdl_time_init())
        return NULL;

    if (once) {
        newtimer = SDL_AddTimer(ticks, _pg_timer_callback_once, (void *)event);
    } else {
        newtimer = SDL_AddTimer(ticks, _pg_timer_callback, (void *)event);
    }
    if (!newtimer)
        return RAISE(pgExc_SDLError, SDL_GetError());
    event_timers[event] = newtimer;

    Py_RETURN_NONE;
}
#endif /* IS_SDLv1 */

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
    pgClockObject *_clock = (pgClockObject *)self;
    PyObject_DEL(self);
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
    TYPE_HEAD(NULL, 0) "Clock", /* name */
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
    0,                          /* tp_members */
    0,                          /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    0,                          /* tp_init */
    0,                          /* tp_alloc */
    0,                          /* tp_new */
};

PyObject *
pg_ClockInit(PyObject *self)
{
    pgClockObject *_clock = PyObject_NEW(pgClockObject, &pgClock_Type);

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
    {"get_ticks", (PyCFunction)pg_time_get_ticks, METH_NOARGS,
     DOC_PYGAMETIMEGETTICKS},
    {"delay", pg_time_wait, METH_VARARGS, DOC_PYGAMETIMEDELAY},
    {"wait", pg_time_wait, METH_VARARGS, DOC_PYGAMETIMEWAIT},
    {"set_timer", pg_time_set_timer, METH_VARARGS, DOC_PYGAMETIMESETTIMER},

    {"Clock", (PyCFunction)pg_ClockInit, METH_NOARGS, DOC_PYGAMETIMECLOCK},

    {NULL, NULL, 0, NULL}};

#ifdef __SYMBIAN32__
PYGAME_EXPORT
void
initpygame_time(void)
#else
MODINIT_DEFINE(time)
#endif
{
#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "time",
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
#if IS_SDLv2
    import_pygame_event();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
#endif /* IS_SDLv2 */

    /* type preparation */
    if (PyType_Ready(&pgClock_Type) < 0) {
        MODINIT_ERROR;
    }

    /* create the module */
#if PY3
    return PyModule_Create(&_module);
#else
    Py_InitModule3(MODPREFIX "time", _time_methods, DOC_PYGAMETIME);
#endif
}
