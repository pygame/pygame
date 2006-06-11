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
#include "pygamedocs.h"
#if 0
#include "font.h" /*needed for render function, does lazy init*/
#endif


static SDL_TimerID event_timers[SDL_NUMEVENTS] = {NULL};

static Uint32 timer_callback(Uint32 interval, void* param)
{
	if(SDL_WasInit(SDL_INIT_VIDEO))
	{
		SDL_Event event;
		memset(&event, 0, sizeof(event));
		event.type = (intptr_t)param;
		SDL_PushEvent(&event);
	}
	return interval;
}

#define WORST_CLOCK_ACCURACY 12
static int accurate_delay(int ticks)
{
        int funcstart, delay;
        if(ticks <= 0)
            return 0;

        if(!SDL_WasInit(SDL_INIT_TIMER))
	{
		if(SDL_InitSubSystem(SDL_INIT_TIMER))
                {
			RAISE(PyExc_SDLError, SDL_GetError());
                        return -1;
                }
	}

        funcstart = SDL_GetTicks();
        if(ticks >= WORST_CLOCK_ACCURACY)
        {
            delay = (ticks - 2) - (ticks % WORST_CLOCK_ACCURACY);
            if(delay >= WORST_CLOCK_ACCURACY)
            {
                Py_BEGIN_ALLOW_THREADS
                SDL_Delay(delay);
                Py_END_ALLOW_THREADS
            }
        }
	do{
		delay = ticks - (SDL_GetTicks() - funcstart);	
        }while(delay > 0);
	
	return SDL_GetTicks() - funcstart;	
}


static PyObject* time_get_ticks(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	if(!SDL_WasInit(SDL_INIT_TIMER))
		return PyInt_FromLong(0);
	return PyInt_FromLong(SDL_GetTicks());
}


static PyObject* time_delay(PyObject* self, PyObject* arg)
{
	int ticks;
	PyObject* arg0;

        /*for some reason PyArg_ParseTuple is puking on -1's! BLARG!*/
	if(PyTuple_Size(arg) != 1)
		return RAISE(PyExc_ValueError, "delay requires one integer argument");
	arg0 = PyTuple_GET_ITEM(arg, 0);
	if(!PyInt_Check(arg0))
		return RAISE(PyExc_TypeError, "delay requires one integer argument");

	ticks = PyInt_AsLong(arg0);
	if(ticks < 0)
		ticks = 0;

        ticks = accurate_delay(ticks);
        if(ticks == -1)
            return NULL;
        return PyInt_FromLong(ticks);
}


static PyObject* time_wait(PyObject* self, PyObject* arg)
{
	int ticks, start;
	PyObject* arg0;

        /*for some reason PyArg_ParseTuple is puking on -1's! BLARG!*/
	if(PyTuple_Size(arg) != 1)
		return RAISE(PyExc_ValueError, "delay requires one integer argument");
	arg0 = PyTuple_GET_ITEM(arg, 0);
	if(!PyInt_Check(arg0))
		return RAISE(PyExc_TypeError, "delay requires one integer argument");

        if(!SDL_WasInit(SDL_INIT_TIMER))
	{
		if(SDL_InitSubSystem(SDL_INIT_TIMER))
                {
			RAISE(PyExc_SDLError, SDL_GetError());
                        return NULL;
                }
	}

        
	ticks = PyInt_AsLong(arg0);
	if(ticks < 0)
		ticks = 0;

        start = SDL_GetTicks();
	Py_BEGIN_ALLOW_THREADS

	SDL_Delay(ticks);
	Py_END_ALLOW_THREADS
	
        return PyInt_FromLong(SDL_GetTicks() - start);
}


static PyObject* time_set_timer(PyObject* self, PyObject* arg)
{
	SDL_TimerID newtimer;
	int ticks = 0;
	intptr_t event = SDL_NOEVENT;
	if(!PyArg_ParseTuple(arg, "ii", &event, &ticks))
		return NULL;

	if(event <= SDL_NOEVENT || event >= SDL_NUMEVENTS)
		return RAISE(PyExc_ValueError, "Event id must be between NOEVENT(0) and NUMEVENTS(32)");

	/*stop original timer*/
	if(event_timers[event])
	{
		SDL_RemoveTimer(event_timers[event]);
		event_timers[event] = NULL;
	}

	if(ticks <= 0)
		RETURN_NONE

	/*just doublecheck that timer is initialized*/
	if(!SDL_WasInit(SDL_INIT_TIMER))
	{
		if(SDL_InitSubSystem(SDL_INIT_TIMER))
			return RAISE(PyExc_SDLError, SDL_GetError());
	}

	newtimer = SDL_AddTimer(ticks, timer_callback, (void*)event);
	if(!newtimer)
		return RAISE(PyExc_SDLError, SDL_GetError());
	event_timers[event] = newtimer;

	RETURN_NONE
}


/*clock object interface*/

typedef struct {
	PyObject_HEAD
	int last_tick;
        int fps_count, fps_tick;
        float fps;
        int timepassed, rawpassed;
        PyObject* rendered;
} PyClockObject;


// to be called by the other tick functions.
static PyObject* clock_tick_base(PyObject* self, PyObject* arg, int use_accurate_delay)
{
	PyClockObject* clock = (PyClockObject*)self;
        float framerate = 0.0f;
        int nowtime;

	if(!PyArg_ParseTuple(arg, "|f", &framerate))
            return NULL;

        if(framerate)
        {
            int delay, endtime = (int)((1.0f/framerate) * 1000.0f);
            clock->rawpassed = SDL_GetTicks() - clock->last_tick;
            delay = endtime - clock->rawpassed;

            /*just doublecheck that timer is initialized*/
            if(!SDL_WasInit(SDL_INIT_TIMER))
            {
                if(SDL_InitSubSystem(SDL_INIT_TIMER))
                {
                    RAISE(PyExc_SDLError, SDL_GetError());
                    return NULL;
                }
            }


            if(use_accurate_delay) {
                delay = accurate_delay(delay);
            } else {
                // this uses sdls delay, which can be inaccurate.
                if(delay < 0)
                    delay = 0;

                Py_BEGIN_ALLOW_THREADS
                SDL_Delay((Uint32)delay);
                Py_END_ALLOW_THREADS
            }


            if(delay == -1)
                return NULL;
        }
        
        nowtime = SDL_GetTicks();
        clock->timepassed = nowtime - clock->last_tick;
        clock->fps_count += 1;
        clock->last_tick = nowtime;
        if(!framerate)
            clock->rawpassed = clock->timepassed;

        if(!clock->fps_tick)
        {
            clock->fps_count = 0;
            clock->fps_tick = nowtime;
        }
        else if(clock->fps_count >= 10)
        {
            clock->fps = clock->fps_count / ((nowtime - clock->fps_tick) / 1000.0f);
            clock->fps_count = 0;
            clock->fps_tick = nowtime;
            Py_XDECREF(clock->rendered);
        }
        return PyInt_FromLong(clock->timepassed);
}

static PyObject* clock_tick(PyObject* self, PyObject* arg) 
{
    return clock_tick_base(self, arg, 0);
}

static PyObject* clock_tick_busy_loop(PyObject* self, PyObject* arg) 
{
    return clock_tick_base(self, arg, 1);
}




static PyObject* clock_get_fps(PyObject* self, PyObject* arg)
{
	PyClockObject* clock = (PyClockObject*)self;
        return PyFloat_FromDouble(clock->fps);
}


static PyObject* clock_get_time(PyObject* self, PyObject* arg)
{
	PyClockObject* clock = (PyClockObject*)self;
        return PyInt_FromLong(clock->timepassed);
}


static PyObject* clock_get_rawtime(PyObject* self, PyObject* arg)
{
	PyClockObject* clock = (PyClockObject*)self;
        return PyInt_FromLong(clock->rawpassed);
}


#if 0

/*this would be a very convenient function, but it's a big mess.
it requires the use of the font and surface modules. it will try
to lazily import them when they are first needed. unfortunately
this adds up to a lot of messy code. we'll wait on this one*/


static PyObject* clock_render_fps(PyObject* self, PyObject* arg)
{
	PyClockObject* clock = (PyClockObject*)self;
        char message[256];
        static int initialized = 0;
    
        if(initialized == 0)
        {
            PyObject *module, *func, *result;
            initialized = -1;
            import_pygame_surface();
            import_pygame_font();
            module = PyImport_ImportModule("pygame.font");
            if(module != NULL)
            {
                func = PyObject_GetAttrString(module, "init");
                if(func)
                {
                    result = PyObject_CallObject(func, NULL);
                    if(result)
                    {
                        initialized = 1;
                        Py_DECREF(result);
                    }
                    Py_DECREF(func);
                }
                Py_DECREF(module);
            }
            PyErr_Clear();
        }
        if(initialized == 1)
        {
            PyObject *module, *func, *result;
            module = PyImport_ImportModule("pygame.font");
            func = PyObject_GetAttrString(module, "Font");
            result = PyObject_CallFunction(func, "Oi", Py_None, 12);
            if(result)
            {
                Py_DECREF(func);
                func = PyObject_GetAttrString(result, "render");
                Py_DECREF(result);
                sprintf(message, "fps: %.2f", clock->fps);
                result = PyObject_CallFunction(func, "si(iii)(iii)",
                        message, 0, 255, 255, 255, 0, 0, 0);
                /**** BLIT IMG ONTO A PRESET IMAGE SIZE AND RETURN IT ****/
                Py_DECREF(result);
            }
            Py_DECREF(func);
            Py_DECREF(module);
        }
        else
        {
            /**** RETURN SMALL IMAGE WITH ALL COLORKEY? ****/
        }
        
        RETURN_NONE
}
#endif


/* clock object internals */

static struct PyMethodDef clock_methods[] =
{
	{"tick",	clock_tick,	1, DOC_CLOCKTICK },
	{"get_fps",	clock_get_fps,	1, DOC_CLOCKGETFPS },
	{"get_time",	clock_get_time,	1, DOC_CLOCKGETTIME },
	{"get_rawtime",	clock_get_rawtime,1, DOC_CLOCKGETRAWTIME },
	{"tick_busy_loop",	clock_tick_busy_loop,	1, DOC_CLOCKTICK },
/*        {"render_fps",  clock_render_fps,1,doc_clock_render_fps},*/
	{NULL,		NULL}
};

static void clock_dealloc(PyObject* self)
{
	PyClockObject* clock = (PyClockObject*)self;
        Py_XDECREF(clock->rendered);
        PyObject_DEL(self);	
}

static PyObject *clock_getattr(PyObject *self, char *name)
{
	return Py_FindMethod(clock_methods, self, name);
}

PyObject* clock_str(PyObject* self)
{
	char str[1024];
	PyClockObject* clock = (PyClockObject*)self;

	sprintf(str, "<Clock(fps=%.2f)>", (float)clock->fps);

	return PyString_FromString(str);
}


static PyTypeObject PyClock_Type =
{
	PyObject_HEAD_INIT(NULL)
	0,				/*size*/
	"Clock",			/*name*/
	sizeof(PyClockObject),          /*basic size*/
	0,				/*itemsize*/
	clock_dealloc,		        /*dealloc*/
	0,				/*print*/
	clock_getattr,		        /*getattr*/
	NULL,				/*setattr*/
	NULL,				/*compare*/
	clock_str,			/*repr*/
	NULL,				/*as_number*/
	NULL,				/*as_sequence*/
	NULL,				/*as_mapping*/
	(hashfunc)NULL, 		/*hash*/
	(ternaryfunc)NULL,		/*call*/
	clock_str, 		        /*str*/
	/* Space for future expansion */
	0L,0L,0L,0L,
	DOC_PYGAMETIMECLOCK /* Documentation string */
};



PyObject* ClockInit(PyObject* self, PyObject* arg)
{
	PyClockObject* clock;

	if(!PyArg_ParseTuple(arg, ""))
            return NULL;

        clock = PyObject_NEW(PyClockObject, &PyClock_Type);
	if(!clock)
		return NULL;

	/*just doublecheck that timer is initialized*/
	if(!SDL_WasInit(SDL_INIT_TIMER))
	{
		if(SDL_InitSubSystem(SDL_INIT_TIMER))
			return RAISE(PyExc_SDLError, SDL_GetError());
	}

        clock->fps_tick = 0;
        clock->last_tick = SDL_GetTicks();
        clock->fps = 0.0f;
        clock->fps_count = 0;
        clock->rendered = NULL;

	return (PyObject*)clock;
}


static PyMethodDef time_builtins[] =
{
	{ "get_ticks", time_get_ticks, 1, DOC_PYGAMETIMEGETTICKS },
	{ "delay", time_delay, 1, DOC_PYGAMETIMEDELAY },
	{ "wait", time_wait, 1, DOC_PYGAMETIMEWAIT },
	{ "set_timer", time_set_timer, 1, DOC_PYGAMETIMESETTIMER },

        { "Clock", ClockInit, 1, DOC_PYGAMETIMECLOCK },
        
	{ NULL, NULL }
};



PYGAME_EXPORT
void inittime(void)
{
	PyObject *module;

	PyType_Init(PyClock_Type);
    
    /* create the module */
	module = Py_InitModule3("time", time_builtins, DOC_PYGAMETIME);

	/*need to import base module, just so SDL is happy*/
	import_pygame_base();
}

