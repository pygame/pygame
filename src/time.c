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
		event.type = (int)param;
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


    /*DOC*/ static char doc_get_ticks[] =
    /*DOC*/    "pygame.time.get_ticks() -> int\n"
    /*DOC*/    "milliseconds since initialization\n"
    /*DOC*/    "\n"
    /*DOC*/    "This is the time in milliseconds since the pygame.time was\n"
    /*DOC*/    "imported. Always returns 0 before pygame.init() is called.\n"
    /*DOC*/ ;

static PyObject* time_get_ticks(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	if(!SDL_WasInit(SDL_INIT_TIMER))
		return PyInt_FromLong(0);
	return PyInt_FromLong(SDL_GetTicks());
}



    /*DOC*/ static char doc_delay[] =
    /*DOC*/    "pygame.time.delay(millseconds) -> time\n"
    /*DOC*/    "accurately delay for a number of milliseconds\n"
    /*DOC*/    "\n"
    /*DOC*/    "Will pause for a given number of milliseconds.\n"
    /*DOC*/    "This function will use the CPU in order to make\n"
    /*DOC*/    "the delay more accurate than wait().\n"
    /*DOC*/    "\n"
    /*DOC*/    "This returns the actual number of milliseconds used.\n"
    /*DOC*/ ;

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


    /*DOC*/ static char doc_wait[] =
    /*DOC*/    "pygame.time.wait(millseconds) -> time\n"
    /*DOC*/    "yielding delay for a number of milliseconds\n"
    /*DOC*/    "\n"
    /*DOC*/    "Will pause for a given number of milliseconds.\n"
    /*DOC*/    "This function sleeps the process to better share\n"
    /*DOC*/    "the CPU with other processes. It is less accurate\n"
    /*DOC*/    "than the delay() function.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This returns the actual number of milliseconds used.\n"
    /*DOC*/ ;

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



    /*DOC*/ static char doc_set_timer[] =
    /*DOC*/    "pygame.time.set_timer(eventid, milliseconds) -> int\n"
    /*DOC*/    "control timer events\n"
    /*DOC*/    "\n"
    /*DOC*/    "Every event id can have a timer attached to it. Calling\n"
    /*DOC*/    "this will set the timer in milliseconds for that event.\n"
    /*DOC*/    "setting milliseconds to 0 or less will disable that timer.\n"
    /*DOC*/    "When a timer for an event is set, that event will be\n"
    /*DOC*/    "placed on the event queue every given number of\n"
    /*DOC*/    "milliseconds.\n"
    /*DOC*/ ;

static PyObject* time_set_timer(PyObject* self, PyObject* arg)
{
	SDL_TimerID newtimer;
	int ticks = 0, event = SDL_NOEVENT;
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

	newtimer = SDL_AddTimer((ticks/10)*10, timer_callback, (void*)event);
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

    /*DOC*/ static char doc_clock_tick[] =
    /*DOC*/    "Clock.tick([ticks_per_sec_delay]) -> milliseconds\n"
    /*DOC*/    "control timer events\n"
    /*DOC*/    "\n"
    /*DOC*/    "Updates the number of ticks for this clock. It should usually\n"
    /*DOC*/    "be called once per frame. If you pass the optional delay argument\n"
    /*DOC*/    "the function will delay to keep the game running slower than the\n"
    /*DOC*/    "given ticks per second. The function also returns the number of\n"
    /*DOC*/    "milliseconds passed since the previous call to tick().\n"
    /*DOC*/ ;

static PyObject* clock_tick(PyObject* self, PyObject* arg)
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
            delay = accurate_delay(delay);
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


    /*DOC*/ static char doc_clock_get_fps[] =
    /*DOC*/    "Clock.get_fps() -> float\n"
    /*DOC*/    "get the current rate of frames per second\n"
    /*DOC*/    "\n"
    /*DOC*/    "This computes the running average of frames per second. This\n"
    /*DOC*/    "is the number of times the tick() method has been called per\n"
    /*DOC*/    "second.\n"
    /*DOC*/ ;

static PyObject* clock_get_fps(PyObject* self, PyObject* arg)
{
	PyClockObject* clock = (PyClockObject*)self;
        return PyFloat_FromDouble(clock->fps);
}


    /*DOC*/ static char doc_clock_get_time[] =
    /*DOC*/    "Clock.get_time() -> int\n"
    /*DOC*/    "get number of milliseconds between last two calls to tick()\n"
    /*DOC*/    "\n"
    /*DOC*/    "This is the same value returned from the call to Clock.tick().\n"
    /*DOC*/    "it is the number of milliseconds that passed between the last\n"
    /*DOC*/    "two calls to tick().\n"
    /*DOC*/ ;

static PyObject* clock_get_time(PyObject* self, PyObject* arg)
{
	PyClockObject* clock = (PyClockObject*)self;
        return PyInt_FromLong(clock->timepassed);
}

    /*DOC*/ static char doc_clock_get_rawtime[] =
    /*DOC*/    "Clock.get_rawtime() -> int\n"
    /*DOC*/    "get number of nondelayed milliseconds between last two calls to tick()\n"
    /*DOC*/    "\n"
    /*DOC*/    "This is similar to get_time(). It does not include the number of\n"
    /*DOC*/    "milliseconds that were delayed to keep the clock tick under a given\n"
    /*DOC*/    "framerate.\n"
    /*DOC*/ ;

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

    /*DOC*/ static char docXXX_clock_render_fps[] =
    /*DOC*/    "time.Clock.render_fps() -> Surface\n"
    /*DOC*/    "render the current rate of frames per second\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will render the current framerate into a pygame Surface.\n"
    /*DOC*/    "It requires the pygame.font module to be installed, or it will\n"
    /*DOC*/    "return a blank surface.\n"
    /*DOC*/ ;


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
	{"tick",	clock_tick,	1, doc_clock_tick },
	{"get_fps",	clock_get_fps,	1, doc_clock_get_fps },
	{"get_time",	clock_get_time,	1, doc_clock_get_time },
	{"get_rawtime",	clock_get_rawtime,1, doc_clock_get_rawtime },
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

    /*DOC*/ static char doc_Clock_MODULE[] =
    /*DOC*/    "Clocks are used to track and control the framerate\n"
    /*DOC*/    "of a game. You create the objects with the time.Clock()\n"
    /*DOC*/    "function. The clock can be used to limit the framerate\n"
    /*DOC*/    "of a game, as well as track the time used per frame.\n"
    /*DOC*/    "Use the pygame.time.Clock() function to create new Clock\n"
    /*DOC*/    "objects.\n"
    /*DOC*/;

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
	doc_Clock_MODULE /* Documentation string */
};


    /*DOC*/ static char doc_Clockinit[] =
    /*DOC*/    "pygame.time.Clock() -> Clock\n"
    /*DOC*/    "create a new clock\n"
    /*DOC*/    "\n"
    /*DOC*/    "Clocks are used to track and control the framerate\n"
    /*DOC*/    "of a game. You create the objects with the time.Clock()\n"
    /*DOC*/    "function. The clock can be used to limit the framerate\n"
    /*DOC*/    "of a game, as well as track the time used per frame.\n"
    /*DOC*/;

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
	{ "get_ticks", time_get_ticks, 1, doc_get_ticks },
	{ "delay", time_delay, 1, doc_delay },
	{ "wait", time_wait, 1, doc_wait },
	{ "set_timer", time_set_timer, 1, doc_set_timer },

        { "Clock", ClockInit, 1, doc_Clockinit },
        
	{ NULL, NULL }
};


    /*DOC*/ static char doc_pygame_time_MODULE[] =
    /*DOC*/    "Contains routines to help keep track of time. The timer\n"
    /*DOC*/    "resolution on most systems is around 10ms.\n"
    /*DOC*/    "\n"
    /*DOC*/    "All times are represented in milliseconds, which is simply\n"
    /*DOC*/    "Seconds*1000. (therefore 2500 milliseconds is 2.5 seconds)\n"
    /*DOC*/    "\n"
    /*DOC*/    "You can also create Clock instances to keep track of framerate.\n"
    /*DOC*/ ;

PYGAME_EXPORT
void inittime(void)
{
	PyObject *module, *dict;

	PyType_Init(PyClock_Type);
    
    /* create the module */
	module = Py_InitModule3("time", time_builtins, doc_pygame_time_MODULE);
	dict = PyModule_GetDict(module);

	/*need to import base module, just so SDL is happy*/
	import_pygame_base();
}

