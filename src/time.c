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





    /*DOC*/ static char doc_get_ticks[] =
    /*DOC*/    "pygame.time.get_ticks() -> int\n"
    /*DOC*/    "milliseconds since startup\n"
    /*DOC*/    "\n"
    /*DOC*/    "This is the time in milliseconds since the pygame.time was\n"
    /*DOC*/    "imported.\n"
    /*DOC*/ ;

static PyObject* get_ticks(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	return PyInt_FromLong(SDL_GetTicks());
}



    /*DOC*/ static char doc_delay[] =
    /*DOC*/    "pygame.time.delay(millseconds) -> none\n"
    /*DOC*/    "delay for a number of milliseconds\n"
    /*DOC*/    "\n"
    /*DOC*/    "Will pause for a given number of milliseconds.\n"
    /*DOC*/ ;

static PyObject* delay(PyObject* self, PyObject* arg)
{
	int ticks;
	PyObject* arg0;
	/*for some reason PyArg_ParseTuple pukes on -1's! BLARG!*/

	if(PyTuple_Size(arg) != 1)
		return RAISE(PyExc_ValueError, "delay requires one integer argument");
	arg0 = PyTuple_GET_ITEM(arg, 0);
	if(!PyInt_Check(arg0))
		return RAISE(PyExc_TypeError, "delay requires one integer argument");

	ticks = PyInt_AsLong(arg0);
	if(ticks < 0)
		ticks = 0;

	Py_BEGIN_ALLOW_THREADS
	SDL_Delay(ticks);
	Py_END_ALLOW_THREADS

	RETURN_NONE
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

static PyObject* set_timer(PyObject* self, PyObject* arg)
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




static PyMethodDef time_builtins[] =
{
	{ "get_ticks", get_ticks, 1, doc_get_ticks },
	{ "delay", delay, 1, doc_delay },
	{ "set_timer", set_timer, 1, doc_set_timer },

	{ NULL, NULL }
};


    /*DOC*/ static char doc_pygame_time_MODULE[] =
    /*DOC*/    "Contains routines to help keep track of time. The timer\n"
    /*DOC*/    "resolution on most systems is around 10ms.\n"
    /*DOC*/    "\n"
    /*DOC*/    "All times are represented in milliseconds, which is simply\n"
    /*DOC*/    "Seconds*1000.(therefore 2500 milliseconds is 2.5 seconds)\n"
    /*DOC*/ ;

PYGAME_EXPORT
void inittime(void)
{
	PyObject *module, *dict;

    /* create the module */
	module = Py_InitModule3("time", time_builtins, doc_pygame_time_MODULE);
	dict = PyModule_GetDict(module);

	/*need to import base module, just so SDL is happy*/
	import_pygame_base();
}

