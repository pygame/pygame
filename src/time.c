/*
    PyGame - Python Game Library
    Copyright (C) 2000  Pete Shinners

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
	if(!PyArg_ParseTuple(arg, "i", &ticks))
		return NULL;

	SDL_Delay(ticks);

	RETURN_NONE
}




static PyMethodDef time_builtins[] =
{
	{ "get_ticks", get_ticks, 1, doc_get_ticks },
	{ "delay", delay, 1, doc_delay },

	{ NULL, NULL }
};


    /*DOC*/ static char doc_pygame_time_MODULE[] =
    /*DOC*/    "Contains routines to help keep track of time. The timer\n"
    /*DOC*/    "resolution on most systems is around 10ms.\n"
    /*DOC*/    "\n"
    /*DOC*/    "All times are represented in milliseconds, which is simply\n"
    /*DOC*/    "Seconds*1000.(therefore 2500 milliseconds is 2.5 seconds)\n"
    /*DOC*/ ;

void inittime()
{
	PyObject *module, *dict;

    /* create the module */
	module = Py_InitModule3("time", time_builtins, doc_pygame_time_MODULE);
	dict = PyModule_GetDict(module);

	/*need to import base module, just so SDL is happy*/
	import_pygame_base();
}

