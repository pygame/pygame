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

/*
 *  PyGAME event module
 */
#define PYGAMEAPI_EVENT_INTERNAL
#include "pygame.h"





staticforward PyTypeObject PyEvent_Type;
static PyObject* PyEvent_New(SDL_Event*);
static PyObject* PyEvent_New2(int, PyObject*);
#define PyEvent_Check(x) ((x)->ob_type == &PyEvent_Type)



static char* name_from_eventtype(int type)
{
	switch(type)
	{
	case SDL_ACTIVEEVENT:	return "ActiveEvent";
	case SDL_KEYDOWN:		return "KeyDown";
	case SDL_KEYUP:			return "KeyUp";
	case SDL_MOUSEMOTION:	return "MouseMotion";
	case SDL_MOUSEBUTTONDOWN:return "MouseButtonDown";
	case SDL_MOUSEBUTTONUP:	return "MouseButtonUp";
	case SDL_JOYAXISMOTION:	return "JoyAxisMotion";
	case SDL_JOYBALLMOTION:	return "JoyBallMotion";
	case SDL_JOYHATMOTION:	return "JoyHatMotion";
	case SDL_JOYBUTTONUP:	return "JoyButtonUp";
	case SDL_JOYBUTTONDOWN:	return "JoyButtonDown";
	case SDL_QUIT:			return "Quit";
	case SDL_SYSWMEVENT:	return "SysWMEvent";
	case SDL_VIDEORESIZE:	return "VideoResize";
	case SDL_NOEVENT:		return "NoEvent";
	}
	if(type >= SDL_USEREVENT && type < SDL_NUMEVENTS)
		return "UserEvent";

	return "Unkown";
}


static PyObject* dict_from_event(SDL_Event* event)
{
	PyObject* dict, *tuple, *obj;

	if(!(dict = PyDict_New()))
		return NULL;

	switch(event->type)
	{
	case SDL_ACTIVEEVENT:
		PyDict_SetItemString(dict, "gain", PyInt_FromLong(event->active.gain));
		PyDict_SetItemString(dict, "state", PyInt_FromLong(event->active.state));
		break;
	case SDL_KEYDOWN:
		if(event->key.keysym.unicode)
			PyDict_SetItemString(dict, "unicode", PyUnicode_FromUnicode(
							(Py_UNICODE*)&event->key.keysym.unicode,
							event->key.keysym.unicode > 0));
		else
			PyDict_SetItemString(dict, "unicode",
							PyUnicode_FromObject(PyString_FromString("")));
	case SDL_KEYUP:
		PyDict_SetItemString(dict, "key", PyInt_FromLong(event->key.keysym.sym));
		PyDict_SetItemString(dict, "mod", PyInt_FromLong(event->key.keysym.mod));
		break;
	case SDL_MOUSEMOTION:
		obj = Py_BuildValue("(ii)", event->motion.x, event->motion.y);
		PyDict_SetItemString(dict, "pos", obj);
		Py_DECREF(obj);
		obj = Py_BuildValue("(ii)", event->motion.xrel, event->motion.yrel);
		PyDict_SetItemString(dict, "rel", obj);
		Py_DECREF(obj);
		if((tuple = PyTuple_New(3)))
		{
			PyTuple_SET_ITEM(tuple, 0, PyInt_FromLong((event->motion.state&SDL_BUTTON(1)) != 0));
			PyTuple_SET_ITEM(tuple, 1, PyInt_FromLong((event->motion.state&SDL_BUTTON(2)) != 0));
			PyTuple_SET_ITEM(tuple, 2, PyInt_FromLong((event->motion.state&SDL_BUTTON(3)) != 0));
			PyDict_SetItemString(dict, "buttons", tuple);
		}
		break;
	case SDL_MOUSEBUTTONDOWN:
	case SDL_MOUSEBUTTONUP:
		obj = Py_BuildValue("(ii)", event->button.x, event->button.y);
		PyDict_SetItemString(dict, "pos", obj);
		Py_DECREF(obj);
		PyDict_SetItemString(dict, "button", PyInt_FromLong(event->button.button));
		break;
	case SDL_JOYAXISMOTION:
		PyDict_SetItemString(dict, "joy", PyInt_FromLong(event->jaxis.which));
		PyDict_SetItemString(dict, "axis", PyInt_FromLong(event->jaxis.axis));
		PyDict_SetItemString(dict, "value", PyFloat_FromDouble(event->jaxis.value/32767.0));
		break;
	case SDL_JOYBALLMOTION:
		PyDict_SetItemString(dict, "joy", PyInt_FromLong(event->jball.which));
		PyDict_SetItemString(dict, "ball", PyInt_FromLong(event->jball.ball));
		obj = Py_BuildValue("(ii)", event->jball.xrel, event->jball.yrel);
		PyDict_SetItemString(dict, "rel", obj);
		Py_DECREF(obj);
		break;
	case SDL_JOYHATMOTION:
		PyDict_SetItemString(dict, "joy", PyInt_FromLong(event->jhat.which));
		PyDict_SetItemString(dict, "hat", PyInt_FromLong(event->jhat.hat));
		PyDict_SetItemString(dict, "value", PyInt_FromLong(event->jhat.value));
		break;
	case SDL_JOYBUTTONUP:
	case SDL_JOYBUTTONDOWN:
		PyDict_SetItemString(dict, "joy", PyInt_FromLong(event->jbutton.which));
		PyDict_SetItemString(dict, "button", PyInt_FromLong(event->jbutton.button));
		break;
	case SDL_VIDEORESIZE:
		obj = Py_BuildValue("(ii)", event->resize.w, event->resize.h);
		PyDict_SetItemString(dict, "size", obj);
		Py_DECREF(obj);
		break;
	}
	if(event->type >= SDL_USEREVENT && event->type < SDL_NUMEVENTS)
	{
		PyDict_SetItemString(dict, "code", PyInt_FromLong(event->user.code));
		PyDict_SetItemString(dict, "data1", PyInt_FromLong((int)event->user.data1));
		PyDict_SetItemString(dict, "data2", PyInt_FromLong((int)event->user.data2));
	}

	return dict;
}




/* event object internals */

static void event_dealloc(PyObject* self)
{
	PyEventObject* e = (PyEventObject*)self;
	Py_XDECREF(e->dict);
	PyMem_DEL(self);	
}


static PyObject *event_getattr(PyObject *self, char *name)
{
	PyEventObject* e = (PyEventObject*)self;
	PyObject* item;

	if(!strcmp(name, "type"))
		return PyInt_FromLong(e->type);

	if(!strcmp(name, "dict"))
	{
		Py_INCREF(e->dict);
		return e->dict;
	}

	item = PyDict_GetItemString(e->dict, name);
	if(item)
		Py_INCREF(item);
	else
		RAISE(PyExc_AttributeError, "event member not defined");
	return item;
}


PyObject* event_str(PyObject* self)
{
	PyEventObject* e = (PyEventObject*)self;
	char str[1024];
	PyObject *strobj;

	strobj = PyObject_Str(e->dict);
	sprintf(str, "<Event(%d-%s %s)>", e->type, name_from_eventtype(e->type),
				PyString_AsString(strobj));

	Py_DECREF(strobj);
	return PyString_FromString(str);
}


static PyTypeObject PyEvent_Type =
{
	PyObject_HEAD_INIT(NULL)
	0,						/*size*/
	"Event",				/*name*/
	sizeof(PyEventObject),	/*basic size*/
	0,						/*itemsize*/
	event_dealloc,			/*dealloc*/
	0,						/*print*/
	event_getattr,			/*getattr*/
	NULL,					/*setattr*/
	NULL,					/*compare*/
	event_str,				/*repr*/
	NULL,					/*as_number*/
	NULL,					/*as_sequence*/
	NULL,					/*as_mapping*/
	(hashfunc)NULL,			/*hash*/
	(ternaryfunc)NULL,		/*call*/
	(reprfunc)NULL,			/*str*/
};



static PyObject* PyEvent_New(SDL_Event* event)
{
	PyEventObject* e;

	e = PyObject_NEW(PyEventObject, &PyEvent_Type);

	if(e)
	{
		e->type = event->type;
		e->dict = dict_from_event(event);
	}
	return (PyObject*)e;
}

static PyObject* PyEvent_New2(int type, PyObject* dict)
{
	PyEventObject* e;
	e = PyObject_NEW(PyEventObject, &PyEvent_Type);
	if(e)
	{
		e->type = type;
		if(!dict)
			dict = PyDict_New();
		else
			Py_INCREF(dict);
		e->dict = dict;
	}
	return (PyObject*)e;
}



/* event module functions */


    /*DOC*/ static char doc_event[] =
    /*DOC*/    "pygame.event.event(type, dict) -> Event\n"
    /*DOC*/    "create new event object\n"
    /*DOC*/    "\n"
    /*DOC*/    "Creates a new event object. The type should be one\n"
    /*DOC*/    "of SDL's event numbers, or above USER_EVENT. The\n"
    /*DOC*/    "given dictionary contains a list of readonly\n"
    /*DOC*/    "attributes that will be members of the event\n"
    /*DOC*/    "object.\n"
    /*DOC*/ ;

static PyObject* event(PyObject* self, PyObject* arg)
{
	PyObject* dict;
	int type;
	if(!PyArg_ParseTuple(arg, "iO!", &type, &PyDict_Type, &dict))
		return NULL;
	return PyEvent_New2(type, dict);
}


    /*DOC*/ static char doc_event_name[] =
    /*DOC*/    "pygame.event.event_name(event type) -> string\n"
    /*DOC*/    "name for event type\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the standard SDL name for an event type.\n"
    /*DOC*/    "Mainly helpful for debugging, when trying to\n"
    /*DOC*/    "determine what the type of an event is.\n"
    /*DOC*/ ;

static PyObject* event_name(PyObject* self, PyObject* arg)
{
	int type;

	if(!PyArg_ParseTuple(arg, "i", &type))
		return NULL;

	return PyString_FromString(name_from_eventtype(type));
}



    /*DOC*/ static char doc_set_grab[] =
    /*DOC*/    "pygame.event.set_grab(bool) -> None\n"
    /*DOC*/    "grab all input events\n"
    /*DOC*/    "\n"
    /*DOC*/    "Grabs all mouse and keyboard input for the\n"
    /*DOC*/    "display. Grabbing the input is not neccessary to\n"
    /*DOC*/    "receive keyboard and mouse events, but it ensures\n"
    /*DOC*/    "all input will go to your application. It also\n"
    /*DOC*/    "keeps the mouse locked inside your window. Set the\n"
    /*DOC*/    "grabbing on or off with the boolean argument. It\n"
    /*DOC*/    "is best to not always grab the input, since it\n"
    /*DOC*/    "prevents the end user from doing anything else on\n"
    /*DOC*/    "their system.\n"
    /*DOC*/ ;

static PyObject* set_grab(PyObject* self, PyObject* arg)
{
	int doit;
	if(!PyArg_ParseTuple(arg, "i", &doit))
		return NULL;
	VIDEO_INIT_CHECK();

	if(doit)
		SDL_WM_GrabInput(SDL_GRAB_ON);
	else
		SDL_WM_GrabInput(SDL_GRAB_OFF);

	RETURN_NONE;
}


    /*DOC*/ static char doc_get_grab[] =
    /*DOC*/    "pygame.get_grab() -> bool\n"
    /*DOC*/    "query the state of input grabbing\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns true if the input is currently grabbed to\n"
    /*DOC*/    "your application.\n"
    /*DOC*/ ;

static PyObject* get_grab(PyObject* self, PyObject* arg)
{
	int mode;

	if(!PyArg_ParseTuple(arg, ""))
		return NULL;
	VIDEO_INIT_CHECK();

	mode = SDL_WM_GrabInput(SDL_GRAB_QUERY);

	return PyInt_FromLong(mode == SDL_GRAB_ON);
}



    /*DOC*/ static char doc_pump[] =
    /*DOC*/    "pygame.event.pump() -> None\n"
    /*DOC*/    "update the internal messages\n"
    /*DOC*/    "\n"
    /*DOC*/    "Pumping the message queue is important if you are not getting\n"
    /*DOC*/    "events off the message queue. The pump will allow pyGame to\n"
    /*DOC*/    "communicate with the window manager, which helps keep your\n"
    /*DOC*/    "application responsive, as well as updating the state for various\n"
    /*DOC*/    "input devices.\n"
    /*DOC*/ ;

static PyObject* pump(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	VIDEO_INIT_CHECK();

	SDL_PumpEvents();

	RETURN_NONE
}



    /*DOC*/ static char doc_wait[] =
    /*DOC*/    "pygame.event.wait() -> Event\n"
    /*DOC*/    "wait for an event\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the current event on the queue. If there are no messages\n"
    /*DOC*/    "waiting on the queue, this will not return until one is\n"
    /*DOC*/    "available. Sometimes it is important to use this wait to get\n"
    /*DOC*/    "events from the queue, it will allow your application to idle\n"
    /*DOC*/    "when the user isn't doing anything with it.\n"
    /*DOC*/ ;

static PyObject* wait(PyObject* self, PyObject* args)
{
	SDL_Event event;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	VIDEO_INIT_CHECK();

	if(!SDL_WaitEvent(&event))
		return RAISE(PyExc_SDLError, SDL_GetError());

	return PyEvent_New(&event);
}



    /*DOC*/ static char doc_poll[] =
    /*DOC*/    "pygame.event.poll() -> Event\n"
    /*DOC*/    "get an available event\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns next event on queue. If there is no event waiting on the\n"
    /*DOC*/    "queue, this will return an event with type NOEVENT.\n"
    /*DOC*/ ;

static PyObject* poll(PyObject* self, PyObject* args)
{
	SDL_Event event;
	
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	VIDEO_INIT_CHECK();

	if(!SDL_PollEvent(&event))
		return PyEvent_New2(SDL_NOEVENT, NULL);

	return PyEvent_New(&event);
}


    /*DOC*/ static char doc_get[] =
    /*DOC*/    "pygame.event.get([type]) -> list of Events\n"
    /*DOC*/    "get all of an event type from the queue\n"
    /*DOC*/    "\n"
    /*DOC*/    "Pass this a type of event that you are interested in, and it will\n"
    /*DOC*/    "return a list of all matching event types from the queue. If no\n"
    /*DOC*/    "types are passed, this will return all the events from the queue.\n"
    /*DOC*/    "You may also optionally pass a sequence of event types. For\n"
    /*DOC*/    "example, to fetch all the keyboard events from the queue, you\n"
    /*DOC*/    "would call, 'pygame.event.get([KEYDOWN,KEYUP])'.\n"
    /*DOC*/ ;

static PyObject* get(PyObject* self, PyObject* args)
{
	SDL_Event event;
	int mask = 0;
	int loop, num;
	PyObject* type, *list, *e;
	short val;

	if(PyTuple_Size(args) != 0 && PyTuple_Size(args) != 1)
		return RAISE(PyExc_ValueError, "peek requires 0 or 1 argument");

	VIDEO_INIT_CHECK();

	if(PyTuple_Size(args) == 0)
		mask = SDL_ALLEVENTS;
	else
	{
		type = PyTuple_GET_ITEM(args, 0);
		if(PySequence_Check(type))
		{
			num = PySequence_Size(type);
			for(loop=0; loop<num; ++loop)
			{
				if(!ShortFromObjIndex(type, loop, &val))
					return RAISE(PyExc_TypeError, "type sequence must contain valid event types");
				mask |= SDL_EVENTMASK(val);
			}
		}
		else if(ShortFromObj(type, &val))
			mask = SDL_EVENTMASK(val);
		else
			return RAISE(PyExc_TypeError, "peek type must be numeric or a sequence");
	}
	
	list = PyList_New(0);
	if(!list)
		return NULL;

	SDL_PumpEvents();

	while(SDL_PeepEvents(&event, 1, SDL_GETEVENT, mask) == 1)
	{
		e = PyEvent_New(&event);
		if(!e)
		{
			Py_DECREF(list);
			return NULL;
		}

		PyList_Append(list, e);
	}

	return list;
}


    /*DOC*/ static char doc_peek[] =
    /*DOC*/    "pygame.event.peek([type]) -> bool\n"
    /*DOC*/    "query if any of event types are waiting\n"
    /*DOC*/    "\n"
    /*DOC*/    "Pass this a type of event that you are interested in, and it will\n"
    /*DOC*/    "return true if there are any of that type of event on the queue.\n"
    /*DOC*/    "If no types are passed, this will return true if any events are\n"
    /*DOC*/    "on the queue. You may also optionally pass a sequence of event\n"
    /*DOC*/    "types. For example, to find if any keyboard events are on the\n"
    /*DOC*/    "queue, you would call, 'pygame.event.peek([KEYDOWN,KEYUP])'.\n"
    /*DOC*/ ;

static PyObject* peek(PyObject* self, PyObject* args)
{
	SDL_Event event;
	int result;
	int mask = 0;
	int loop, num;
	PyObject* type;
	short val;

	if(PyTuple_Size(args) != 0 && PyTuple_Size(args) != 1)
		return RAISE(PyExc_ValueError, "peek requires 0 or 1 argument");

	VIDEO_INIT_CHECK();

	if(PyTuple_Size(args) == 0)
		mask = SDL_ALLEVENTS;
	else
	{
		type = PyTuple_GET_ITEM(args, 0);
		if(PySequence_Check(type))
		{
			num = PySequence_Size(type);
			for(loop=0; loop<num; ++loop)
			{
				if(!ShortFromObjIndex(type, loop, &val))
					return RAISE(PyExc_TypeError, "type sequence must contain valid event types");
				mask |= SDL_EVENTMASK(val);
			}
		}
		else if(ShortFromObj(type, &val))
			mask = SDL_EVENTMASK(val);
		else
			return RAISE(PyExc_TypeError, "peek type must be numeric or a sequence");
	}
	
	SDL_PumpEvents();
	result = SDL_PeepEvents(&event, 1, SDL_PEEKEVENT, mask);
	return PyInt_FromLong(result == 1);
}



    /*DOC*/ static char doc_post[] =
    /*DOC*/    "pygame.event.post(Event) -> None\n"
    /*DOC*/    "place an event on the queue\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will place an event onto the queue. This is most useful for\n"
    /*DOC*/    "putting your own USEREVENT's onto the queue.\n"
    /*DOC*/ ;

static PyObject* post(PyObject* self, PyObject* args)
{
	PyEventObject* e;
	SDL_Event event;

	if(!PyArg_ParseTuple(args, "O!", &PyEvent_Type, &e))
		return NULL;

	VIDEO_INIT_CHECK();

	event.type = e->type;

	if(SDL_PushEvent(&event) == -1)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE
}


    /*DOC*/ static char doc_set_allowed[] =
    /*DOC*/    "pygame.event.set_allowed(type) -> None\n"
    /*DOC*/    "allows certain events onto the queue\n"
    /*DOC*/    "\n"
    /*DOC*/    "By default, all events will appear from the queue. After you have\n"
    /*DOC*/    "blocked some event types, you can use this to re-enable them. You\n"
    /*DOC*/    "can optionally pass a sequence of event types.\n"
    /*DOC*/ ;

static PyObject* set_allowed(PyObject* self, PyObject* args)
{
	int loop, num;
	PyObject* type;
	short val;

	if(PyTuple_Size(args) != 1)
		return RAISE(PyExc_ValueError, "set_allowed requires 1 argument");

	VIDEO_INIT_CHECK();

	type = PyTuple_GET_ITEM(args, 0);
	if(PySequence_Check(type))
	{
		num = PySequence_Length(type);
		for(loop=0; loop<num; ++loop)
		{
			if(!ShortFromObjIndex(type, loop, &val))
				return RAISE(PyExc_TypeError, "type sequence must contain valid event types");
			SDL_EventState((Uint8)val, SDL_ENABLE);
		}
	}
	else if(ShortFromObj(type, &val))
		SDL_EventState((Uint8)val, SDL_ENABLE);
	else
		return RAISE(PyExc_TypeError, "type must be numeric or a sequence");

	RETURN_NONE
}


    /*DOC*/ static char doc_set_blocked[] =
    /*DOC*/    "pygame.event.set_blocked(type) -> None\n"
    /*DOC*/    "blocks certain events from the queue\n"
    /*DOC*/    "\n"
    /*DOC*/    "By default, all events will appear from the queue. This will\n"
    /*DOC*/    "allow you to prevent event types from appearing on the queue. You\n"
    /*DOC*/    "can optionally pass a sequence of event types.\n"
    /*DOC*/ ;

static PyObject* set_blocked(PyObject* self, PyObject* args)
{
	int loop, num;
	PyObject* type;
	short val;

	if(PyTuple_Size(args) != 1)
		return RAISE(PyExc_ValueError, "set_blocked requires 1 argument");

	VIDEO_INIT_CHECK();

	type = PyTuple_GET_ITEM(args, 0);
	if(PySequence_Check(type))
	{
		num = PySequence_Length(type);
		for(loop=0; loop<num; ++loop)
		{
			if(!ShortFromObjIndex(type, loop, &val))
				return RAISE(PyExc_TypeError, "type sequence must contain valid event types");
			SDL_EventState((Uint8)val, SDL_IGNORE);
		}
	}
	else if(ShortFromObj(type, &val))
		SDL_EventState((Uint8)val, SDL_IGNORE);
	else
		return RAISE(PyExc_TypeError, "type must be numeric or a sequence");

	RETURN_NONE
}



static PyMethodDef event_builtins[] =
{
	{ "event", event, 1, doc_event },
	{ "event_name", event_name, 1, doc_event_name },

	{ "set_grab", set_grab, 1, doc_set_grab },
	{ "get_grab", get_grab, 1, doc_get_grab },

	{ "pump", pump, 1, doc_pump },
	{ "wait", wait, 1, doc_wait },
	{ "poll", poll, 1, doc_poll },
	{ "get", get, 1, doc_get },
	{ "peek", peek, 1, doc_peek },
	{ "post", post, 1, doc_post },

	{ "set_allowed", set_allowed, 1, doc_set_allowed },
	{ "set_blocked", set_blocked, 1, doc_set_blocked },

	{ NULL, NULL }
};



    /*DOC*/ static char doc_pygame_event_MODULE[] =
    /*DOC*/    "Contains event routines and object.\n"
    /*DOC*/ ;

void initevent()
{
	PyObject *module, *dict, *apiobj;
	static void* c_api[PYGAMEAPI_EVENT_NUMSLOTS];

	PyType_Init(PyEvent_Type);


    /* create the module */
	module = Py_InitModule3("event", event_builtins, doc_pygame_event_MODULE);
	dict = PyModule_GetDict(module);

	/* export the c api */
	c_api[0] = &PyEvent_Type;
	c_api[1] = PyEvent_New;
	apiobj = PyCObject_FromVoidPtr(c_api, NULL);
	PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);

	/*imported needed apis*/
	import_pygame_base();
}

