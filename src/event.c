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

/*
 *  pygame event module
 */
#define PYGAMEAPI_EVENT_INTERNAL
#include "pygame.h"


/*this user event object is for safely passing
 *objects through the event queue.
 */

#define USEROBJECT_CHECK1 0xDEADBEEF
#define USEROBJECT_CHECK2 0xFEEDF00D

typedef struct UserEventObject
{
	struct UserEventObject* next;
	PyObject* object;
}UserEventObject;

static UserEventObject* user_event_objects = NULL;


/*must pass dictionary as this object*/
static UserEventObject* user_event_addobject(PyObject* obj)
{
	UserEventObject* userobj = PyMem_New(UserEventObject, 1);
	if(!userobj) return NULL;

	Py_INCREF(obj);
	userobj->next = user_event_objects;
	userobj->object = obj;
	user_event_objects = userobj;

	return userobj;
}

/*note, we doublecheck to make sure the pointer is in our list,
 *not just some random pointer. this will keep us safe(r).
 */
static PyObject* user_event_getobject(UserEventObject* userobj)
{
	PyObject* obj = NULL;
	if(!user_event_objects) /*fail in most common case*/
		return NULL;
	if(user_event_objects == userobj)
	{
		obj = userobj->object;
		user_event_objects = userobj->next;
	}
	else
	{
		UserEventObject* hunt = user_event_objects;
		while(hunt && hunt->next != userobj)
			hunt = hunt->next;
		if(hunt)
		{
			hunt->next = userobj->next;
			obj = userobj->object;
		}
	}
	if(obj)
		PyMem_Del(userobj);
	return obj;
}


static void user_event_cleanup(void)
{
	if(user_event_objects)
	{
		UserEventObject *hunt, *kill;
		hunt = user_event_objects;
		while(hunt)
		{
			kill = hunt;
			hunt = hunt->next;
			Py_DECREF(kill->object);
			PyMem_Del(kill);
		}
		user_event_objects = NULL;
	}
}



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
	case SDL_KEYUP: 		return "KeyUp";
	case SDL_MOUSEMOTION:	return "MouseMotion";
	case SDL_MOUSEBUTTONDOWN:return "MouseButtonDown";
	case SDL_MOUSEBUTTONUP: return "MouseButtonUp";
	case SDL_JOYAXISMOTION: return "JoyAxisMotion";
	case SDL_JOYBALLMOTION: return "JoyBallMotion";
	case SDL_JOYHATMOTION:	return "JoyHatMotion";
	case SDL_JOYBUTTONUP:	return "JoyButtonUp";
	case SDL_JOYBUTTONDOWN: return "JoyButtonDown";
	case SDL_QUIT:			return "Quit";
	case SDL_SYSWMEVENT:	return "SysWMEvent";
	case SDL_VIDEORESIZE:	return "VideoResize";
	case SDL_VIDEOEXPOSE:	return "VideoExpose";
	case SDL_NOEVENT:		return "NoEvent";
	}
	if(type >= SDL_USEREVENT && type < SDL_NUMEVENTS)
		return "UserEvent";

	return "Unknown";
}


/* Helper for adding objects to dictionaries. Check for errors with
   PyErr_Occurred() */
static void insobj(PyObject *dict, char *name, PyObject *v)
{
	if(v)
	{
		PyDict_SetItemString(dict, name, v);
		Py_DECREF(v);
	}
}

#ifdef Py_USING_UNICODE

static PyObject* our_unichr(long uni)
{
	static PyObject* bltin_unichr = NULL;

	if (bltin_unichr == NULL) {
		PyObject* bltins;

		bltins = PyImport_ImportModule("__builtin__");
		bltin_unichr = PyObject_GetAttrString(bltins, "unichr");

		Py_DECREF(bltins);
	}

	return PyEval_CallFunction(bltin_unichr, "(l)", uni);
}

static PyObject* our_empty_ustr(void)
{
	static PyObject* empty_ustr = NULL;

	if (empty_ustr == NULL) {
		PyObject* bltins;
		PyObject* bltin_unicode;

		bltins = PyImport_ImportModule("__builtin__");
		bltin_unicode = PyObject_GetAttrString(bltins, "unicode");
		empty_ustr = PyEval_CallFunction(bltin_unicode, "(s)", "");

		Py_DECREF(bltin_unicode);
		Py_DECREF(bltins);
	}

	Py_INCREF(empty_ustr);

	return empty_ustr;
}

#else

static PyObject* our_unichr(long uni)
{
	return PyInt_FromLong(uni);
}

static PyObject* our_empty_ustr(void)
{
	return PyInt_FromLong(0);
}

#endif /* Py_USING_UNICODE */

static PyObject* dict_from_event(SDL_Event* event)
{
	PyObject *dict=NULL, *tuple, *obj;
	int hx, hy;

	/*check if it is an event the user posted*/
	if(event->user.code == USEROBJECT_CHECK1 && event->user.data1 == (void*)USEROBJECT_CHECK2)
	{
		dict = user_event_getobject((UserEventObject*)event->user.data2);
		if(dict)
			return dict;
	}

	if(!(dict = PyDict_New()))
		return NULL;
	switch(event->type)
	{
	case SDL_ACTIVEEVENT:
		insobj(dict, "gain", PyInt_FromLong(event->active.gain));
		insobj(dict, "state", PyInt_FromLong(event->active.state));
		break;
	case SDL_KEYDOWN:
		if(event->key.keysym.unicode)
			insobj(dict, "unicode", our_unichr(event->key.keysym.unicode));
		else
			insobj(dict, "unicode", our_empty_ustr());
	case SDL_KEYUP:
		insobj(dict, "key", PyInt_FromLong(event->key.keysym.sym));
		insobj(dict, "mod", PyInt_FromLong(event->key.keysym.mod));
		break;
	case SDL_MOUSEMOTION:
		obj = Py_BuildValue("(ii)", event->motion.x, event->motion.y);
		insobj(dict, "pos", obj);
		obj = Py_BuildValue("(ii)", event->motion.xrel, event->motion.yrel);
		insobj(dict, "rel", obj);
		if((tuple = PyTuple_New(3)))
		{
			PyTuple_SET_ITEM(tuple, 0, PyInt_FromLong((event->motion.state&SDL_BUTTON(1)) != 0));
			PyTuple_SET_ITEM(tuple, 1, PyInt_FromLong((event->motion.state&SDL_BUTTON(2)) != 0));
			PyTuple_SET_ITEM(tuple, 2, PyInt_FromLong((event->motion.state&SDL_BUTTON(3)) != 0));
			insobj(dict, "buttons", tuple);
		}
		break;
	case SDL_MOUSEBUTTONDOWN:
	case SDL_MOUSEBUTTONUP:
		obj = Py_BuildValue("(ii)", event->button.x, event->button.y);
		insobj(dict, "pos", obj);
		insobj(dict, "button", PyInt_FromLong(event->button.button));
		break;
	case SDL_JOYAXISMOTION:
		insobj(dict, "joy", PyInt_FromLong(event->jaxis.which));
		insobj(dict, "axis", PyInt_FromLong(event->jaxis.axis));
		insobj(dict, "value", PyFloat_FromDouble(event->jaxis.value/32767.0));
		break;
	case SDL_JOYBALLMOTION:
		insobj(dict, "joy", PyInt_FromLong(event->jball.which));
		insobj(dict, "ball", PyInt_FromLong(event->jball.ball));
		obj = Py_BuildValue("(ii)", event->jball.xrel, event->jball.yrel);
		insobj(dict, "rel", obj);
		break;
	case SDL_JOYHATMOTION:
		insobj(dict, "joy", PyInt_FromLong(event->jhat.which));
		insobj(dict, "hat", PyInt_FromLong(event->jhat.hat));
		hx = hy = 0;
		if(event->jhat.value&SDL_HAT_UP) hy = 1;
		else if(event->jhat.value&SDL_HAT_DOWN) hy = -1;
		if(event->jhat.value&SDL_HAT_RIGHT) hx = 1;
		else if(event->jhat.value&SDL_HAT_LEFT) hx = -1;
		insobj(dict, "value", Py_BuildValue("(ii)", hx, hy));
		break;
	case SDL_JOYBUTTONUP:
	case SDL_JOYBUTTONDOWN:
		insobj(dict, "joy", PyInt_FromLong(event->jbutton.which));
		insobj(dict, "button", PyInt_FromLong(event->jbutton.button));
		break;
	case SDL_VIDEORESIZE:
		obj = Py_BuildValue("(ii)", event->resize.w, event->resize.h);
		insobj(dict, "size", obj);
		break;
/* SDL_VIDEOEXPOSE and SDL_QUIT have no attributes */
	}
	if(event->type >= SDL_USEREVENT && event->type < SDL_NUMEVENTS)
	{
		insobj(dict, "code", PyInt_FromLong(event->user.code));
	}

	return dict;
}




/* event object internals */

static void event_dealloc(PyObject* self)
{
	PyEventObject* e = (PyEventObject*)self;
	Py_XDECREF(e->dict);
	PyObject_DEL(self);
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


/*this fools the docs to putting more in our docs, but not the code*/
#define __SECRET_COLON__ ; char *docdata=

    /*DOC*/ static char doc_pygame_event_EXTRA[] =
    /*DOC*/    "An Event object contains an event type and a readonly set of\n"
    /*DOC*/    "member data. The Event object contains no method functions, just\n"
    /*DOC*/    "member data. Event objects are retrieved from the pygame event\n"
    /*DOC*/    "queue. You can create your own new events with the\n"
    /*DOC*/    "pygame.event.Event() function.\n"
    /*DOC*/    "\n"
    /*DOC*/    "All Event objects contain an event type identifier in the\n"
    /*DOC*/    "Event.type member. You may also get full access to the Event's\n"
    /*DOC*/    "member data through the Event.dict method. All other member\n"
    /*DOC*/    "lookups will be passed through to the Event's dictionary values.\n"
    /*DOC*/    "\n"
    /*DOC*/    "While debugging and experimenting, you can print the Event\n"
    /*DOC*/    "objects for a quick display of its type and members.\n" __SECRET_COLON__
    /*DOC*/    "Events that come from the system will have a guaranteed set of\n"
    /*DOC*/    "member items based on the type. Here is a list of the Event members\n"
    /*DOC*/    "that are defined with each type.<br><table align=center>"
    /*DOC*/    "<tr><td><b>QUIT</b></td><td><i>none</i></td></tr>\n"
    /*DOC*/    "<tr><td><b>ACTIVEEVENT</b></td><td>gain, state</td></tr>\n"
    /*DOC*/    "<tr><td><b>KEYDOWN</b></td><td>unicode, key, mod</td></tr>\n"
    /*DOC*/    "<tr><td><b>KEYUP</b></td><td>key, mod</td></tr>\n"
    /*DOC*/    "<tr><td><b>MOUSEMOTION</b></td><td>pos, rel, buttons</td></tr>\n"
    /*DOC*/    "<tr><td><b>MOUSEBUTTONUP</b></td><td>pos, button</td></tr>\n"
    /*DOC*/    "<tr><td><b>MOUSEBUTTONDOWN</b></td><td>pos, button</td></tr>\n"
    /*DOC*/    "<tr><td><b>JOYAXISMOTION</b></td><td>joy, axis, value</td></tr>\n"
    /*DOC*/    "<tr><td><b>JOYBALLMOTION</b></td><td>joy, ball, rel</td></tr>\n"
    /*DOC*/    "<tr><td><b>JOYHATMOTION</b></td><td>joy, hat, value</td></tr>\n"
    /*DOC*/    "<tr><td><b>JOYBUTTONUP</b></td><td>joy, button</td></tr>\n"
    /*DOC*/    "<tr><td><b>JOYBUTTONDOWN</b></td><td>joy, button</td></tr>\n"
    /*DOC*/    "<tr><td><b>VIDEORESIZE</b></td><td>size</td></tr>\n"
    /*DOC*/    "<tr><td><b>VIDEOEXPOSE</b></td><td><i>none</i></td></tr>\n"
    /*DOC*/    "<tr><td><b>USEREVENT</b></td><td>code</td></tr></table>\n"
    /*DOC*/ ;

static int event_nonzero(PyEventObject *self)
{
	return self->type != SDL_NOEVENT;
}

static PyNumberMethods event_as_number = {
	(binaryfunc)NULL,		/*add*/
	(binaryfunc)NULL,		/*subtract*/
	(binaryfunc)NULL,		/*multiply*/
	(binaryfunc)NULL,		/*divide*/
	(binaryfunc)NULL,		/*remainder*/
	(binaryfunc)NULL,		/*divmod*/
	(ternaryfunc)NULL,		/*power*/
	(unaryfunc)NULL,		/*negative*/
	(unaryfunc)NULL,		/*pos*/
	(unaryfunc)NULL,		/*abs*/
	(inquiry)event_nonzero,	/*nonzero*/
	(unaryfunc)NULL,		/*invert*/
	(binaryfunc)NULL,		/*lshift*/
	(binaryfunc)NULL,		/*rshift*/
	(binaryfunc)NULL,		/*and*/
	(binaryfunc)NULL,		/*xor*/
	(binaryfunc)NULL,		/*or*/
	(coercion)NULL,			/*coerce*/
	(unaryfunc)NULL,		/*int*/
	(unaryfunc)NULL,		/*long*/
	(unaryfunc)NULL,		/*float*/
	(unaryfunc)NULL,		/*oct*/
	(unaryfunc)NULL,		/*hex*/
};


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
	&event_as_number,		/*as_number*/
	NULL,					/*as_sequence*/
	NULL,					/*as_mapping*/
	(hashfunc)NULL, 		/*hash*/
	(ternaryfunc)NULL,		/*call*/
	(reprfunc)NULL, 		/*str*/
	0L,0L,0L,0L,
	doc_pygame_event_EXTRA /* Documentation string */
};



static PyObject* PyEvent_New(SDL_Event* event)
{
	PyEventObject* e;
	e = PyObject_NEW(PyEventObject, &PyEvent_Type);
	if(!e) return NULL;

	if(event)
	{
		e->type = event->type;
		e->dict = dict_from_event(event);
	}
	else
	{
		e->type = SDL_NOEVENT;
		e->dict = PyDict_New();
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


    /*DOC*/ static char doc_Event[] =
    /*DOC*/    "pygame.event.Event(type, [dict], [keyword_args]) -> Event\n"
    /*DOC*/    "create new event object\n"
    /*DOC*/    "\n"
    /*DOC*/    "Creates a new event object. The type should be one of SDL's\n"
    /*DOC*/    "event numbers, or above USEREVENT. The given dictionary contains\n"
    /*DOC*/    "the keys that will be members of the new event.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Also, instead of passing a dictionary to create the event\n"
    /*DOC*/    "members, you can pass keyword arguments that will become the\n"
    /*DOC*/    "attributes of the new event.\n"
    /*DOC*/ ;

static PyObject* Event(PyObject* self, PyObject* arg, PyObject* keywords)
{
	PyObject* dict = NULL;
	PyObject* event;
	int type;
	if(!PyArg_ParseTuple(arg, "i|O!", &type, &PyDict_Type, &dict))
		return NULL;

	if(!dict)
		dict = PyDict_New();
	else
		Py_INCREF(dict);

	if(keywords)
	{
		PyObject *key, *value;
		int pos  = 0;
		while(PyDict_Next(keywords, &pos, &key, &value))
			PyDict_SetItem(dict, key, value);
	}

	event = PyEvent_New2(type, dict);

	Py_DECREF(dict);
	return event;
}


    /*DOC*/ static char doc_event_name[] =
    /*DOC*/    "pygame.event.event_name(event type) -> string\n"
    /*DOC*/    "name for event type\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the standard SDL name for an event type. Mainly helpful\n"
    /*DOC*/    "for debugging, when trying to determine what the type of an event\n"
    /*DOC*/    "is.\n"
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
    /*DOC*/    "Grabs all mouse and keyboard input for the display. Grabbing the\n"
    /*DOC*/    "input is not neccessary to receive keyboard and mouse events, but\n"
    /*DOC*/    "it ensures all input will go to your application. It also keeps\n"
    /*DOC*/    "the mouse locked inside your window. Set the grabbing on or off\n"
    /*DOC*/    "with the boolean argument. It is best to not always grab the\n"
    /*DOC*/    "input, since it prevents the end user from doing anything else on\n"
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
    /*DOC*/    "pygame.event.get_grab() -> bool\n"
    /*DOC*/    "query the state of input grabbing\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns true if the input is currently grabbed to your\n"
    /*DOC*/    "application.\n"
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
    /*DOC*/    "For each frame of your game, you will need to make some sort\n"
    /*DOC*/    "of call to the event queue. This ensures your program can internally\n"
    /*DOC*/    "interact with the rest of the operating system. If you are not using\n"
    /*DOC*/    "other event functions in your game, you should call pump() to allow\n"
    /*DOC*/    "pygame to handle internal actions.\n"
    /*DOC*/    "\n"
    /*DOC*/    "There are important things that must be dealt with internally in the\n"
    /*DOC*/    "event queue. The main window may need to be repainted. Certain joysticks\n"
    /*DOC*/    "must be polled for their values. If you fail to make a call to the event\n"
    /*DOC*/    "queue for too long, the system may decide your program has locked up.\n"
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

static PyObject* pygame_wait(PyObject* self, PyObject* args)
{
	SDL_Event event;
	int status;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	VIDEO_INIT_CHECK();

	Py_BEGIN_ALLOW_THREADS
	status = SDL_WaitEvent(&event);
	Py_END_ALLOW_THREADS

	if(!status)
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

	if(SDL_PollEvent(&event))
		return PyEvent_New(&event);
	return PyEvent_New(NULL);
}


    /*DOC*/ static char doc_event_clear[] =
    /*DOC*/    "pygame.event.clear([type]) -> None\n"
    /*DOC*/    "remove all of an event type from the queue\n"
    /*DOC*/    "\n"
    /*DOC*/    "Pass this a type of event to discard, and it will\n"
    /*DOC*/    "remove all matching event types from the queue. If no\n"
    /*DOC*/    "types are passed, this will remove all the events from the queue.\n"
    /*DOC*/    "You may also optionally pass a sequence of event types. For\n"
    /*DOC*/    "example, to remove all the mouse motion events from the queue, you\n"
    /*DOC*/    "would call, 'pygame.event.clear(MOUSEMOTION)'.\n"
    /*DOC*/ ;

static PyObject* event_clear(PyObject* self, PyObject* args)
{
	SDL_Event event;
	int mask = 0;
	int loop, num;
	PyObject* type;
	int val;

	if(PyTuple_Size(args) != 0 && PyTuple_Size(args) != 1)
		return RAISE(PyExc_ValueError, "get requires 0 or 1 argument");

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
				if(!IntFromObjIndex(type, loop, &val))
					return RAISE(PyExc_TypeError, "type sequence must contain valid event types");
				mask |= SDL_EVENTMASK(val);
			}
		}
		else if(IntFromObj(type, &val))
			mask = SDL_EVENTMASK(val);
		else
			return RAISE(PyExc_TypeError, "get type must be numeric or a sequence");
	}

	SDL_PumpEvents();

	while(SDL_PeepEvents(&event, 1, SDL_GETEVENT, mask) == 1)
	{}

	RETURN_NONE;
}


    /*DOC*/ static char doc_event_get[] =
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

static PyObject* event_get(PyObject* self, PyObject* args)
{
	SDL_Event event;
	int mask = 0;
	int loop, num;
	PyObject* type, *list, *e;
	int val;

	if(PyTuple_Size(args) != 0 && PyTuple_Size(args) != 1)
		return RAISE(PyExc_ValueError, "get requires 0 or 1 argument");

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
				if(!IntFromObjIndex(type, loop, &val))
					return RAISE(PyExc_TypeError, "type sequence must contain valid event types");
				mask |= SDL_EVENTMASK(val);
			}
		}
		else if(IntFromObj(type, &val))
			mask = SDL_EVENTMASK(val);
		else
			return RAISE(PyExc_TypeError, "get type must be numeric or a sequence");
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
		Py_DECREF(e);
	}

	return list;
}


    /*DOC*/ static char doc_peek[] =
    /*DOC*/    "pygame.event.peek([type]) -> bool\n"
    /*DOC*/    "query if any of event types are waiting\n"
    /*DOC*/    "\n"
    /*DOC*/    "Pass this a type of event that you are interested in, and it will\n"
    /*DOC*/    "return true if there are any of that type of event on the queue.\n"
    /*DOC*/    "If no types are passed, this will return the next event on the queue\n"
    /*DOC*/    "without removing it. You may also optionally pass a sequence of event\n"
    /*DOC*/    "types. For example, to find if any keyboard events are on the\n"
    /*DOC*/    "queue, you would call, 'pygame.event.peek([KEYDOWN,KEYUP])'.\n"
    /*DOC*/ ;

static PyObject* event_peek(PyObject* self, PyObject* args)
{
	SDL_Event event;
	int result;
	int mask = 0;
	int loop, num, noargs=0;
	PyObject* type;
	int val;

	if(PyTuple_Size(args) != 0 && PyTuple_Size(args) != 1)
		return RAISE(PyExc_ValueError, "peek requires 0 or 1 argument");

	VIDEO_INIT_CHECK();

	if(PyTuple_Size(args) == 0)
	{
		mask = SDL_ALLEVENTS;
		noargs=1;
	}
	else
	{
		type = PyTuple_GET_ITEM(args, 0);
		if(PySequence_Check(type))
		{
			num = PySequence_Size(type);
			for(loop=0; loop<num; ++loop)
			{
				if(!IntFromObjIndex(type, loop, &val))
					return RAISE(PyExc_TypeError, "type sequence must contain valid event types");
				mask |= SDL_EVENTMASK(val);
			}
		}
		else if(IntFromObj(type, &val))
			mask = SDL_EVENTMASK(val);
		else
			return RAISE(PyExc_TypeError, "peek type must be numeric or a sequence");
	}

	SDL_PumpEvents();
	result = SDL_PeepEvents(&event, 1, SDL_PEEKEVENT, mask);

	if(noargs)
		return PyEvent_New(&event);
	return PyInt_FromLong(result == 1);
}



    /*DOC*/ static char doc_post[] =
    /*DOC*/    "pygame.event.post(Event) -> None\n"
    /*DOC*/    "place an event on the queue\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will post your own event objects onto the event queue.\n"
    /*DOC*/    "You can past any event type you want, but some care must be\n"
    /*DOC*/    "taken. For example, if you post a MOUSEBUTTONDOWN event to the\n"
    /*DOC*/    "queue, it is likely any code receiving the event will excpect\n"
    /*DOC*/    "the standard MOUSEBUTTONDOWN attributes to be available, like\n"
    /*DOC*/    "'pos' and 'button'.\n"
    /*DOC*/ ;

static PyObject* event_post(PyObject* self, PyObject* args)
{
	PyEventObject* e;
	SDL_Event event;
	UserEventObject* userobj;

	if(!PyArg_ParseTuple(args, "O!", &PyEvent_Type, &e))
		return NULL;

	VIDEO_INIT_CHECK();

	userobj = user_event_addobject(e->dict);
	if(!userobj)
		return NULL;

	event.type = e->type;
	event.user.code = USEROBJECT_CHECK1;
	event.user.data1 = (void*)USEROBJECT_CHECK2;
	event.user.data2 = userobj;

	if(SDL_PushEvent(&event) == -1)
		return RAISE(PyExc_SDLError, "Event queue full");

	RETURN_NONE
}


    /*DOC*/ static char doc_set_allowed[] =
    /*DOC*/    "pygame.event.set_allowed(type) -> None\n"
    /*DOC*/    "allows certain events onto the queue\n"
    /*DOC*/    "\n"
    /*DOC*/    "By default, all events will appear from the queue. After you have\n"
    /*DOC*/    "blocked some event types, you can use this to re-enable them. You\n"
    /*DOC*/    "can optionally pass a sequence of event types.\n"
    /*DOC*/    "\n"
    /*DOC*/    "You can pass None and this will allow no events on the queue.\n"
    /*DOC*/ ;

static PyObject* set_allowed(PyObject* self, PyObject* args)
{
	int loop, num;
	PyObject* type;
	int val;

	if(PyTuple_Size(args) != 1)
		return RAISE(PyExc_ValueError, "set_allowed requires 1 argument");

	VIDEO_INIT_CHECK();

	type = PyTuple_GET_ITEM(args, 0);
	if(PySequence_Check(type))
	{
		num = PySequence_Length(type);
		for(loop=0; loop<num; ++loop)
		{
			if(!IntFromObjIndex(type, loop, &val))
				return RAISE(PyExc_TypeError, "type sequence must contain valid event types");
			SDL_EventState((Uint8)val, SDL_ENABLE);
		}
	}
	else if(type == Py_None)
		SDL_EventState((Uint8)0xFF, SDL_IGNORE);
	else if(IntFromObj(type, &val))
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
    /*DOC*/    "\n"
    /*DOC*/    "You can pass None and this will allow all events on the queue.\n"
    /*DOC*/ ;

static PyObject* set_blocked(PyObject* self, PyObject* args)
{
	int loop, num;
	PyObject* type;
	int val;

	if(PyTuple_Size(args) != 1)
		return RAISE(PyExc_ValueError, "set_blocked requires 1 argument");

	VIDEO_INIT_CHECK();

	type = PyTuple_GET_ITEM(args, 0);
	if(PySequence_Check(type))
	{
		num = PySequence_Length(type);
		for(loop=0; loop<num; ++loop)
		{
			if(!IntFromObjIndex(type, loop, &val))
				return RAISE(PyExc_TypeError, "type sequence must contain valid event types");
			SDL_EventState((Uint8)val, SDL_IGNORE);
		}
	}
	else if(type == Py_None)
		SDL_EventState((Uint8)0, SDL_IGNORE);
	else if(IntFromObj(type, &val))
		SDL_EventState((Uint8)val, SDL_IGNORE);
	else
		return RAISE(PyExc_TypeError, "type must be numeric or a sequence");

	RETURN_NONE
}


    /*DOC*/ static char doc_get_blocked[] =
    /*DOC*/    "pygame.event.get_blocked(type) -> boolean\n"
    /*DOC*/    "checks if an event is being blocked\n"
    /*DOC*/    "\n"
    /*DOC*/    "This returns a true value if the given event type is being blocked\n"
    /*DOC*/    "from the queue. You can optionally pass a sequence of event types,\n"
    /*DOC*/    "and it will return TRUE if any of the types are blocked.\n"
    /*DOC*/ ;

static PyObject* get_blocked(PyObject* self, PyObject* args)
{
	int loop, num;
	PyObject* type;
	int val;
	int isblocked = 0;

	if(PyTuple_Size(args) != 1)
		return RAISE(PyExc_ValueError, "set_blocked requires 1 argument");

	VIDEO_INIT_CHECK();

	type = PyTuple_GET_ITEM(args, 0);
	if(PySequence_Check(type))
	{
		num = PySequence_Length(type);
		for(loop=0; loop<num; ++loop)
		{
			if(!IntFromObjIndex(type, loop, &val))
				return RAISE(PyExc_TypeError, "type sequence must contain valid event types");
			isblocked |= SDL_EventState((Uint8)val, SDL_QUERY) == SDL_IGNORE;
		}
	}
	else if(IntFromObj(type, &val))
		isblocked = SDL_EventState((Uint8)val, SDL_QUERY) == SDL_IGNORE;
	else
		return RAISE(PyExc_TypeError, "type must be numeric or a sequence");

	return PyInt_FromLong(isblocked);
}



static PyMethodDef event_builtins[] =
{
	{ "Event", (PyCFunction)Event, 3, doc_Event },
	{ "event_name", event_name, 1, doc_event_name },

	{ "set_grab", set_grab, 1, doc_set_grab },
	{ "get_grab", get_grab, 1, doc_get_grab },

	{ "pump", pump, 1, doc_pump },
	{ "wait", pygame_wait, 1, doc_wait },
	{ "poll", poll, 1, doc_poll },
	{ "clear", event_clear, 1, doc_event_clear },
	{ "get", event_get, 1, doc_event_get },
	{ "peek", event_peek, 1, doc_peek },
	{ "post", event_post, 1, doc_post },

	{ "set_allowed", set_allowed, 1, doc_set_allowed },
	{ "set_blocked", set_blocked, 1, doc_set_blocked },
	{ "get_blocked", get_blocked, 1, doc_get_blocked },

	{ NULL, NULL }
};



    /*DOC*/ static char doc_pygame_event_MODULE[] =
    /*DOC*/    "Pygame handles all it's event messaging through an event queue.\n"
    /*DOC*/    "The routines in this module help you manage that event queue. The\n"
    /*DOC*/    "input queue is heavily dependent on the pygame display module. If\n"
    /*DOC*/    "the display has not been initialized and a video mode not set,\n"
    /*DOC*/    "the event queue will not really work.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The queue is a stack of Event objects, there are a variety of\n"
    /*DOC*/    "ways to access the data on the queue. From simply checking for\n"
    /*DOC*/    "the existance of events, to grabbing them directly off the stack.\n"
    /*DOC*/    "\n"
    /*DOC*/    "All events have a type identifier. This event type is in between\n"
    /*DOC*/    "the values of NOEVENT and NUMEVENTS. All user defined events can\n"
    /*DOC*/    "have the value of USEREVENT or higher. It is recommended make\n"
    /*DOC*/    "sure your event id's follow this system.\n"
    /*DOC*/    "\n"
    /*DOC*/    "To get the state of various input devices, you can forego the\n"
    /*DOC*/    "event queue and access the input devices directly with their\n"
    /*DOC*/    "appropriate modules; mouse, key, and joystick. If you use this\n"
    /*DOC*/    "method, remember that pygame requires some form of communication\n"
    /*DOC*/    "with the system window manager and other parts of the platform.\n"
    /*DOC*/    "To keep pygame in synch with the system, you will need to call\n"
    /*DOC*/    "pygame.event.pump() to keep everything current. You'll want to\n"
    /*DOC*/    "call this function usually once per game loop.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The event queue offers some simple filtering. This can help\n"
    /*DOC*/    "performance slightly by blocking certain event types from the\n"
    /*DOC*/    "queue, use the pygame.event.set_allowed() and\n"
    /*DOC*/    "pygame.event.set_blocked() to work with this filtering. All\n"
    /*DOC*/    "events default to allowed.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Also know that you will not receive any events from a joystick\n"
    /*DOC*/    "device, until you have initialized that individual joystick from\n"
    /*DOC*/    "the joystick module.\n"
    /*DOC*/ ;

PYGAME_EXPORT
void initevent(void)
{
	PyObject *module, *dict, *apiobj;
	static void* c_api[PYGAMEAPI_EVENT_NUMSLOTS];

	PyType_Init(PyEvent_Type);


    /* create the module */
	module = Py_InitModule3("event", event_builtins, doc_pygame_event_MODULE);
	dict = PyModule_GetDict(module);

	PyDict_SetItemString(dict, "EventType", (PyObject *)&PyEvent_Type);

	/* export the c api */
	c_api[0] = &PyEvent_Type;
	c_api[1] = PyEvent_New;
	apiobj = PyCObject_FromVoidPtr(c_api, NULL);
	PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
	Py_DECREF(apiobj);

	/*imported needed apis*/
	import_pygame_base();
	PyGame_RegisterQuit(user_event_cleanup);
}



