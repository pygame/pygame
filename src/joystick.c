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

#define PYGAMEAPI_JOYSTICK_INTERNAL
#include "pygame.h"




staticforward PyTypeObject Joystick_Type;
static PyObject* PyJoystick_New(SDL_Joystick*);
#define PyJoystick_Check(x) ((x)->ob_type == &PyCD_Type)



static void joy_autoquit()
{
	if(SDL_WasInit(SDL_INIT_JOYSTICK))
	{
		SDL_JoystickEventState(SDL_DISABLE);
		SDL_QuitSubSystem(SDL_INIT_JOYSTICK);
	}
}

static PyObject* joy_autoinit(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	if(!SDL_WasInit(SDL_INIT_JOYSTICK))
	{
		if(SDL_InitSubSystem(SDL_INIT_JOYSTICK))
			return PyInt_FromLong(0);
		SDL_JoystickEventState(SDL_ENABLE);
		PyGame_RegisterQuit(cdrom_autoquit);
	}
	return PyInt_FromLong(1);
}


    /*DOC*/ static char doc_joy_quit[] =
    /*DOC*/    "pygame.joystick.quit() -> None\n"
    /*DOC*/    "uninitialize cdrom module\n"
    /*DOC*/    "\n"
    /*DOC*/    "Uninitialize the joystick module manually\n"
    /*DOC*/ ;

static PyObject* joy_quit(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	joy_autoquit();

	RETURN_NONE
}

    /*DOC*/ static char doc_joy_init[] =
    /*DOC*/    "pygame.joystick.init() -> None\n"
    /*DOC*/    "initialize cdrom module\n"
    /*DOC*/    "\n"
    /*DOC*/    "Initialize the joystick module manually\n"
    /*DOC*/ ;

static PyObject* joy_init(PyObject* self, PyObject* arg)
{
	PyObject* result;
	int istrue;

	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	result = joy_autoinit(self, arg);
	istrue = PyObject_IsTrue(result);
	Py_DECREF(result);
	if(!istrue)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE
}



    /*DOC*/ static char doc_get_init[] =
    /*DOC*/    "pygame.joystick.get_init() -> bool\n"
    /*DOC*/    "query initialization of joystick module\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns true when the joystick module is\n"
    /*DOC*/    "initialized.\n"
    /*DOC*/ ;

static PyObject* get_init(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	return PyInt_FromLong(SDL_WasInit(SDL_INIT_JOYSTICK)!=0);
}



/*joystick object funcs*/


static void joy_dealloc(PyObject* self)
{
	PyJoystickObject* joy_ref = (PyJoystickObject*)self;

	if(SDL_WasInit(SDL_INIT_JOYSTICK)
		SDL_JoystickClose(joy_ref->joy);

	PyMem_DEL(self);
}


    /*DOC*/ static char doc_joy_get_id[] =
    /*DOC*/    "Joystick.get_id() -> id\n"
    /*DOC*/    "query id of joystick\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the joystick id number for the Joystick\n"
    /*DOC*/ ;

static PyObject* joy_get_id(PyObject* self, PyObject* args)
{
	PyJoystickObject* joy_ref = (PyJoystickObject*)self;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong(SDL_JoystickIndex(joy_ref->joy));
}



    /*DOC*/ static char doc_joy_get_axes[] =
    /*DOC*/    "Joystick.get_axes() -> count\n"
    /*DOC*/    "query number of axis\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the number of axis on this Joystick\n"
    /*DOC*/ ;

static PyObject* joy_get_axes(PyObject* self, PyObject* args)
{
	PyJoystickObject* joy_ref = (PyJoystickObject*)self;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong(SDL_JoystickNumAxes(joy_ref->joy));
}



    /*DOC*/ static char doc_joy_get_balls[] =
    /*DOC*/    "Joystick.get_balls() -> count\n"
    /*DOC*/    "query number of balls\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns number of trackballs on this Joystick\n"
    /*DOC*/ ;

static PyObject* joy_get_balls(PyObject* self, PyObject* args)
{
	PyJoystickObject* joy_ref = (PyJoystickObject*)self;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong(SDL_JoystickNumBalls(joy_ref->joy));
}



    /*DOC*/ static char doc_joy_get_hats[] =
    /*DOC*/    "Joystick.get_hats() -> count\n"
    /*DOC*/    "query number of hats\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns number of directional hats on this\n"
    /*DOC*/    "Joystick\n"
    /*DOC*/ ;

static PyObject* joy_get_hats(PyObject* self, PyObject* args)
{
	PyJoystickObject* joy_ref = (PyJoystickObject*)self;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong(SDL_JoystickNumHats(joy_ref->joy));
}



    /*DOC*/ static char doc_joy_get_buttons[] =
    /*DOC*/    "Joystick.get_buttons() -> count\n"
    /*DOC*/    "query number of buttons\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns number of pushable buttons on this\n"
    /*DOC*/    "Joystick\n"
    /*DOC*/ ;

static PyObject* joy_get_buttons(PyObject* self, PyObject* args)
{
	PyJoystickObject* joy_ref = (PyJoystickObject*)self;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong(SDL_JoystickNumButtons(joy_ref->joy));
}



    /*DOC*/ static char doc_joy_get_axis[] =
    /*DOC*/    "Joystick.get_axis(axis) -> position\n"
    /*DOC*/    "query axis of a joystick\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the current position of an axis control on\n"
    /*DOC*/    "the Joystick. Value is in the range -1.0 to 1.0.\n"
    /*DOC*/ ;

static PyObject* joy_get_axis(PyObject* self, PyObject* args)
{
	PyJoystickObject* joy_ref = (PyJoystickObject*)self;
	int axis;

	if(!PyArg_ParseTuple(args, "i", &axis))
		return NULL;

	return PyFloat_FromDouble(SDL_JoystickGetAxis(joy_ref->joy, axis)/32767.0);
}



    /*DOC*/ static char doc_joy_get_hat[] =
    /*DOC*/    "Joystick.get_hat(pov_hat) -> state\n"
    /*DOC*/    "query position of hat\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the current position of a directional hat\n"
    /*DOC*/    "on the Joystick. Value in a position on the\n"
    /*DOC*/    "following compass. (think 1 is up, and goes around\n"
    /*DOC*/    "clockwise)\n"
    /*DOC*/    "8 1 2\n"
    /*DOC*/    "7 0 3\n"
    /*DOC*/    "6 5 4\n"
    /*DOC*/ ;

static PyObject* joy_get_hat(PyObject* self, PyObject* args)
{
	PyJoystickObject* joy_ref = (PyJoystickObject*)self;
	int hat;

	if(!PyArg_ParseTuple(args, "i", &hat))
		return NULL;

	return PyInt_FromLong(SDL_JoystickGetHat(joy_ref->joy, hat));
}



    /*DOC*/ static char doc_joy_get_button[] =
    /*DOC*/    "Joystick.get_button(button) -> bool\n"
    /*DOC*/    "query state of button\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns true if the given Joystick button is being\n"
    /*DOC*/    "pressed.\n"
    /*DOC*/ ;

static PyObject* joy_get_button(PyObject* self, PyObject* args)
{
	PyJoystickObject* joy_ref = (PyJoystickObject*)self;
	int button;

	if(!PyArg_ParseTuple(args, "i", &button))
		return NULL;

	return PyInt_FromLong(SDL_JoystickGetButton(joy_ref->joy, button));
}



/*joystick module funcs*/

static PyObject* PyJoystick_New(SDL_Joystick* joy)
{
	PyJoystickObject* joyobj;

	if(!joy)
		return RAISE(PyExc_SDLError, SDL_GetError());

	joyobj = PyObject_NEW(PyJoystickObject, &Joystick_Type);
	if(!joyobj)
		return NULL;

	joyobj->joy = joy;

	return (PyObject*)joyobj;
}



    /*DOC*/ static char doc_joy_open[] =
    /*DOC*/    "pygame.joystick.open(id) -> Joystick\n"
    /*DOC*/    "return new joystick object\n"
    /*DOC*/    "\n"
    /*DOC*/    "Creates a new joystick object for the given\n"
    /*DOC*/    "joystick id. Once a joystick has been opened, it\n"
    /*DOC*/    "will start receiving joystick events on the event\n"
    /*DOC*/    "queue.\n"
    /*DOC*/ ;

static PyObject* joy_open(PyObject* self, PyObject* args)
{
	int id;

	if(!PyArg_ParseTuple(args, "i", &id))
		return NULL;

	JOY_INIT_CHECK

	return PyJoystick_New(SDL_JoystickOpen(id));
}



    /*DOC*/ static char doc_joy_get_count[] =
    /*DOC*/    "pygame.joystick.get_count() -> int\n"
    /*DOC*/    "number of joysticks in system\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the number of joysticks available on the\n"
    /*DOC*/    "system. Will be 0 if there are no joysticks.\n"
    /*DOC*/ ;

static PyObject* joy_get_count(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	JOY_INIT_CHECK

	return PyInt_FromLong(SDL_NumJoysticks());
}



    /*DOC*/ static char doc_joy_get_name[] =
    /*DOC*/    "pygame.joystick.get_name(id) -> string\n"
    /*DOC*/    "system name for joystick\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns a readable name for the joystick device,\n"
    /*DOC*/    "given by the system.\n"
    /*DOC*/ ;

static PyObject* joy_get_name(PyObject* self, PyObject* args)
{
	int id;

	if(!PyArg_ParseTuple(args, "i", &id))
		return NULL;

	JOY_INIT_CHECK

	return PyString_FromString(SDL_JoystickName(id));
}



    /*DOC*/ static char doc_joy_is_opened[] =
    /*DOC*/    "pygame.joystick.is_opened(id) -> bool\n"
    /*DOC*/    "query opened joystick\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns true if the given joystick id has been\n"
    /*DOC*/    "previously opened.\n"
    /*DOC*/ ;

static PyObject* joy_is_opened(PyObject* self, PyObject* args)
{
	int id;
	
	if(!PyArg_ParseTuple(args, "i", &id))
		return NULL;

	JOY_INIT_CHECK

	return PyInt_FromLong(SDL_JoystickOpened(id));
}




static PyMethodDef joy__builtins__[] =
{
	{ "get_id", joy_get_id, 1, joy_get_id },
	{ "get_axes", joy_get_axes, 1, joy_get_axes },
	{ "get_balls", joy_get_balls, 1, joy_get_balls },
	{ "get_hats", joy_get_hats, 1, joy_get_hats },
	{ "get_buttons", joy_get_buttons, 1, joy_get_buttons },
	{ "get_axis", joy_get_axis, 1, joy_get_axis },
	{ "get_hat", joy_get_hat, 1, joy_get_hat },
	{ "get_button", joy_get_button, 1, joy_get_button },
	{ NULL, NULL }
};

static PyObject* joy_getattr(PyObject* self, char* attrname)
{
	return Py_FindMethod(joy__builtins__, self, attrname);

	PyErr_SetString(PyExc_NameError,	attrname);
	return NULL;
}


    /*DOC*/ static char doc_Joystick_MODULE[] =
    /*DOC*/    "Thin object wrapper around the SDL joystick\n"
    /*DOC*/    "interface. Likely to be changed.\n"
    /*DOC*/ ;

static PyTypeObject Joystick_Type =
{
	PyObject_HEAD_INIT(NULL)
	0,
	"Joystick",
	sizeof(PyJoystickObject),
	0,
	joy_dealloc,
	0,
	joy_getattr
};


static PyMethodDef joystick_builtins[] =
{
	{ "__PYGAMEinit__", joy_autoinit, 1, joy_init_doc },
	{ "init", joy_init, 1, joy_init_doc },
	{ "quit", joy_quit, 1, joy_quit_doc },
	{ "get_count", joy_get_count, 1, joy_get_count },
	{ "get_name", joy_get_name, 1, joy_get_name },
	{ "open", joy_open, 1, joy_open },
	{ "is_opened", joy_is_opened, 1, joy_is_opened },
	{ "update", joy_update, 1, joy_update },
	{ "event_state", joy_event_state, 1, joy_event_state },
	{ NULL, NULL }
};


    /*DOC*/ static char doc_pygame_joystick_MODULE[] =
    /*DOC*/    "Thin wrapper around the SDL joystick interface.\n"
    /*DOC*/    "Likely to be changed.\n"
    /*DOC*/ ;

void initcdrom()
{
	PyObject *module, *dict, *apiobj;
	static void* c_api[PYGAMEAPI_CDROM_NUMSLOTS];

	PyType_Init(PyCD_Type);


    /* create the module */
	module = Py_InitModule3("joystick", joystick_builtins, doc_pygame_joystick_MODULE);
	dict = PyModule_GetDict(module);

	/* export the c api */
	c_api[0] = &PyJoystick_Type;
	c_api[1] = PyJoystick_New;
	apiobj = PyCObject_FromVoidPtr(c_api, NULL);
	PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);

	/*imported needed apis*/
	import_pygame_base();
}



