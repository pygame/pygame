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

#define PYGAMEAPI_JOYSTICK_INTERNAL
#include "pygame.h"


#define JOYSTICK_MAXSTICKS 32
static SDL_Joystick* joystick_stickdata[JOYSTICK_MAXSTICKS] = {NULL};


staticforward PyTypeObject PyJoystick_Type;
static PyObject* PyJoystick_New(int);
#define PyJoystick_Check(x) ((x)->ob_type == &PyJoystick_Type)


static void joy_autoquit(void)
{
	int loop;
	for(loop = 0; loop < JOYSTICK_MAXSTICKS; ++loop)
	{
		if(joystick_stickdata[loop])
		{
			SDL_JoystickClose(joystick_stickdata[loop]);
			joystick_stickdata[loop] = NULL;
		}
	}

	if(SDL_WasInit(SDL_INIT_JOYSTICK))
	{
		SDL_JoystickEventState(SDL_ENABLE);
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
		PyGame_RegisterQuit(joy_autoquit);
	}
	return PyInt_FromLong(1);
}


    /*DOC*/ static char doc_quit[] =
    /*DOC*/    "pygame.joystick.quit() -> None\n"
    /*DOC*/    "uninitialize joystick module\n"
    /*DOC*/    "\n"
    /*DOC*/    "Uninitialize the joystick module manually\n"
    /*DOC*/ ;

static PyObject* quit(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	joy_autoquit();

	RETURN_NONE
}

    /*DOC*/ static char doc_init[] =
    /*DOC*/    "pygame.joystick.init() -> None\n"
    /*DOC*/    "initialize joystick module\n"
    /*DOC*/    "\n"
    /*DOC*/    "Initialize the joystick module manually\n"
    /*DOC*/ ;

static PyObject* init(PyObject* self, PyObject* arg)
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
    /*DOC*/    "Returns true when the joystick module is initialized.\n"
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
	PyObject_DEL(self);
}


    /*DOC*/ static char doc_Joystick[] =
    /*DOC*/    "pygame.joystick.Joystick(id) -> Joystick\n"
    /*DOC*/    "create new joystick object\n"
    /*DOC*/    "\n"
    /*DOC*/    "Creates a new joystick object for the given device id. The given id\n"
    /*DOC*/    "must be less than the value from pygame.joystick.get_count().\n"
    /*DOC*/ ;

static PyObject* Joystick(PyObject* self, PyObject* args)
{
	int id;	
	if(!PyArg_ParseTuple(args, "i", &id))
		return NULL;

	JOYSTICK_INIT_CHECK();

	return PyJoystick_New(id);
}



    /*DOC*/ static char doc_get_count[] =
    /*DOC*/    "pygame.joystick.get_count() -> int\n"
    /*DOC*/    "query number of joysticks on system\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the number of joysticks devices available on\n"
    /*DOC*/    "the system.\n"
    /*DOC*/ ;

static PyObject* get_count(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	JOYSTICK_INIT_CHECK();

	return PyInt_FromLong(SDL_NumJoysticks());
}




    /*DOC*/ static char doc_joy_init[] =
    /*DOC*/    "Joystick.init() -> None\n"
    /*DOC*/    "initialize a joystick device for use\n"
    /*DOC*/    "\n"
    /*DOC*/    "In order to call most members in the Joystick object, the\n"
    /*DOC*/    "Joystick must be initialized. You can initialzie the Joystick object\n"
    /*DOC*/    "at anytime, and it is ok to initialize more than once.\n"
    /*DOC*/ ;

static PyObject* joy_init(PyObject* self, PyObject* args)
{
	int joy_id = PyJoystick_AsID(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	JOYSTICK_INIT_CHECK();

	if(!joystick_stickdata[joy_id])
	{
		joystick_stickdata[joy_id] = SDL_JoystickOpen(joy_id);
		if(!joystick_stickdata[joy_id])
			return RAISE(PyExc_SDLError, SDL_GetError());
	}
	RETURN_NONE
}


    /*DOC*/ static char doc_joy_quit[] =
    /*DOC*/    "Joystick.quit() -> None\n"
    /*DOC*/    "uninitialize a joystick device for use\n"
    /*DOC*/    "\n"
    /*DOC*/    "After you are completely finished with a joystick device, you\n"
    /*DOC*/    "can use this quit() function to free access to the drive.\n"
    /*DOC*/    "This will be cleaned up automatically when the joystick module is.\n"
    /*DOC*/    "uninitialized. It is safe to call this function on an uninitialized Joystick.\n"
    /*DOC*/ ;

static PyObject* joy_quit(PyObject* self, PyObject* args)
{
	int joy_id = PyJoystick_AsID(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	JOYSTICK_INIT_CHECK();

	if(joystick_stickdata[joy_id])
	{
		SDL_JoystickClose(joystick_stickdata[joy_id]);
		joystick_stickdata[joy_id] = NULL;
	}
	RETURN_NONE
}



    /*DOC*/ static char doc_joy_get_init[] =
    /*DOC*/    "Joystick.get_init() -> bool\n"
    /*DOC*/    "check if joystick is initialized\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns a true value if the Joystick is initialized.\n"
    /*DOC*/ ;

static PyObject* joy_get_init(PyObject* self, PyObject* args)
{
	int joy_id = PyJoystick_AsID(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong(joystick_stickdata[joy_id] != NULL);
}



    /*DOC*/ static char doc_joy_get_id[] =
    /*DOC*/    "Joystick.get_id() -> idnum\n"
    /*DOC*/    "get device id number for joystick\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the device id number for this Joystick. This is the\n"
    /*DOC*/    "same number used in the call to pygame.joystick.Joystick() to create\n"
    /*DOC*/    "the object. The Joystick does not need to be initialized for this\n"
    /*DOC*/    "function to work.\n"
    /*DOC*/ ;

static PyObject* joy_get_id(PyObject* self, PyObject* args)
{
	int joy_id = PyJoystick_AsID(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;
	return PyInt_FromLong(joy_id);
}


    /*DOC*/ static char doc_joy_get_name[] =
    /*DOC*/    "Joystick.get_name() -> string\n"
    /*DOC*/    "query name of joystick drive\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the name of the Joystick device, given by the\n"
    /*DOC*/    "system. This function can be called before the Joystick\n"
    /*DOC*/    "object is initialized.\n"
    /*DOC*/ ;

static PyObject* joy_get_name(PyObject* self, PyObject* args)
{
	int joy_id = PyJoystick_AsID(self);
	
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	JOYSTICK_INIT_CHECK();

	return PyString_FromString(SDL_JoystickName(joy_id));
}



    /*DOC*/ static char doc_joy_get_numaxes[] =
    /*DOC*/    "Joystick.get_numaxes() -> int\n"
    /*DOC*/    "get number of axes on a joystick\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the number of available axes on the Joystick.\n"
    /*DOC*/ ;

static PyObject* joy_get_numaxes(PyObject* self, PyObject* args)
{
	int joy_id = PyJoystick_AsID(self);
	SDL_Joystick* joy = joystick_stickdata[joy_id];

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	JOYSTICK_INIT_CHECK();
	if(!joy)
		return RAISE(PyExc_SDLError, "Joystick not initialized");

	return PyInt_FromLong(SDL_JoystickNumAxes(joy));
}



    /*DOC*/ static char doc_joy_get_axis[] =
    /*DOC*/    "Joystick.get_axis(axis) -> float\n"
    /*DOC*/    "get the position of a joystick axis\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the current position of a joystick axis. The value\n"
    /*DOC*/    "will range from -1 to 1 with a value of 0 being centered. You\n"
    /*DOC*/    "may want to take into account some tolerance to handle jitter,\n"
    /*DOC*/    "and joystick drift may keep the joystick from centering at 0 or\n"
    /*DOC*/    "using the full range of position values.\n"
    /*DOC*/ ;

static PyObject* joy_get_axis(PyObject* self, PyObject* args)
{
	int joy_id = PyJoystick_AsID(self);
	SDL_Joystick* joy = joystick_stickdata[joy_id];
	int axis, value;
	
	if(!PyArg_ParseTuple(args, "i", &axis))
		return NULL;

	JOYSTICK_INIT_CHECK();
	if(!joy)
		return RAISE(PyExc_SDLError, "Joystick not initialized");
	if(axis < 0 || axis >= SDL_JoystickNumAxes(joy))
		return RAISE(PyExc_SDLError, "Invalid joystick axis");

	value = SDL_JoystickGetAxis(joy, axis);
	return PyFloat_FromDouble(value / 32768.0);
}


    /*DOC*/ static char doc_joy_get_numbuttons[] =
    /*DOC*/    "Joystick.get_numbuttons() -> int\n"
    /*DOC*/    "get number of buttons on a joystick\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the number of available buttons on the Joystick.\n"
    /*DOC*/ ;

static PyObject* joy_get_numbuttons(PyObject* self, PyObject* args)
{
	int joy_id = PyJoystick_AsID(self);
	SDL_Joystick* joy = joystick_stickdata[joy_id];

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	JOYSTICK_INIT_CHECK();
	if(!joy)
		return RAISE(PyExc_SDLError, "Joystick not initialized");

	return PyInt_FromLong(SDL_JoystickNumButtons(joy));
}



    /*DOC*/ static char doc_joy_get_button[] =
    /*DOC*/    "Joystick.get_button(button) -> bool\n"
    /*DOC*/    "get the position of a joystick button\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the current state of a joystick button.\n"
    /*DOC*/ ;

static PyObject* joy_get_button(PyObject* self, PyObject* args)
{
	int joy_id = PyJoystick_AsID(self);
	SDL_Joystick* joy = joystick_stickdata[joy_id];
	int index, value;
	
	if(!PyArg_ParseTuple(args, "i", &index))
		return NULL;

	JOYSTICK_INIT_CHECK();
	if(!joy)
		return RAISE(PyExc_SDLError, "Joystick not initialized");
	if(index < 0 || index >= SDL_JoystickNumButtons(joy))
		return RAISE(PyExc_SDLError, "Invalid joystick button");

	value = SDL_JoystickGetButton(joy, index);
	return PyInt_FromLong(value);
}


    /*DOC*/ static char doc_joy_get_numballs[] =
    /*DOC*/    "Joystick.get_numballs() -> int\n"
    /*DOC*/    "get number of trackballs on a joystick\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the number of available trackballs on the Joystick.\n"
    /*DOC*/ ;

static PyObject* joy_get_numballs(PyObject* self, PyObject* args)
{
	int joy_id = PyJoystick_AsID(self);
	SDL_Joystick* joy = joystick_stickdata[joy_id];

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	JOYSTICK_INIT_CHECK();
	if(!joy)
		return RAISE(PyExc_SDLError, "Joystick not initialized");

	return PyInt_FromLong(SDL_JoystickNumBalls(joy));
}



    /*DOC*/ static char doc_joy_get_ball[] =
    /*DOC*/    "Joystick.get_ball(button) -> x, y\n"
    /*DOC*/    "get the movement of a joystick trackball\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the relative movement of a joystick button. The\n"
    /*DOC*/    "value is a x, y pair holding the relative movement since the\n"
    /*DOC*/    "last call to get_ball()\n"
    /*DOC*/ ;

static PyObject* joy_get_ball(PyObject* self, PyObject* args)
{
	int joy_id = PyJoystick_AsID(self);
	SDL_Joystick* joy = joystick_stickdata[joy_id];
	int index, dx, dy;
	
	if(!PyArg_ParseTuple(args, "i", &index))
		return NULL;

	JOYSTICK_INIT_CHECK();
	if(!joy)
		return RAISE(PyExc_SDLError, "Joystick not initialized");
	if(index < 0 || index >= SDL_JoystickNumBalls(joy))
		return RAISE(PyExc_SDLError, "Invalid joystick trackball");

	SDL_JoystickGetBall(joy, index, &dx, &dy);
	return Py_BuildValue("(ii)", dx, dy);
}


    /*DOC*/ static char doc_joy_get_numhats[] =
    /*DOC*/    "Joystick.get_numhats() -> int\n"
    /*DOC*/    "get number of hats on a joystick\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the number of available directional hats on the Joystick.\n"
    /*DOC*/ ;

static PyObject* joy_get_numhats(PyObject* self, PyObject* args)
{
	int joy_id = PyJoystick_AsID(self);
	SDL_Joystick* joy = joystick_stickdata[joy_id];

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	JOYSTICK_INIT_CHECK();
	if(!joy)
		return RAISE(PyExc_SDLError, "Joystick not initialized");

	return PyInt_FromLong(SDL_JoystickNumHats(joy));
}



    /*DOC*/ static char doc_joy_get_hat[] =
    /*DOC*/    "Joystick.get_hat(button) -> x, y\n"
    /*DOC*/    "get the position of a joystick hat\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the current position of a position hat. The position\n"
    /*DOC*/    "is given as two values representing the X and Y position for the\n"
    /*DOC*/    "hat. (0, 0) means centered. A value of -1 means left/down a value\n"
    /*DOC*/    "of one means right/up\n"
    /*DOC*/ ;

static PyObject* joy_get_hat(PyObject* self, PyObject* args)
{
	int joy_id = PyJoystick_AsID(self);
	SDL_Joystick* joy = joystick_stickdata[joy_id];
	int index, px, py;
	Uint8 value;

	if(!PyArg_ParseTuple(args, "i", &index))
		return NULL;

	JOYSTICK_INIT_CHECK();
	if(!joy)
		return RAISE(PyExc_SDLError, "Joystick not initialized");
	if(index < 0 || index >= SDL_JoystickNumHats(joy))
		return RAISE(PyExc_SDLError, "Invalid joystick hat");

	px = py = 0;
	value = SDL_JoystickGetHat(joy, index);
	if(value&SDL_HAT_UP) py = 1;
	else if(value&SDL_HAT_DOWN) py = -1;
	if(value&SDL_HAT_RIGHT) px = 1;
	else if(value&SDL_HAT_LEFT) px = -1;
	
	return Py_BuildValue("(ii)", px, py);
}



static PyMethodDef joy_builtins[] =
{
	{ "init", joy_init, 1, doc_joy_init },
	{ "quit", joy_quit, 1, doc_joy_quit },
	{ "get_init", joy_get_init, 1, doc_joy_get_init },

	{ "get_id", joy_get_id, 1, doc_joy_get_id },
	{ "get_name", joy_get_name, 1, doc_joy_get_name },

	{ "get_numaxes", joy_get_numaxes, 1, doc_joy_get_numaxes },
	{ "get_axis", joy_get_axis, 1, doc_joy_get_axis },
	{ "get_numbuttons", joy_get_numbuttons, 1, doc_joy_get_numbuttons },
	{ "get_button", joy_get_button, 1, doc_joy_get_button },
	{ "get_numballs", joy_get_numballs, 1, doc_joy_get_numballs },
	{ "get_ball", joy_get_ball, 1, doc_joy_get_ball },
	{ "get_numhats", joy_get_numhats, 1, doc_joy_get_numhats },
	{ "get_hat", joy_get_hat, 1, doc_joy_get_hat },

	{ NULL, NULL }
};

static PyObject* joy_getattr(PyObject* self, char* attrname)
{
	return Py_FindMethod(joy_builtins, self, attrname);
}


    /*DOC*/ static char doc_Joystick_MODULE[] =
    /*DOC*/    "The Joystick object represents a joystick device and allows you to\n"
    /*DOC*/    "access the controls on that joystick. All functions (except get_name()\n"
    /*DOC*/    "and get_id()) require the Joystick object to be initialized. This is done\n"
    /*DOC*/    "with the Joystick.init() function.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Joystick control values are only updated during the calls to the event\n"
    /*DOC*/    "queue. Call pygame.event.pump() if you are not using the event queue for\n"
    /*DOC*/    "any input handling. Once a joystick object has been initialized, it will\n"
    /*DOC*/    "start to send joystick events to the input queue.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Be sure to understand there is a difference between the joystick module\n"
    /*DOC*/    "and the Joystick objects.\n"
    /*DOC*/ ;


static PyTypeObject PyJoystick_Type =
{
	PyObject_HEAD_INIT(NULL)
	0,
	"Joystick",
	sizeof(PyJoystickObject),
	0,
	joy_dealloc,
	0,
	joy_getattr,
	0,
	0,
	0,
	0,
	NULL,
	0, 
	(hashfunc)NULL,
	(ternaryfunc)NULL,
	(reprfunc)NULL,
	0L,0L,0L,0L,
	doc_Joystick_MODULE /* Documentation string */
};



static PyObject* PyJoystick_New(int id)
{
	PyJoystickObject* joy;

	if(id < 0 || id >= JOYSTICK_MAXSTICKS || id >= SDL_NumJoysticks())
		return RAISE(PyExc_SDLError, "Invalid joystick device number");
	
	joy = PyObject_NEW(PyJoystickObject, &PyJoystick_Type);
	if(!joy) return NULL;

	joy->id = id;

	return (PyObject*)joy;
}





static PyMethodDef joystick_builtins[] =
{
	{ "__PYGAMEinit__", joy_autoinit, 1, doc_joy_init },
	{ "init", init, 1, doc_init },
	{ "quit", quit, 1, doc_quit },
	{ "get_init", get_init, 1, doc_get_init },
	{ "get_count", get_count, 1, doc_get_count },
	{ "Joystick", Joystick, 1, doc_Joystick },
	{ NULL, NULL }
};




    /*DOC*/ static char doc_pygame_joystick_MODULE[] =
    /*DOC*/    "The joystick module provides a few functions to initialize\n"
    /*DOC*/    "the joystick subsystem and to manage the Joystick objects. These\n"
    /*DOC*/    "objects are created with the pygame.joystick.Joystick() function.\n"
    /*DOC*/    "This function needs a joystick device number to work on. All\n"
    /*DOC*/    "joystick devices on the system are enumerated for use as a Joystick\n"
    /*DOC*/    "object. To access most of the Joystick functions, you'll need to\n"
    /*DOC*/    "init() the Joystick. (note that the joystick module will already\n"
    /*DOC*/    "be initialized). When multiple Joysticks objects are created for the\n"
    /*DOC*/    "same joystick device, the state and values for those Joystick objects\n"
    /*DOC*/    "will be shared.\n"
    /*DOC*/    "\n"
    /*DOC*/    "You can call the Joystick.get_name() and Joystick.get_id() functions\n"
    /*DOC*/    "without initializing the Joystick object.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Joystick control values are only updated during the calls to the event\n"
    /*DOC*/    "queue. Call pygame.event.pump() if you are not using the event queue for\n"
    /*DOC*/    "any input handling. Once a joystick object has been initialized, it will\n"
    /*DOC*/    "start to send joystick events to the input queue.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Be sure to understand there is a difference between the joystick module\n"
    /*DOC*/    "and the Joystick objects.\n"
    /*DOC*/ ;

PYGAME_EXPORT
void initjoystick(void)
{
	PyObject *module, *dict, *apiobj;
	static void* c_api[PYGAMEAPI_JOYSTICK_NUMSLOTS];

	PyType_Init(PyJoystick_Type);


    /* create the module */
	module = Py_InitModule3("joystick", joystick_builtins, doc_pygame_joystick_MODULE);
	dict = PyModule_GetDict(module);

	PyDict_SetItemString(dict, "JoystickType", (PyObject *)&PyJoystick_Type);

	/* export the c api */
	c_api[0] = &PyJoystick_Type;
	c_api[1] = PyJoystick_New;
	apiobj = PyCObject_FromVoidPtr(c_api, NULL);
	PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
	Py_DECREF(apiobj);

	/*imported needed apis*/
	import_pygame_base();
}



