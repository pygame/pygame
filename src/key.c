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
 *  pygame keyboard module
 */
#include "pygame.h"




/* keyboard module functions */


    /*DOC*/ static char doc_key_set_repeat[] =
    /*DOC*/    "pygame.key.set_repeat([delay, interval]) -> None\n"
    /*DOC*/    "change the keyboard repeat\n"
    /*DOC*/    "\n"
    /*DOC*/    "When the keyboard repeat is enabled, you will receive multiple\n"
    /*DOC*/    "KEYDOWN events when the user holds a key. You can control the\n"
    /*DOC*/    "repeat timing with the delay and interval values. If no arguments\n"
    /*DOC*/    "are passed, keyboard repeat will be disabled.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Good values for delay and interval are 500 and 30.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Delay is the amount of milliseconds before the first repeated\n"
    /*DOC*/    "KEYDOWN event is received. The interval is the amount of\n"
    /*DOC*/    "milliseconds for each repeated KEYDOWN event after that.\n"
    /*DOC*/ ;

static PyObject* key_set_repeat(PyObject* self, PyObject* args)
{
	int delay = 0, interval = 0;

	if(!PyArg_ParseTuple(args, "|ii", &delay, &interval))
		return NULL;

	VIDEO_INIT_CHECK();

	if(delay && !interval)
		interval = delay;

	if(SDL_EnableKeyRepeat(delay, interval) == -1)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE
}



    /*DOC*/ static char doc_key_get_pressed[] =
    /*DOC*/    "pygame.key.get_pressed() -> tuple of bools\n"
    /*DOC*/    "get the pressed state for all keys\n"
    /*DOC*/    "\n"
    /*DOC*/    "This gives you a big tuple with the pressed state for all keys.\n"
    /*DOC*/    "You index the sequence using the keysym constant (K_SPACE, etc)\n"
    /*DOC*/ ;

static PyObject* key_get_pressed(PyObject* self, PyObject* args)
{
	int num_keys;
	Uint8* key_state;
	PyObject* key_tuple;
	int i;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	VIDEO_INIT_CHECK();

	key_state = SDL_GetKeyState(&num_keys);

	if(!key_state || !num_keys)
	{
		Py_INCREF(Py_None);
		return Py_None;
	}

	if(!(key_tuple = PyTuple_New(num_keys)))
		return NULL;

	for(i = 0;i < num_keys;i++)
	{
		PyObject* key_elem;
		
		key_elem = PyInt_FromLong(key_state[i]);
		if(!key_elem)
		{
			Py_DECREF(key_tuple);
			return NULL;
		}		
		PyTuple_SET_ITEM(key_tuple, i, key_elem);
	}
	return key_tuple;
}



    /*DOC*/ static char doc_key_name[] =
    /*DOC*/    "pygame.key.name(int) -> string\n"
    /*DOC*/    "get the name of a key\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will provide you with the keyboard name for a keysym. For\n"
    /*DOC*/    "example 'pygame.key.name(K_SPACE)' will return 'space'.\n"
    /*DOC*/ ;

static PyObject* key_name(PyObject* self, PyObject* args)
{
	int key;
	
	if(!PyArg_ParseTuple(args, "i", &key))
		return NULL;

	return PyString_FromString(SDL_GetKeyName(key));	
}



    /*DOC*/ static char doc_key_get_mods[] =
    /*DOC*/    "pygame.key.get_mods() -> int\n"
    /*DOC*/    "get current state of modifier keys\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns a bitwise combination of the pressed state for all\n"
    /*DOC*/    "modifier keys (KMOD_LSHIFT, etc).\n"
    /*DOC*/ ;

static PyObject* key_get_mods(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	VIDEO_INIT_CHECK();

	return PyInt_FromLong(SDL_GetModState());
}



    /*DOC*/ static char doc_key_set_mods[] =
    /*DOC*/    "pygame.key.set_mods(int) -> None\n"
    /*DOC*/    "set the state of the modifier keys\n"
    /*DOC*/    "\n"
    /*DOC*/    "Allows you to control the internal state of the modifier keys.\n"
    /*DOC*/    "Pass an interger built from using the bitwise-or (|) of all the\n"
    /*DOC*/    "modifier keys you want to be treated as pressed.\n"
    /*DOC*/ ;

static PyObject* key_set_mods(PyObject* self, PyObject* args)
{
	int mods;

	if(!PyArg_ParseTuple(args, "i", &mods))
		return NULL;

	VIDEO_INIT_CHECK();

	SDL_SetModState(mods);
	RETURN_NONE
}



    /*DOC*/ static char doc_key_get_focused[] =
    /*DOC*/    "pygame.key.get_focused() -> bool\n"
    /*DOC*/    "state of keyboard focus\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns true when the application has the keyboard input focus.\n"
    /*DOC*/ ;

static PyObject* key_get_focused(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	VIDEO_INIT_CHECK();

	return PyInt_FromLong((SDL_GetAppState()&SDL_APPINPUTFOCUS) != 0);
}






static PyMethodDef key_builtins[] =
{
	{ "set_repeat", key_set_repeat, 1, doc_key_set_repeat },
	{ "get_pressed", key_get_pressed, 1, doc_key_get_pressed },
	{ "name", key_name, 1, doc_key_name },
	{ "get_mods", key_get_mods, 1, doc_key_get_mods },
	{ "set_mods", key_set_mods, 1, doc_key_set_mods },
	{ "get_focused", key_get_focused, 1, doc_key_get_focused },

	{ NULL, NULL }
};



    /*DOC*/ static char doc_pygame_key_MODULE[] =
    /*DOC*/    "Contains routines for dealing with the keyboard. All keyboard\n"
    /*DOC*/    "events can be retreived through the pygame.event module. With the\n"
    /*DOC*/    "key module, you can get the current state of the keyboard, as\n"
    /*DOC*/    "well as set the rate of keyboard repeating and lookup names of\n"
    /*DOC*/    "keysyms.\n"
    /*DOC*/ ;

PYGAME_EXPORT
void initkey(void)
{
	PyObject *module, *dict;

    /* create the module */
	module = Py_InitModule3("key", key_builtins, doc_pygame_key_MODULE);
	dict = PyModule_GetDict(module);


	/*imported needed apis*/
	import_pygame_base();
}




