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
 *  PyGAME mouse module
 */
#include "pygame.h"



/* mouse module functions */

    /*DOC*/ static char doc_mouse_set_pos[] =
    /*DOC*/    "pygame.mouse.set_pos(pos) -> None\n"
    /*DOC*/    "moves the cursor position\n"
    /*DOC*/    "\n"
    /*DOC*/    "Moves the mouse cursor to the specified position.\n"
    /*DOC*/    "This will generate a MOUSEMOTION event on the\n"
    /*DOC*/    "input queue. The pos argument is a\n"
    /*DOC*/    "2-number-sequence containing the desired x and y\n"
    /*DOC*/    "position.\n"
    /*DOC*/ ;

static PyObject* mouse_set_pos(PyObject* self, PyObject* args)
{
	short x, y;

	if(!TwoShortsFromObj(args, &x, &y))
		return RAISE(PyExc_TypeError, "invalid position argument for set_pos");

	VIDEO_INIT_CHECK();

	SDL_WarpMouse(x, y);

	RETURN_NONE
}

    /*DOC*/ static char doc_mouse_get_pos[] =
    /*DOC*/    "pygame.mouse.get_pos() -> x, y\n"
    /*DOC*/    "gets the cursor position\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the current position of the mouse cursor.\n"
    /*DOC*/    "This is the absolute mouse position on the screen.\n"
    /*DOC*/ ;

static PyObject* mouse_get_pos(PyObject* self, PyObject* args)
{
	int x, y;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;
	VIDEO_INIT_CHECK();

	SDL_GetMouseState(&x, &y);
	return Py_BuildValue("(ii)", x, y);
}



    /*DOC*/ static char doc_mouse_get_rel[] =
    /*DOC*/    "pygame.mouse.get_rel() -> x, y\n"
    /*DOC*/    "gets the movement of the mouse\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the total distance the mouse has moved\n"
    /*DOC*/    "since your last call to get_rel(). On the first\n"
    /*DOC*/    "call to get_rel the movement will always be 0,0.\n"
    /*DOC*/    "\n"
    /*DOC*/    "When the mouse is at the edges of the screen, the\n"
    /*DOC*/    "relative movement will be stopped. See\n"
    /*DOC*/    "mouse_visible for a way to resolve this.\n"
    /*DOC*/ ;

static PyObject* mouse_get_rel(PyObject* self, PyObject* args)
{
	int x, y;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	VIDEO_INIT_CHECK();

	SDL_GetRelativeMouseState(&x, &y);
	return Py_BuildValue("(ii)", x, y);
}



    /*DOC*/ static char doc_mouse_get_pressed[] =
    /*DOC*/    "pygame.mouse.get_pressed() -> button1, button2, button3\n"
    /*DOC*/    "state of the mouse buttons\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will return a small sequence containing the\n"
    /*DOC*/    "pressed state of the mouse buttons.\n"
    /*DOC*/ ;

static PyObject* mouse_get_pressed(PyObject* self, PyObject* args)
{
	PyObject* tuple;
	int state;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	VIDEO_INIT_CHECK();

	state = SDL_GetMouseState(NULL, NULL);
	if(!(tuple = PyTuple_New(3)))
		return NULL;

	PyTuple_SET_ITEM(tuple, 0, PyInt_FromLong((state&SDL_BUTTON(1)) != 0));
	PyTuple_SET_ITEM(tuple, 1, PyInt_FromLong((state&SDL_BUTTON(2)) != 0));
	PyTuple_SET_ITEM(tuple, 2, PyInt_FromLong((state&SDL_BUTTON(3)) != 0));

	return tuple;
}



    /*DOC*/ static char doc_mouse_set_visible[] =
    /*DOC*/    "pygame.mouse.set_visible(bool) -> bool\n"
    /*DOC*/    "show or hide the mouse cursor\n"
    /*DOC*/    "\n"
    /*DOC*/    "Shows or hides the mouse cursor. This will return\n"
    /*DOC*/    "the previous visible state of the mouse cursor.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Note that when the cursor is hidden and the\n"
    /*DOC*/    "application has grabbed the input. pyGame will\n"
    /*DOC*/    "force the mouse to stay in the center of the\n"
    /*DOC*/    "screen. Since the mouse is hidden it won't matter\n"
    /*DOC*/    "that it's not moving, but it will keep the mouse\n"
    /*DOC*/    "from the edges of the screen so the relative mouse\n"
    /*DOC*/    "position will always be true.\n"
    /*DOC*/ ;

static PyObject* mouse_set_visible(PyObject* self, PyObject* args)
{
	int toggle;

	if(!PyArg_ParseTuple(args, "i", &toggle))
		return NULL;
	VIDEO_INIT_CHECK();

	toggle = SDL_ShowCursor(toggle);
	return PyInt_FromLong(toggle);
}



    /*DOC*/ static char doc_mouse_get_focused[] =
    /*DOC*/    "pygame.mouse.get_focused() -> bool\n"
    /*DOC*/    "state of mouse input focus\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns true when the application is receiving the\n"
    /*DOC*/    "mouse input focus.\n"
    /*DOC*/ ;

static PyObject* mouse_get_focused(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	VIDEO_INIT_CHECK();

	return PyInt_FromLong((SDL_GetAppState()&SDL_APPMOUSEFOCUS) != 0);
}



static PyMethodDef mouse_builtins[] =
{
	{ "set_pos", mouse_set_pos, 1, doc_mouse_set_pos },
	{ "get_pos", mouse_get_pos, 1, doc_mouse_get_pos },
	{ "get_rel", mouse_get_rel, 1, doc_mouse_get_rel },
	{ "get_pressed", mouse_get_pressed, 1, doc_mouse_get_pressed },
	{ "set_visible", mouse_set_visible, 1, doc_mouse_set_visible },
	{ "get_focused", mouse_get_focused, 1, doc_mouse_get_focused },

	{ NULL, NULL }
};



    /*DOC*/ static char doc_pygame_mouse_MODULE[] =
    /*DOC*/    "Contains routines for dealing with the mouse.\n"
    /*DOC*/ ;

void initmouse()
{
	PyObject *module, *dict;

    /* create the module */
	module = Py_InitModule3("mouse", mouse_builtins, doc_pygame_mouse_MODULE);
	dict = PyModule_GetDict(module);

	/*imported needed apis*/
	import_pygame_base();
}

