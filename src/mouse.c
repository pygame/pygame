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
 *  pygame mouse module
 */
#include "pygame.h"



/* mouse module functions */

    /*DOC*/ static char doc_mouse_set_pos[] =
    /*DOC*/    "pygame.mouse.set_pos(pos) -> None\n"
    /*DOC*/    "moves the cursor position\n"
    /*DOC*/    "\n"
    /*DOC*/    "Moves the mouse cursor to the specified position. This will\n"
    /*DOC*/    "generate a MOUSEMOTION event on the input queue. The pos argument\n"
    /*DOC*/    "is a 2-number-sequence containing the desired x and y position.\n"
    /*DOC*/ ;

static PyObject* mouse_set_pos(PyObject* self, PyObject* args)
{
	int x, y;

	if(!TwoIntsFromObj(args, &x, &y))
		return RAISE(PyExc_TypeError, "invalid position argument for set_pos");

	VIDEO_INIT_CHECK();

	SDL_WarpMouse((Uint16)x, (Uint16)y);

	RETURN_NONE
}

    /*DOC*/ static char doc_mouse_get_pos[] =
    /*DOC*/    "pygame.mouse.get_pos() -> x, y\n"
    /*DOC*/    "gets the cursor position\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the current position of the mouse cursor. This is the\n"
    /*DOC*/    "absolute mouse position inside your game window.\n"
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
    /*DOC*/    "Returns the total distance the mouse has moved since your last\n"
    /*DOC*/    "call to get_rel(). On the first call to get_rel the movement will\n"
    /*DOC*/    "always be 0,0.\n"
    /*DOC*/    "\n"
    /*DOC*/    "When the mouse is at the edges of the screen, the relative\n"
    /*DOC*/    "movement will be stopped. See mouse_visible for a way to resolve\n"
    /*DOC*/    "this.\n"
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
    /*DOC*/    "This will return a small sequence containing the pressed state of\n"
    /*DOC*/    "each mouse button.\n"
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
    /*DOC*/    "Shows or hides the mouse cursor. This will return the previous\n"
    /*DOC*/    "visible state of the mouse cursor.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Note that when the cursor is hidden and the application has\n"
    /*DOC*/    "grabbed the input. pygame will force the mouse to stay in the\n"
    /*DOC*/    "center of the screen. Since the mouse is hidden it won't matter\n"
    /*DOC*/    "that it's not moving, but it will keep the mouse from the edges\n"
    /*DOC*/    "of the screen so the relative mouse position will always be true.\n"
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
    /*DOC*/    "Returns true when the application is receiving the mouse input\n"
    /*DOC*/    "focus.\n"
    /*DOC*/ ;

static PyObject* mouse_get_focused(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	VIDEO_INIT_CHECK();

	return PyInt_FromLong((SDL_GetAppState()&SDL_APPMOUSEFOCUS) != 0);
}



    /*DOC*/ static char doc_mouse_set_cursor[] =
    /*DOC*/    "pygame.mouse.set_cursor(size, hotspot, xormasks, andmasks) -> None\n"
    /*DOC*/    "state of shape of the mouse cursor\n"
    /*DOC*/    "\n"
    /*DOC*/    "When the mouse cursor is visible, it will be displayed\n"
    /*DOC*/    "as a black and white bitmap using the given bitmask arrays.\n"
    /*DOC*/    "The size is a sequence containing the cursor width and height.\n"
    /*DOC*/    "Hotspot is a sequence containing the cursor hotspot position.\n"
    /*DOC*/    "xormasks is a sequence of bytes containing the cursor xor data\n"
    /*DOC*/    "masks. Lastly is andmasks, a sequence of bytes containting the\n"
    /*DOC*/    "cursor bitmask data.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Width must be a multiple of 8, and the mask arrays must be the\n"
    /*DOC*/    "correct size for the given width and height. Otherwise an exception.\n"
    /*DOC*/ ;

static PyObject* mouse_set_cursor(PyObject* self, PyObject* args)
{
	int w, h, spotx, spoty;
	PyObject *xormask, *andmask;
	Uint8 *xordata=NULL, *anddata=NULL;
	int xorsize, andsize, loop;
	int val;
	SDL_Cursor *lastcursor, *cursor = NULL;

	if(!PyArg_ParseTuple(args, "(ii)(ii)OO", &w, &h, &spotx, &spoty, &xormask, &andmask))
		return NULL;

	VIDEO_INIT_CHECK();

	if(!PySequence_Check(xormask) || !PySequence_Check(andmask))
		return RAISE(PyExc_TypeError, "xormask and andmask must be sequences");

	if(w % 8)
		return RAISE(PyExc_ValueError, "Cursor width must be divisible by 8.");

	xorsize = PySequence_Length(xormask);
	andsize = PySequence_Length(andmask);

	if(xorsize != w*h/8 || andsize != w*h/8)
		return RAISE(PyExc_ValueError, "bitmasks must be sized width*height/8");

	xordata = (Uint8*)malloc(xorsize);
	anddata = (Uint8*)malloc(andsize);

	for(loop = 0; loop < xorsize; ++loop)
	{
		if(!IntFromObjIndex(xormask, loop, &val))
			goto interror;
		xordata[loop] = (Uint8)val;
		if(!IntFromObjIndex(andmask, loop, &val))
			goto interror;
		anddata[loop] = (Uint8)val;
	}

	cursor = SDL_CreateCursor(xordata, anddata, w, h, spotx, spoty);
	free(xordata);
	free(anddata);
	xordata = NULL;
	anddata = NULL;

	if(!cursor)
		return RAISE(PyExc_SDLError, SDL_GetError());

	lastcursor = SDL_GetCursor();
	SDL_SetCursor(cursor);
	SDL_FreeCursor(lastcursor);

	RETURN_NONE;

interror:
	if(xordata) free(xordata);
	if(anddata) free(anddata);
	return RAISE(PyExc_TypeError, "Invalid number in mask array");
}


    /*DOC*/ static char doc_mouse_get_cursor[] =
    /*DOC*/    "pygame.mouse.get_cursor() -> size, hotspot, xormasks, andmasks\n"
    /*DOC*/    "get mouse cursor data\n"
    /*DOC*/    "\n"
    /*DOC*/    "The mouse cursor data is the same as those passed into set_cursor.\n"
    /*DOC*/ ;

static PyObject* mouse_get_cursor(PyObject* self, PyObject* args)
{
	SDL_Cursor *cursor = NULL;
	PyObject* xordata, *anddata;
	int size, loop, w, h, spotx, spoty;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	VIDEO_INIT_CHECK();

	cursor = SDL_GetCursor();
	if(!cursor)
		return RAISE(PyExc_SDLError, SDL_GetError());

	w = cursor->area.w;
	h = cursor->area.h;
	spotx = cursor->hot_x;
	spoty = cursor->hot_y;

	size = cursor->area.w * cursor->area.h / 8;
	xordata = PyTuple_New(size);
	if(!xordata)
		return NULL;
	anddata = PyTuple_New(size);
	if(!anddata)
	{
		Py_DECREF(anddata);
		return NULL;
	}

	for(loop = 0; loop < size; ++loop)
	{
		PyTuple_SET_ITEM(xordata, loop, PyInt_FromLong(cursor->data[loop]));
		PyTuple_SET_ITEM(anddata, loop, PyInt_FromLong(cursor->mask[loop]));
	}

	return Py_BuildValue("((ii)(ii)OO)", w, h, spotx, spoty, xordata, anddata);
}




static PyMethodDef mouse_builtins[] =
{
	{ "set_pos", mouse_set_pos, 1, doc_mouse_set_pos },
	{ "get_pos", mouse_get_pos, 1, doc_mouse_get_pos },
	{ "get_rel", mouse_get_rel, 1, doc_mouse_get_rel },
	{ "get_pressed", mouse_get_pressed, 1, doc_mouse_get_pressed },
	{ "set_visible", mouse_set_visible, 1, doc_mouse_set_visible },
	{ "get_focused", mouse_get_focused, 1, doc_mouse_get_focused },
	{ "set_cursor", mouse_set_cursor, 1, doc_mouse_set_cursor },
	{ "get_cursor", mouse_get_cursor, 1, doc_mouse_get_cursor },

	{ NULL, NULL }
};



    /*DOC*/ static char doc_pygame_mouse_MODULE[] =
    /*DOC*/    "Contains routines for dealing with the mouse. All mouse events\n"
    /*DOC*/    "are retrieved through the pygame.event module. The mouse module\n"
    /*DOC*/    "can be used to get the current state of the mouse. It can also be\n"
    /*DOC*/    "used to set the state of the system cursor.\n"
    /*DOC*/    "\n"
    /*DOC*/    "If you hide the mouse cursor with pygame.mouse.set_visible(0) and\n"
    /*DOC*/    "lock the mouse focus to your game with pygame.event.set_grab(1),\n"
    /*DOC*/    "the hidden mouse will be forced to the center of the screen. This\n"
    /*DOC*/    "will help your relative mouse motions keep from getting stuck on\n"
    /*DOC*/    "the edges of the screen.\n"
    /*DOC*/ ;

PYGAME_EXPORT
void initmouse(void)
{
	PyObject *module, *dict;

    /* create the module */
	module = Py_InitModule3("mouse", mouse_builtins, doc_pygame_mouse_MODULE);
	dict = PyModule_GetDict(module);

	/*imported needed apis*/
	import_pygame_base();
}

