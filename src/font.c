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
 *  font module for PyGAME
 */
#define PYGAMEAPI_FONT_INTERNAL
#include "pygame.h"
#include "font.h"




staticforward PyTypeObject PyFont_Type;
static PyObject* PyFont_New(TTF_Font*);
#define PyFont_Check(x) ((x)->ob_type == &PyFont_Type)

static int font_initialized = 0;


static void font_autoquit()
{
	if(font_initialized)
	{
		font_initialized = 0;
		TTF_Quit();
	}
}


static PyObject* font_autoinit(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	if(!font_initialized)
	{
		if(TTF_Init())
			return PyInt_FromLong(0);
		font_initialized = 1;
	}
	return PyInt_FromLong(1);
}


    /*DOC*/ static char doc_quit[] =
    /*DOC*/    "pygame.font.quit() -> none\n"
    /*DOC*/    "uninitialize the font module\n"
    /*DOC*/    "\n"
    /*DOC*/    "Manually uninitialize SDL's video subsystem. It is\n"
    /*DOC*/    "safe to call this if font is currently not\n"
    /*DOC*/    "initialized.\n"
    /*DOC*/ ;

static PyObject* font_quit(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	font_autoquit();

	RETURN_NONE
}


    /*DOC*/ static char doc_init[] =
    /*DOC*/    "pygame.font.init() -> None\n"
    /*DOC*/    "initialize the display module\n"
    /*DOC*/    "\n"
    /*DOC*/    "Manually initialize the font module. Will raise an\n"
    /*DOC*/    "exception if it cannot be initialized. It is safe\n"
    /*DOC*/    "to call this function if font is currently\n"
    /*DOC*/    "initialized.\n"
    /*DOC*/ ;

static PyObject* font_init(PyObject* self, PyObject* arg)
{
	PyObject* result;
	int istrue;

	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	result = font_autoinit(self, arg);
	istrue = PyObject_IsTrue(result);
	Py_DECREF(result);
	if(!istrue)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE
}



    /*DOC*/ static char doc_get_init[] =
    /*DOC*/    "pygame.font.get_init() -> bool\n"
    /*DOC*/    "get status of font module initialization\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns true if the font module is currently\n"
    /*DOC*/    "intialized.\n"
    /*DOC*/ ;

static PyObject* get_init(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	return PyInt_FromLong(font_initialized);
}





/* font object methods */

    /*DOC*/ static char doc_font_get_height[] =
    /*DOC*/    "Font.get_height() -> int\n"
    /*DOC*/    "average height of font glyph\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the average size of each glyph in the\n"
    /*DOC*/    "font.\n"
    /*DOC*/ ;

static PyObject* font_get_height(PyObject* self, PyObject* args)
{
	TTF_Font* font = PyFont_AsFont(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong(TTF_FontHeight(font));
}



    /*DOC*/ static char doc_font_get_descent[] =
    /*DOC*/    "Font.get_descent() -> int\n"
    /*DOC*/    "gets the font descent\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the descent for the font. The descent is\n"
    /*DOC*/    "the number of pixels from the font baseline to the\n"
    /*DOC*/    "bottom of the font.\n"
    /*DOC*/ ;

static PyObject* font_get_descent(PyObject* self, PyObject* args)
{
	TTF_Font* font = PyFont_AsFont(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong(TTF_FontDescent(font));
}



    /*DOC*/ static char doc_font_get_ascent[] =
    /*DOC*/    "Font.get_ascent() -> int\n"
    /*DOC*/    "gets the font ascent\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the ascent for the font. The ascent is the\n"
    /*DOC*/    "number of pixels from the font baseline to the top\n"
    /*DOC*/    "of the font.\n"
    /*DOC*/ ;

static PyObject* font_get_ascent(PyObject* self, PyObject* args)
{
	TTF_Font* font = PyFont_AsFont(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong(TTF_FontAscent(font));
}



    /*DOC*/ static char doc_font_get_linesize[] =
    /*DOC*/    "Font.get_linesize() -> int\n"
    /*DOC*/    "gets the font recommended linesize\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the linesize for the font. Each font comes\n"
    /*DOC*/    "with it's own recommendation for the spacing\n"
    /*DOC*/    "number of pixels between each line of the font.\n"
    /*DOC*/ ;

static PyObject* font_get_linesize(PyObject* self, PyObject* args)
{
	TTF_Font* font = PyFont_AsFont(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong(TTF_FontLineSkip(font));
}



    /*DOC*/ static char doc_font_get_bold[] =
    /*DOC*/    "Font.get_bold() -> bool\n"
    /*DOC*/    "status of the bold attribute\n"
    /*DOC*/    "\n"
    /*DOC*/    "Get the current status of the font's bold\n"
    /*DOC*/    "attribute\n"
    /*DOC*/ ;

static PyObject* font_get_bold(PyObject* self, PyObject* args)
{
	TTF_Font* font = PyFont_AsFont(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong((TTF_GetFontStyle(font)&TTF_STYLE_BOLD) != 0);
}



    /*DOC*/ static char doc_font_set_bold[] =
    /*DOC*/    "Font.set_bold(bool) -> None\n"
    /*DOC*/    "assign the bold attribute\n"
    /*DOC*/    "\n"
    /*DOC*/    "Enables or disables the bold attribute for the\n"
    /*DOC*/    "font. Making the font bold does not work as well\n"
    /*DOC*/    "as you expect.\n"
    /*DOC*/ ;

static PyObject* font_set_bold(PyObject* self, PyObject* args)
{
	TTF_Font* font = PyFont_AsFont(self);
	int style, val;

	if(!PyArg_ParseTuple(args, "i", &val))
		return NULL;

	style = TTF_GetFontStyle(font);
	if(val)
		style |= TTF_STYLE_BOLD;
	else
		style &= ~TTF_STYLE_BOLD;
	TTF_SetFontStyle(font, style);

	RETURN_NONE
}



    /*DOC*/ static char doc_font_get_italic[] =
    /*DOC*/    "Font.get_bold() -> bool\n"
    /*DOC*/    "status of the italic attribute\n"
    /*DOC*/    "\n"
    /*DOC*/    "Get the current status of the font's italic\n"
    /*DOC*/    "attribute\n"
    /*DOC*/ ;

static PyObject* font_get_italic(PyObject* self, PyObject* args)
{
	TTF_Font* font = PyFont_AsFont(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong((TTF_GetFontStyle(font)&TTF_STYLE_ITALIC) != 0);
}



    /*DOC*/ static char doc_font_set_italic[] =
    /*DOC*/    "Font.set_italic(bool) -> None\n"
    /*DOC*/    "assign the italic attribute\n"
    /*DOC*/    "\n"
    /*DOC*/    "Enables or disables the italic attribute for the\n"
    /*DOC*/    "font.\n"
    /*DOC*/ ;

static PyObject* font_set_italic(PyObject* self, PyObject* args)
{
	TTF_Font* font = PyFont_AsFont(self);
	int style, val;

	if(!PyArg_ParseTuple(args, "i", &val))
		return NULL;

	style = TTF_GetFontStyle(font);
	if(val)
		style |= TTF_STYLE_ITALIC;
	else
		style &= ~TTF_STYLE_ITALIC;
	TTF_SetFontStyle(font, style);

	RETURN_NONE
}



    /*DOC*/ static char doc_font_get_underline[] =
    /*DOC*/    "Font.get_underline() -> bool\n"
    /*DOC*/    "status of the underline attribute\n"
    /*DOC*/    "\n"
    /*DOC*/    "Get the current status of the font's underline\n"
    /*DOC*/    "attribute\n"
    /*DOC*/ ;

static PyObject* font_get_underline(PyObject* self, PyObject* args)
{
	TTF_Font* font = PyFont_AsFont(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong((TTF_GetFontStyle(font)&TTF_STYLE_UNDERLINE) != 0);
}



    /*DOC*/ static char doc_font_set_underline[] =
    /*DOC*/    "Font.set_underline(bool) -> None\n"
    /*DOC*/    "assign the underline attribute\n"
    /*DOC*/    "\n"
    /*DOC*/    "Enables or disables the underline attribute for\n"
    /*DOC*/    "the font.\n"
    /*DOC*/ ;

static PyObject* font_set_underline(PyObject* self, PyObject* args)
{
	TTF_Font* font = PyFont_AsFont(self);
	int style, val;

	if(!PyArg_ParseTuple(args, "i", &val))
		return NULL;

	style = TTF_GetFontStyle(font);
	if(val)
		style |= TTF_STYLE_UNDERLINE;
	else
		style &= ~TTF_STYLE_UNDERLINE;
	TTF_SetFontStyle(font, style);

	RETURN_NONE
}



    /*DOC*/ static char doc_font_render[] =
    /*DOC*/    "Font.render(text, antialias, fgcolor, [bgcolor]) -> Surface\n"
    /*DOC*/    "render text to a new image\n"
    /*DOC*/    "\n"
    /*DOC*/    "Render the given text onto a new image surface.\n"
    /*DOC*/    "The given text can be standard python text or\n"
    /*DOC*/    "unicode. Antialiasing will smooth the edges of the\n"
    /*DOC*/    "font for a much cleaner look. The foreground color\n"
    /*DOC*/    "is a 3-number-sequence containing the desired RGB\n"
    /*DOC*/    "components for the text. The background color is\n"
    /*DOC*/    "also a 3-number-sequence of RGB. This sets the\n"
    /*DOC*/    "background color for the text. If the background\n"
    /*DOC*/    "color is omitted, the text will have a transparent\n"
    /*DOC*/    "background.\n"
    /*DOC*/ ;

static PyObject* font_render(PyObject* self, PyObject* args)
{
	TTF_Font* font = PyFont_AsFont(self);
	int aa, fr, fg, fb, br, bg, bb = -1;
	PyObject* text;
	SDL_Surface* surf;
	SDL_Color foreg, backg;

	if(!PyArg_ParseTuple(args, "Oi(iii)|(iii)", &text, &aa, &fr, &fg, &fb, &br, &bg, &bb))
		return NULL;

	foreg.r = (Uint8)fr; foreg.g = (Uint8)fg; foreg.b = (Uint8)fb;
	backg.r = (Uint8)br; backg.g = (Uint8)bg; backg.b = (Uint8)bb; 

	if(PyUnicode_Check(text))
	{
		Py_UNICODE* string = PyUnicode_AsUnicode(text);
		if(aa)
		{
			if(bb == -1)
				surf = TTF_RenderUNICODE_Blended(font, string, foreg);
			else
				surf = TTF_RenderUNICODE_Shaded(font, string, foreg, backg);
		}
		else
			surf = TTF_RenderUNICODE_Solid(font, string, foreg);
	}
	else if(PyString_Check(text))
	{
		char* string = PyString_AsString(text);
		if(aa)
		{
			if(bb == -1)
				surf = TTF_RenderText_Blended(font, string, foreg);
			else
				surf = TTF_RenderText_Shaded(font, string, foreg, backg);
		}
		else
			surf = TTF_RenderText_Solid(font, string, foreg);
	}
	else
		return RAISE(PyExc_TypeError, "text must be a string or unicode");

	if(!surf)
		return RAISE(PyExc_SDLError, SDL_GetError());

	if(!aa && bb != -1) /*turn off transparancy*/
	{			
		SDL_SetColorKey(surf, 0, 0);
		surf->format->palette->colors[0].r = backg.r;
		surf->format->palette->colors[0].g = backg.g;
		surf->format->palette->colors[0].b = backg.b;
	}

	return PySurface_New(surf);
}



    /*DOC*/ static char doc_font_size[] =
    /*DOC*/    "Font.size(text) -> width, height\n"
    /*DOC*/    "size of rendered text\n"
    /*DOC*/    "\n"
    /*DOC*/    "Computes the rendered size of the given text. The\n"
    /*DOC*/    "text can be standard python text or unicode. Know\n"
    /*DOC*/    "that changing the bold and italic attributes will\n"
    /*DOC*/    "change the size of the rendered text.\n"
    /*DOC*/ ;

static PyObject* font_size(PyObject* self, PyObject* args)
{
	TTF_Font* font = PyFont_AsFont(self);
	int w, h;
	PyObject* text;

	if(!PyArg_ParseTuple(args, "O", &text))
		return NULL;

	if(PyUnicode_Check(text))
	{
		Py_UNICODE* string = PyUnicode_AsUnicode(text);
		TTF_SizeUNICODE(font, string, &w, &h);
	}
	else if(PyString_Check(text))
	{
		char* string = PyString_AsString(text);
		TTF_SizeText(font, string, &w, &h);
	}
	else
		return RAISE(PyExc_TypeError, "text must be a string or unicode");

	return Py_BuildValue("(ii)", w, h);
}



static PyMethodDef fontobj_builtins[] =
{
	{ "get_height", font_get_height, 1, doc_font_get_height },
	{ "get_descent", font_get_descent, 1, doc_font_get_descent },
	{ "get_ascent", font_get_ascent, 1, doc_font_get_ascent },
	{ "get_linesize", font_get_linesize, 1, doc_font_get_linesize },

	{ "get_bold", font_get_bold, 1, doc_font_get_bold },
	{ "set_bold", font_set_bold, 1, doc_font_set_bold },
	{ "get_italic", font_get_italic, 1, doc_font_get_italic },
	{ "set_italic", font_set_italic, 1, doc_font_set_italic },
	{ "get_underline", font_get_underline, 1, doc_font_get_underline },
	{ "set_underline", font_set_underline, 1, doc_font_set_underline },

	{ "render", font_render, 1, doc_font_render },
	{ "size", font_size, 1, doc_font_size },

	{ NULL, NULL }
};



/*font object internals*/

static void font_dealloc(PyObject* self)
{
	TTF_Font* font = PyFont_AsFont(self);
	
	if(font_initialized)
		TTF_CloseFont(font);

	PyMem_DEL(self);
}


static PyObject* font_getattr(PyObject* self, char* attrname)
{
	if(font_initialized)
		return Py_FindMethod(fontobj_builtins, self, attrname);

	PyErr_SetString(PyExc_NameError, attrname);
	return NULL; 
}


    /*DOC*/ static char doc_Font_MODULE[] =
    /*DOC*/    "Font objects can control and render text.\n"
    /*DOC*/ ;

static PyTypeObject PyFont_Type = 
{
	PyObject_HEAD_INIT(NULL)
	0,
	"Font",
	sizeof(PyFontObject),
	0,
	font_dealloc,	
	0,
	font_getattr
};



/*font module methods*/

    /*DOC*/ static char doc_font_font[] =
    /*DOC*/    "pygame.font(file, size) -> Font\n"
    /*DOC*/    "create a new font object\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will create a new font object. The given file\n"
    /*DOC*/    "must be an existing filename. The font loader does\n"
    /*DOC*/    "not work with python file-like objects. The size\n"
    /*DOC*/    "represents the height of the font in pixels.\n"
    /*DOC*/ ;

static PyObject* font_font(PyObject* self, PyObject* args)
{
	char* filename;
	int fontsize;
	TTF_Font* font;
	PyObject* fontobj;
	if(!PyArg_ParseTuple(args, "si", &filename, &fontsize))
		return NULL;

	if(!font_initialized)
		return RAISE(PyExc_SDLError, "font not initialized");

	font = TTF_OpenFont(filename, fontsize);
	if(!font)
		return RAISE(PyExc_RuntimeError, SDL_GetError());
	fontobj = PyFont_New(font);
	return fontobj;
}



static PyMethodDef font_builtins[] =
{
	{ "__PYGAMEinit__", font_autoinit, 1, doc_init },
	{ "init", font_init, 1, doc_init },
	{ "quit", font_quit, 1, doc_quit },
	{ "get_init", get_init, 1, doc_get_init },

	{ "font", font_font, 1, doc_font_font },

	{ NULL, NULL }
};



static PyObject* PyFont_New(TTF_Font* font)
{
	PyFontObject* fontobj;
	
	if(!font)
		return RAISE(PyExc_RuntimeError, "unable to load font.");

	fontobj = PyObject_NEW(PyFontObject, &PyFont_Type);
	if(!fontobj)
		return NULL;

	fontobj->font = font;
	return (PyObject*)fontobj;
}



    /*DOC*/ static char doc_pygame_font_MODULE[] =
    /*DOC*/    "Contains the font object and the functions used to\n"
    /*DOC*/    "create them.\n"
    /*DOC*/ ;

void initfont()
{
	PyObject *module, *dict, *apiobj;
	static void* c_api[PYGAMEAPI_FONT_NUMSLOTS];

	PyType_Init(PyFont_Type);

    /* create the module */
	module = Py_InitModule3("font", font_builtins, doc_pygame_font_MODULE);
	dict = PyModule_GetDict(module);

	/* export the c api */
	c_api[0] = &PyFont_Type;
	c_api[1] = PyFont_New;
	c_api[2] = &font_initialized;
	apiobj = PyCObject_FromVoidPtr(c_api, NULL);
	PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);

	/*imported needed apis*/
	import_pygame_base();
	import_pygame_surface();
}

