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
 *  font module for pygame
 */
#define PYGAMEAPI_FONT_INTERNAL
#include <stdio.h>
#include <string.h>
#include "pygame.h"
#include "font.h"
#include "structmember.h"


staticforward PyTypeObject PyFont_Type;
static PyObject* PyFont_New(TTF_Font*);
#define PyFont_Check(x) ((x)->ob_type == &PyFont_Type)

static int font_initialized = 0;
static char* font_defaultname = "freesansbold.ttf";
static PyObject* self_module = NULL;

static char* pkgdatamodule_name = "pygame.pkgdata";
static char* resourcefunc_name = "getResource";

static PyObject *font_resource(char *filename) {
	PyObject* load_basicfunc = NULL;
	PyObject* pkgdatamodule = NULL;
	PyObject* resourcefunc = NULL;
	PyObject* result = NULL;

	pkgdatamodule = PyImport_ImportModule(pkgdatamodule_name);
	if (!pkgdatamodule) goto font_resource_end;

	resourcefunc = PyObject_GetAttrString(pkgdatamodule, resourcefunc_name);
	if (!resourcefunc) goto font_resource_end;

	result = PyObject_CallFunction(resourcefunc, "s", filename);
	if (!result) goto font_resource_end;

font_resource_end:
	Py_XDECREF(pkgdatamodule);
	Py_XDECREF(resourcefunc);
	Py_XDECREF(load_basicfunc);
	return result;
}


static void font_autoquit(void)
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
		PyGame_RegisterQuit(font_autoquit);

		if(TTF_Init())
			return PyInt_FromLong(0);
		font_initialized = 1;

	}
	return PyInt_FromLong(font_initialized);
}


    /*DOC*/ static char doc_quit[] =
    /*DOC*/    "pygame.font.quit() -> none\n"
    /*DOC*/    "uninitialize the font module\n"
    /*DOC*/    "\n"
    /*DOC*/    "Manually uninitialize SDL_ttf's font system. It is safe to call\n"
    /*DOC*/    "this if font is currently not initialized.\n"
    /*DOC*/ ;

static PyObject* fontmodule_quit(PyObject* self, PyObject* arg)
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
    /*DOC*/    "Manually initialize the font module. Will raise an exception if\n"
    /*DOC*/    "it cannot be initialized. It is safe to call this function if\n"
    /*DOC*/    "font is currently initialized.\n"
    /*DOC*/ ;

static PyObject* fontmodule_init(PyObject* self, PyObject* arg)
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
    /*DOC*/    "Returns true if the font module is currently intialized.\n"
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
    /*DOC*/    "Returns the average size of each glyph in the font.\n"
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
    /*DOC*/    "Returns the descent for the font. The descent is the number of\n"
    /*DOC*/    "pixels from the font baseline to the bottom of the font.\n"
    /*DOC*/    "With most fonts this is a negative number.\n"
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
    /*DOC*/    "Returns the ascent for the font. The ascent is the number of\n"
    /*DOC*/    "pixels from the font baseline to the top of the font.\n"
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
    /*DOC*/    "Returns the linesize for the font. Each font comes with it's own\n"
    /*DOC*/    "recommendation for the spacing number of pixels between each line\n"
    /*DOC*/    "of the font.\n"
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
    /*DOC*/    "Get the current status of the font's bold attribute\n"
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
    /*DOC*/    "Enables or disables the bold attribute for the font. Making the\n"
    /*DOC*/    "font bold does not work as well as you expect.\n"
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
    /*DOC*/    "Font.get_italic() -> bool\n"
    /*DOC*/    "status of the italic attribute\n"
    /*DOC*/    "\n"
    /*DOC*/    "Get the current status of the font's italic attribute\n"
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
    /*DOC*/    "Enables or disables the italic attribute for the font.\n"
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
    /*DOC*/    "Get the current status of the font's underline attribute\n"
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
    /*DOC*/    "Enables or disables the underline attribute for the font.\n"
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
    /*DOC*/    "Font.render(text, antialias, fore_RGBA, [back_RGBA]) -> Surface\n"
    /*DOC*/    "render text to a new image\n"
    /*DOC*/    "\n"
    /*DOC*/    "Render the given text onto a new image surface. The given text\n"
    /*DOC*/    "can be standard python text or unicode. Antialiasing will smooth\n"
    /*DOC*/    "the edges of the font for a much cleaner look. The foreground\n"
    /*DOC*/    "and background color are both RGBA, the alpha component is ignored\n"
    /*DOC*/    "if given. If the background color is omitted, the text will have a\n"
    /*DOC*/    "transparent background.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Note that font rendering is not thread safe, therefore only one\n"
    /*DOC*/    "thread can render text at any given time.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Also, rendering smooth text with underlines will crash with SDL_ttf\n"
    /*DOC*/    "less that version 2.0, be careful.\n"
    /*DOC*/    "\n"
    /*DOC*/    "If you pass an empty string, render() will return a blank surface\n"
    /*DOC*/    "1 pixel wide and the same height as the font.\n"
    /*DOC*/ ;

static PyObject* font_render(PyObject* self, PyObject* args)
{
	TTF_Font* font = PyFont_AsFont(self);
	int aa;
	PyObject* text, *final;
	PyObject* fg_rgba_obj, *bg_rgba_obj = NULL;
	Uint8 rgba[4];
	SDL_Surface* surf;
	SDL_Color foreg, backg;
	if(!PyArg_ParseTuple(args, "OiO|O", &text, &aa, &fg_rgba_obj, &bg_rgba_obj))
		return NULL;

	if(!RGBAFromObj(fg_rgba_obj, rgba))
		return RAISE(PyExc_TypeError, "Invalid foreground RGBA argument");
	foreg.r = rgba[0]; foreg.g = rgba[1]; foreg.b = rgba[2];
	if(bg_rgba_obj)
	{
		if(!RGBAFromObj(bg_rgba_obj, rgba))
			return RAISE(PyExc_TypeError, "Invalid background RGBA argument");
		backg.r = rgba[0]; backg.g = rgba[1]; backg.b = rgba[2];
	}


	if(!PyObject_IsTrue(text))
	{
		int height = TTF_FontHeight(font);
		surf = SDL_CreateRGBSurface(SDL_SWSURFACE, 1, height, 32, 0xff<<16, 0xff<<8, 0xff, 0);
		if(bg_rgba_obj)
		{
			Uint32 c = SDL_MapRGB(surf->format, backg.r, backg.g, backg.b);
			SDL_FillRect(surf, NULL, c);
		}
		else
			SDL_SetColorKey(surf, SDL_SRCCOLORKEY, 0);
	}
	else if(PyUnicode_Check(text))
	{
		PyObject* strob = PyEval_CallMethod(text, "encode", "(s)", "utf-8");
		char *string = PyString_AsString(strob);

		if(aa)
		{
			if(!bg_rgba_obj)
				surf = TTF_RenderUTF8_Blended(font, string, foreg);
			else
				surf = TTF_RenderUTF8_Shaded(font, string, foreg, backg);
		}
		else
			surf = TTF_RenderUTF8_Solid(font, string, foreg);

		Py_DECREF(strob);
	}
	else if(PyString_Check(text))
	{
		char* string = PyString_AsString(text);
		if(aa)
		{
			if(!bg_rgba_obj)
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
		return RAISE(PyExc_SDLError, "SDL_ttf render failed");

	if(!aa && bg_rgba_obj) /*turn off transparancy*/
	{
		SDL_SetColorKey(surf, 0, 0);
		surf->format->palette->colors[0].r = backg.r;
		surf->format->palette->colors[0].g = backg.g;
		surf->format->palette->colors[0].b = backg.b;
	}

	final = PySurface_New(surf);
	if(!final)
		SDL_FreeSurface(surf);
	return final;
}



    /*DOC*/ static char doc_font_size[] =
    /*DOC*/    "Font.size(text) -> width, height\n"
    /*DOC*/    "size of rendered text\n"
    /*DOC*/    "\n"
    /*DOC*/    "Computes the rendered size of the given text. The text can be\n"
    /*DOC*/    "standard python text or unicode. Changing the bold and italic\n"
    /*DOC*/    "attributes can change the size of the rendered text.\n"
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
		PyObject* strob = PyEval_CallMethod(text, "encode", "(s)", "utf-8");
		char *string = PyString_AsString(strob);

		TTF_SizeUTF8(font, string, &w, &h);

		Py_DECREF(strob);
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



static PyMethodDef font_methods[] =
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

static void font_dealloc(PyFontObject* self)
{
	TTF_Font* font = PyFont_AsFont(self);

	if(font && font_initialized)
		TTF_CloseFont(font);

	if(self->weakreflist)
		PyObject_ClearWeakRefs((PyObject*)self);
	self->ob_type->tp_free((PyObject*)self);
}


static int font_init(PyFontObject *self, PyObject *args, PyObject *kwds)
{
	int fontsize;
	TTF_Font* font = NULL;
	PyObject* fileobj;
	
	self->font = NULL;
	if(!PyArg_ParseTuple(args, "Oi", &fileobj, &fontsize))
		return -1;

	if(!font_initialized)
	{
		RAISE(PyExc_SDLError, "font not initialized");
		return -1;
	}

	Py_INCREF(fileobj);

	if(fontsize <= 1)
		fontsize = 1;

	if(fileobj == Py_None) {
		Py_DECREF(fileobj);
		fileobj = font_resource(font_defaultname);
		if(!fileobj)
		{
			RAISE(PyExc_RuntimeError, "default font not found");
			return -1;
		}
		fontsize = (int)(fontsize * .6875);
		if(fontsize <= 1)
			fontsize = 1;
	}
	if(PyString_Check(fileobj) || PyUnicode_Check(fileobj))
	{
		FILE* test;
		char* filename = PyString_AsString(fileobj);
		Py_DECREF(fileobj);
		fileobj = NULL;

		if(!filename)
			return -1;

		/*check if it is a valid file, else SDL_ttf segfaults*/
		test = fopen(filename, "rb");
		if(!test)
		{
			if (!strcmp(filename, font_defaultname))
			{
				fileobj = font_resource(font_defaultname);
			}
			if (!fileobj) {
				PyErr_SetString(PyExc_IOError, "unable to read font filename");
				return -1;
			}
		}
		else
		{
			fclose(test);
			Py_BEGIN_ALLOW_THREADS
			font = TTF_OpenFont(filename, fontsize);
			Py_END_ALLOW_THREADS
		}
	}
	if (!font)
	{
#ifdef TTF_MAJOR_VERSION
		SDL_RWops *rw;
		rw = RWopsFromPython(fileobj);
		if (!rw) {
			Py_DECREF(fileobj);
			return -1;
		}
		Py_BEGIN_ALLOW_THREADS
		font = TTF_OpenFontIndexRW(rw, 1, fontsize, 0);
		Py_END_ALLOW_THREADS
#else
		Py_DECREF(fileobj);
		RAISE(PyExc_NotImplementedError, "nonstring fonts require SDL_ttf-2.0.6");
		return -1;
#endif
	}

	if(!font)
	{
		RAISE(PyExc_RuntimeError, SDL_GetError());
		return -1;
	}

	self->font = font;
	return 0;
}


    /*DOC*/ static char doc_Font_MODULE[] =
    /*DOC*/    "The font object is created only from pygame.font.Font(). Once a\n"
    /*DOC*/    "font is created it's size and TTF file cannot be changed. The\n"
    /*DOC*/    "Font objects are mainly used to render() text into a new Surface.\n"
    /*DOC*/    "The Font objects also have a few states that can be set with\n"
    /*DOC*/    "set_underline(bool), set_bold(bool), set_italic(bool). Each of\n"
    /*DOC*/    "these functions contains an equivalent get_XXX() routine to find\n"
    /*DOC*/    "the current state. There are also many routines to query the\n"
    /*DOC*/    "dimensions of the text. The rendering functions work with both\n"
    /*DOC*/    "normal python strings, as well as with unicode strings.\n"
    /*DOC*/ ;

static PyTypeObject PyFont_Type =
{
	PyObject_HEAD_INIT(NULL)
	0,
	"pygame.font.Font",
	sizeof(PyFontObject),
	0,
	(destructor)font_dealloc,
	0,
	0, /*getattr*/
	0,
	0,
	0,
	0,
	NULL,
	0,
	(hashfunc)NULL,
	(ternaryfunc)NULL,
	(reprfunc)NULL,
	0L,0L,0L,
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
	doc_Font_MODULE, /* Documentation string */
	0,					/* tp_traverse */
	0,					/* tp_clear */
	0,					/* tp_richcompare */
	offsetof(PyFontObject, weakreflist),    /* tp_weaklistoffset */
	0,					/* tp_iter */
	0,					/* tp_iternext */
	font_methods,			        /* tp_methods */
	0,				        /* tp_members */
	0,				        /* tp_getset */
	0,					/* tp_base */
	0,					/* tp_dict */
	0,					/* tp_descr_get */
	0,					/* tp_descr_set */
	0,					/* tp_dictoffset */
	(initproc)font_init,			/* tp_init */
	0,					/* tp_alloc */
	0,	                /* tp_new */
};

	//PyType_GenericNew,	                /* tp_new */


/*font module methods*/

    /*DOC*/ static char doc_get_default_font[] =
    /*DOC*/    "pygame.font.get_default_font() -> string\n"
    /*DOC*/    "get the name of the default font\n"
    /*DOC*/    "\n"
    /*DOC*/    "returns the filename for the default truetype font.\n"
    /*DOC*/ ;

static PyObject* get_default_font(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;
	return PyString_FromString(font_defaultname);
}



/*font module methods*/
#if 0
    /*DOC*/ static char doc_Font[] =
    /*DOC*/    "pygame.font.Font(file, size) -> Font\n"
    /*DOC*/    "create a new font object\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will create a new font object. The given file can\n"
    /*DOC*/    "be a filename or any python file-like object.\n"
    /*DOC*/    "The size represents the height of the font in\n"
    /*DOC*/    "pixels. The file argument can be 'None', which will\n"
    /*DOC*/    "use a plain default font.\n"
    /*DOC*/    "\n"
    /*DOC*/    "You must have at least SDL_ttf-2.0.6 for file object\n"
    /*DOC*/    "support. You can load TTF and FON fonts.\n"
    /*DOC*/ ;
#endif


static PyMethodDef font_builtins[] =
{
	{ "__PYGAMEinit__", font_autoinit, 1, doc_init },
	{ "init", fontmodule_init, 1, doc_init },
	{ "quit", fontmodule_quit, 1, doc_quit },
	{ "get_init", get_init, 1, doc_get_init },
	{ "get_default_font", get_default_font, 1, doc_get_default_font },
	{ NULL, NULL }
};



static PyObject* PyFont_New(TTF_Font* font)
{
	PyFontObject* fontobj;

	if(!font)
		return RAISE(PyExc_RuntimeError, "unable to load font.");
	fontobj = (PyFontObject *)PyFont_Type.tp_new(&PyFont_Type, NULL, NULL);

	if(fontobj)
		fontobj->font = font;

	return (PyObject*)fontobj;
}



    /*DOC*/ static char doc_pygame_font_MODULE[] =
    /*DOC*/    "The font module allows for rendering TrueType fonts into a new\n"
    /*DOC*/    "Surface object. This module is optional and requires SDL_ttf as a\n"
    /*DOC*/    "dependency. You may want to check for pygame.font to import and\n"
    /*DOC*/    "initialize before attempting to use the module.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Most of the work done with fonts are done by using the actual\n"
    /*DOC*/    "Font objects. The module by itself only has routines to\n"
    /*DOC*/    "initialize the module and create Font objects with\n"
    /*DOC*/    "pygame.font.Font().\n"
    /*DOC*/    "\n"
    /*DOC*/    "You can load fonts from the standard system fonts by using the\n"
    /*DOC*/    "pygame.font.SysFont() method. There are also other functions to\n"
    /*DOC*/    "help you work with system fonts.\n"
    /*DOC*/    "\n"
    /*DOC*/ ;

PYGAME_EXPORT
void initfont(void)
{
	PyObject *module, *apiobj;
	static void* c_api[PYGAMEAPI_FONT_NUMSLOTS];

	PyFONT_C_API[0] = PyFONT_C_API[0]; /*clean an unused warning*/

		if (PyType_Ready(&PyFont_Type) < 0)
			return;

	/* create the module */
		PyFont_Type.ob_type = &PyType_Type;
		PyFont_Type.tp_new = &PyType_GenericNew;

	module = Py_InitModule3("font", font_builtins, doc_pygame_font_MODULE);
	self_module = module;

	Py_INCREF((PyObject*)&PyFont_Type);
	PyModule_AddObject(module, "FontType", (PyObject *)&PyFont_Type);
	Py_INCREF((PyObject*)&PyFont_Type);
	PyModule_AddObject(module, "Font", (PyObject *)&PyFont_Type);

	/* export the c api */
	c_api[0] = &PyFont_Type;
	c_api[1] = PyFont_New;
	c_api[2] = &font_initialized;
	apiobj = PyCObject_FromVoidPtr(c_api, NULL);
	PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj);

	/*imported needed apis*/
	import_pygame_base();
	import_pygame_surface();
	import_pygame_rwobject();
}

