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
 *  image module for PyGAME
 */
#include "pygame.h"
#include <SDL_image.h>


static char* find_extension(char* fullname)
{
	char* dot;

	if(!fullname)
		return NULL;

	dot = strrchr(fullname, '.');
	if(!dot)
		return fullname;

	return dot+1;
}



    /*DOC*/ static char doc_load[] =
    /*DOC*/    "pygame.image.load(file, [namehint]) -> Surface\n"
    /*DOC*/    "load an image to a new Surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will load an image into a new surface. You can pass it\n"
    /*DOC*/    "either a filename, or a python file-like object to load the image\n"
    /*DOC*/    "from. If you pass a file-like object that isn't actually a file\n"
    /*DOC*/    "(like the StringIO class), then you might want to also pass\n"
    /*DOC*/    "either the filename or extension as the namehint string. The\n"
    /*DOC*/    "namehint can help the loader determine the filetype.\n"
    /*DOC*/    "\n"
    /*DOC*/    "You will only be able to load the types of images supported by\n"
    /*DOC*/    "your build of SDL_image. This will always include GIF, BMP, PPM,\n"
    /*DOC*/    "PCX, and TGA. SDL_image can also load JPG, PNG, and TIF, but they are\n"
    /*DOC*/    "optional.\n"
    /*DOC*/ ;

static PyObject* load(PyObject* self, PyObject* arg)
{
	PyObject* file;
	char* name = NULL;
	SDL_Surface* surf;
	SDL_RWops *rw;
	if(!PyArg_ParseTuple(arg, "O|s", &file, &name))
		return NULL;

	VIDEO_INIT_CHECK();

	if(PyString_Check(file))
	{
		name = PyString_AsString(file);
		surf = IMG_Load(name);
	}
	else
	{
		if(!name && PyFile_Check(file))
			name = PyString_AsString(PyFile_Name(file));

		if(!(rw = RWopsFromPython(file)))
			return NULL;
		Py_BEGIN_ALLOW_THREADS
		surf = IMG_LoadTyped_RW(rw, 1, find_extension(name));
		Py_END_ALLOW_THREADS
	}
	if(!surf)
		return RAISE(PyExc_SDLError, IMG_GetError());

	return PySurface_New(surf);
}




static PyMethodDef image_builtins[] =
{
	{ "load", load, 1, doc_load },

	{ NULL, NULL }
};



    /*DOC*/ static char doc_pygame_image_MODULE[] =
    /*DOC*/    "Contains routines to load and (someday) save surfaces. This\n"
    /*DOC*/    "module must be manually imported, since it requires the use of\n"
    /*DOC*/    "the SDL_image library.\n"
    /*DOC*/ ;

void initimage()
{
	PyObject *module, *dict;

    /* create the module */
	module = Py_InitModule3("image", image_builtins, doc_pygame_image_MODULE);
	dict = PyModule_GetDict(module);

	/*imported needed apis*/
	import_pygame_base();
	import_pygame_surface();
	import_pygame_rwobject();
}

