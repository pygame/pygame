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
 *  image module for pygame
 */
#include "pygame.h"


static int is_extended = 0;




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
    /*DOC*/    "If pygame was installed without SDL_image support, the load\n"
    /*DOC*/    "will only work with BMP images. You can test if SDL_image is\n"
    /*DOC*/    "available with the get_extended() function. These extended\n"
    /*DOC*/    "file formats usually include GIF, PNG, JPG, PCX, TGA, and more.\n"
    /*DOC*/ ;

static PyObject* image_load_basic(PyObject* self, PyObject* arg)
{
	PyObject* file, *final;
	char* name = NULL;
	SDL_Surface* surf;
	SDL_RWops *rw;
	if(!PyArg_ParseTuple(arg, "O|s", &file, &name))
		return NULL;

	VIDEO_INIT_CHECK();

	if(PyString_Check(file))
	{
		name = PyString_AsString(file);
		Py_BEGIN_ALLOW_THREADS
		surf = SDL_LoadBMP(name);
		Py_END_ALLOW_THREADS
	}
	else
	{
		if(!name && PyFile_Check(file))
			name = PyString_AsString(PyFile_Name(file));

		if(!(rw = RWopsFromPython(file)))
			return NULL;
		if(RWopsCheckPython(rw))
			surf = SDL_LoadBMP_RW(rw, 1);
		else
		{
			Py_BEGIN_ALLOW_THREADS
			surf = SDL_LoadBMP_RW(rw, 1);
			Py_END_ALLOW_THREADS
		}
	}
	if(!surf)
		return RAISE(PyExc_SDLError, SDL_GetError());

	final = PySurface_New(surf);
	if(!final)
		SDL_FreeSurface(surf);
	return final;
}



    /*DOC*/ static char doc_save[] =
    /*DOC*/    "pygame.image.save(Surface, file) -> None\n"
    /*DOC*/    "save surface as BMP data\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will save your surface in the BMP format. The given file\n"
    /*DOC*/    "argument can be either a filename or a python file-like object\n"
    /*DOC*/    "to save the BMP image to. This will also work for an opengl\n"
    /*DOC*/    "display surface.\n"
    /*DOC*/ ;

PyObject* image_save(PyObject* self, PyObject* arg)
{
	PyObject* surfobj, *file;
	SDL_Surface *surf;
	SDL_Surface *temp = NULL;
	int i, result;
	
	if(!PyArg_ParseTuple(arg, "O!O", &PySurface_Type, &surfobj, &file))
		return NULL;
	surf = PySurface_AsSurface(surfobj);

	if(surf->flags & SDL_OPENGL)
	{
		/*we need to get ahold of the pyopengl glReadPixels function*/
		/*we use pyopengl's so we don't need to link with opengl at compiletime*/
		PyObject *pyopengl, *readpixels = NULL;
		int typeflag, formatflag;

		pyopengl = PyImport_ImportModule("OpenGL.GL");
		if(pyopengl)
		{
			PyObject* dict = PyModule_GetDict(pyopengl);
			if(dict)
			{
				formatflag = PyInt_AsLong(PyDict_GetItemString(dict, "GL_RGB"));
				typeflag = PyInt_AsLong(PyDict_GetItemString(dict, "GL_UNSIGNED_BYTE"));
				readpixels = PyDict_GetItemString(dict, "glReadPixels");
			}
			Py_DECREF(pyopengl);
		}

		if(readpixels)
		{
			unsigned char *pixels;
			PyObject *data;

			data = PyObject_CallFunction(readpixels, "iiiiii", 
						0, 0, surf->w, surf->h, formatflag, typeflag);
			if(!data)
				return NULL;
			pixels = PyString_AsString(data);

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
#define IMGMASKS 0x000000FF, 0x0000FF00, 0x00FF0000, 0
#else
#define IMGMASKS 0x00FF0000, 0x0000FF00, 0x000000FF, 0
#endif

			temp = SDL_CreateRGBSurface(SDL_SWSURFACE, surf->w, surf->h, 24, IMGMASKS);
			if(!temp)
			{
				Py_DECREF(data);
				return NULL;
			}
#undef IMGMASKS

			for(i=0; i<surf->h; ++i)
				memcpy(((char *) temp->pixels) + temp->pitch * i, pixels + 3*surf->w * (surf->h-i-1), surf->w*3);
			
			Py_DECREF(data);
		}
		else
			return RAISE(PyExc_SDLError, "Cannot locate pyopengl module for OPENGL Surface save");

		surf = temp;
	}
	else
		PySurface_Prep(surfobj);

	if(PyString_Check(file))
	{
		char* name = PyString_AsString(file);
		Py_BEGIN_ALLOW_THREADS
		result = SDL_SaveBMP(surf, name);
		Py_END_ALLOW_THREADS
	}
	else
	{
		SDL_RWops* rw;
		if(!(rw = RWopsFromPython(file)))
			return NULL;
		result = SDL_SaveBMP_RW(surf, rw, 1);
	}


	if(temp)
		SDL_FreeSurface(temp);
	else
		PySurface_Unprep(surfobj);

	if(result == -1)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE
}



    /*DOC*/ static char doc_get_extended[] =
    /*DOC*/    "pygame.image.get_extended() -> int\n"
    /*DOC*/    "save surface as BMP data\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will return a true value if the extended image formats\n"
    /*DOC*/    "from SDL_image are available for loading.\n"
    /*DOC*/ ;

PyObject* image_get_extended(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;
	return PyInt_FromLong(is_extended);
}


static PyMethodDef image_builtins[] =
{
	{ "load_basic", image_load_basic, 1, doc_load },
	{ "save", image_save, 1, doc_save },
	{ "get_extended", image_get_extended, 1, doc_get_extended },

	{ NULL, NULL }
};



    /*DOC*/ static char doc_pygame_image_MODULE[] =
    /*DOC*/    "This module contains functions to transfer images in and out\n"
    /*DOC*/    "of Surfaces. At the minimum the included load() function will\n"
    /*DOC*/    "support BMP files. If SDL_image is properly installed when\n"
    /*DOC*/    "pygame is installed, it will support all the formats included\n"
    /*DOC*/    "with SDL_image. You can call the get_extended() function to test\n"
    /*DOC*/    "if the SDL_image support is available.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Some functions that communicate with other libraries will require\n"
    /*DOC*/    "that those libraries are properly installed. For example, the save()\n"
    /*DOC*/    "function can only save OPENGL surfaces if pyopengl is available.\n"
    /*DOC*/ ;

PYGAME_EXPORT
void initimage(void)
{
	PyObject *module, *dict;
	PyObject *extmodule;

    /* create the module */
	module = Py_InitModule3("image", image_builtins, doc_pygame_image_MODULE);
	dict = PyModule_GetDict(module);


	/* try to get extended formats */
	extmodule = PyImport_ImportModule("pygame.imageext");
	if(extmodule)
	{
		PyObject *extdict = PyModule_GetDict(extmodule);
		PyObject* extload = PyDict_GetItemString(extdict, "load_extended");
		PyDict_SetItemString(dict, "load_extended", extload);
		PyDict_SetItemString(dict, "load", extload);
		Py_INCREF(extload);
		Py_INCREF(extload);
		is_extended = 1;
	}
	else
	{
		PyObject* basicload = PyDict_GetItemString(dict, "load_basic");
		PyErr_Clear();
		PyDict_SetItemString(dict, "load_extended", Py_None);
		PyDict_SetItemString(dict, "load", basicload);
		Py_INCREF(Py_None);
		Py_INCREF(basicload);
		is_extended = 0;
	}


	/*imported needed apis*/
	import_pygame_base();
	import_pygame_surface();
	import_pygame_rwobject();
}

