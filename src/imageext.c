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
 *  extended image module for pygame, note this only has
 *  the extended load function, which is autmatically used
 *  by the normal pygame.image module if it is available.
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



static PyObject* image_load_ext(PyObject* self, PyObject* arg)
{
	PyObject* file, *final;
	char* name = NULL;
	SDL_Surface* surf;
	SDL_RWops *rw;
	if(!PyArg_ParseTuple(arg, "O|s", &file, &name))
		return NULL;
	if(PyString_Check(file) || PyUnicode_Check(file))
	{
		if(!PyArg_ParseTuple(arg, "s|O", &name, &file))
			return NULL;
		Py_BEGIN_ALLOW_THREADS
		surf = IMG_Load(name);
		Py_END_ALLOW_THREADS
	}
	else
	{
		if(!name && PyFile_Check(file))
			name = PyString_AsString(PyFile_Name(file));

		if(!(rw = RWopsFromPython(file)))
			return NULL;
		if(RWopsCheckPython(rw))
                {
			surf = IMG_LoadTyped_RW(rw, 1, find_extension(name));
                }
		else
		{
			Py_BEGIN_ALLOW_THREADS
			surf = IMG_LoadTyped_RW(rw, 1, find_extension(name));
			Py_END_ALLOW_THREADS
		}
	}

	if(!surf)
		return RAISE(PyExc_SDLError, IMG_GetError());

	final = PySurface_New(surf);
	if(!final)
		SDL_FreeSurface(surf);
	return final;
}




static PyMethodDef image_builtins[] =
{
	{ "load_extended", image_load_ext, 1, NULL },

	{ NULL, NULL }
};



PYGAME_EXPORT
void initimageext(void)
{
    /* create the module */
	Py_InitModule3("imageext", image_builtins, NULL);

	/*imported needed apis*/
	import_pygame_base();
	import_pygame_surface();
	import_pygame_rwobject();
}

