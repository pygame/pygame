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
		int typeflag=0, formatflag=0;

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


    /*DOC*/ static char doc_tostring[] =
    /*DOC*/    "pygame.image.tostring(Surface, format) -> string\n"
    /*DOC*/    "create a raw string buffer of the surface data\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will copy the image data into a large string buffer.\n"
    /*DOC*/    "This can be used to transfer images to other libraries like\n"
    /*DOC*/    "PIL's fromstring() and PyOpenGL's glTexImage2D(). \n"
    /*DOC*/    "\n"
    /*DOC*/    "The format argument is a string representing which type of\n"
    /*DOC*/    "string data you need. It can be one of the following, \"P\"\n"
    /*DOC*/    "for 8bit palette indices. \"RGB\" for 24bit RGB data, \"RGBA\"\n"
    /*DOC*/    "for 32bit RGB and alpha, or \"RGBX\" for 32bit padded RGB colors.\n"
    /*DOC*/    "\n"
    /*DOC*/    "These flags are a subset of the formats supported the PIL\n"
    /*DOC*/    "Python Image Library. Note that the \"P\" format only will\n"
    /*DOC*/    "work for 8bit Surfaces.\n"
    /*DOC*/ ;

PyObject* image_tostring(PyObject* self, PyObject* arg)
{
	PyObject *surfobj, *string=NULL;
	char *format, *data, *pixels;
	SDL_Surface *surf;
	int w, h, color, len;
	int Rmask, Gmask, Bmask, Amask, Rshift, Gshift, Bshift, Ashift, Rloss, Gloss, Bloss, Aloss;

	if(!PyArg_ParseTuple(arg, "O!s", &PySurface_Type, &surfobj, &format))
		return NULL;
	surf = PySurface_AsSurface(surfobj);

	Rmask = surf->format->Rmask; Gmask = surf->format->Gmask;
	Bmask = surf->format->Bmask; Amask = surf->format->Amask;
	Rshift = surf->format->Rshift; Gshift = surf->format->Gshift;
	Bshift = surf->format->Bshift; Ashift = surf->format->Ashift;
	Rloss = surf->format->Rloss; Gloss = surf->format->Gloss;
	Bloss = surf->format->Bloss; Aloss = surf->format->Aloss;

	if(!strcmp(format, "P"))
	{
		if(surf->format->BytesPerPixel != 1)
			return RAISE(PyExc_ValueError, "Can only create \"P\" format data with 8bit Surfaces");
		string = PyString_FromStringAndSize(NULL, surf->w*surf->h);
		if(!string)
			return NULL;
		PyString_AsStringAndSize(string, &data, &len);

		PySurface_Lock(surfobj);
		pixels = (char*)surf->pixels;
		for(h=0; h<surf->h; ++h)
			memcpy(data+(h*surf->w), pixels+(h*surf->pitch), surf->w);
		PySurface_Unlock(surfobj);
	}
	else if(!strcmp(format, "RGB"))
	{
		string = PyString_FromStringAndSize(NULL, surf->w*surf->h*3);
		if(!string)
			return NULL;
		PyString_AsStringAndSize(string, &data, &len);

		PySurface_Lock(surfobj);
		pixels = (char*)surf->pixels;
		switch(surf->format->BytesPerPixel)
		{
		case 1:
			for(h=0; h<surf->h; ++h)
			{
				Uint8* ptr = (Uint8*)((Uint8*)surf->pixels + (h*surf->pitch));
				for(w=0; w<surf->w; ++w)
				{
					color = *ptr++;
					data[0] = (char)surf->format->palette->colors[color].r;
					data[1] = (char)surf->format->palette->colors[color].g;
					data[2] = (char)surf->format->palette->colors[color].b;
					data += 3;
				}
			}break;
		case 2:
			for(h=0; h<surf->h; ++h)
			{
				Uint16* ptr = (Uint16*)((Uint8*)surf->pixels + (h*surf->pitch));
				for(w=0; w<surf->w; ++w)
				{
					color = *ptr++;
					data[0] = (char)(((color & Rmask) >> Rshift) << Rloss);
					data[1] = (char)(((color & Gmask) >> Gshift) << Gloss);
					data[2] = (char)(((color & Bmask) >> Bshift) << Bloss);
					data += 3;
				}
			}break;
		case 3:
			for(h=0; h<surf->h; ++h)
			{
				Uint8* ptr = (Uint8*)((Uint8*)surf->pixels + (h*surf->pitch));
				for(w=0; w<surf->w; ++w)
				{
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
					color = ptr[0] + (ptr[1]<<8) + (ptr[2]<<16);
#else
					color = ptr[2] + (ptr[1]<<8) + (ptr[0]<<16);
#endif
					ptr += 3;
					data[0] = (char)(((color & Rmask) >> Rshift) << Rloss);
					data[1] = (char)(((color & Gmask) >> Gshift) << Gloss);
					data[2] = (char)(((color & Bmask) >> Bshift) << Bloss);
					data += 3;
				}
			}break;
		case 4:
			for(h=0; h<surf->h; ++h)
			{
				Uint32* ptr = (Uint32*)((Uint8*)surf->pixels + (h*surf->pitch));
				for(w=0; w<surf->w; ++w)
				{
					color = *ptr++;
					data[0] = (char)(((color & Rmask) >> Rshift) << Rloss);
					data[1] = (char)(((color & Gmask) >> Gshift) << Rloss);
					data[2] = (char)(((color & Bmask) >> Bshift) << Rloss);
					data += 3;
				}
			}break;
		}
		PySurface_Unlock(surfobj);
	}
	else if(!strcmp(format, "RGBX") || !strcmp(format, "RGBA"))
	{
		string = PyString_FromStringAndSize(NULL, surf->w*surf->h*4);
		if(!string)
			return NULL;
		PyString_AsStringAndSize(string, &data, &len);

		PySurface_Lock(surfobj);
		pixels = (char*)surf->pixels;
		switch(surf->format->BytesPerPixel)
		{
		case 1:
			for(h=0; h<surf->h; ++h)
			{
				Uint8* ptr = (Uint8*)((Uint8*)surf->pixels + (h*surf->pitch));
				for(w=0; w<surf->w; ++w)
				{
					color = *ptr++;
					data[0] = (char)surf->format->palette->colors[color].r;
					data[1] = (char)surf->format->palette->colors[color].g;
					data[2] = (char)surf->format->palette->colors[color].b;
					data[3] = (char)255;
					data += 4;
				}
			}break;
		case 2:
			for(h=0; h<surf->h; ++h)
			{
				Uint16* ptr = (Uint16*)((Uint8*)surf->pixels + (h*surf->pitch));
				for(w=0; w<surf->w; ++w)
				{
					color = *ptr++;
					data[0] = (char)(((color & Rmask) >> Rshift) << Rloss);
					data[1] = (char)(((color & Gmask) >> Gshift) << Gloss);
					data[2] = (char)(((color & Bmask) >> Bshift) << Bloss);
					data[3] = (char)(Amask ? (((color & Amask) >> Ashift) << Aloss) : 255);
					data += 4;
				}
			}break;
		case 3:
			for(h=0; h<surf->h; ++h)
			{
				Uint8* ptr = (Uint8*)((Uint8*)surf->pixels + (h*surf->pitch));
				for(w=0; w<surf->w; ++w)
				{
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
					color = ptr[0] + (ptr[1]<<8) + (ptr[2]<<16);
#else
					color = ptr[2] + (ptr[1]<<8) + (ptr[0]<<16);
#endif
					ptr += 3;
					data[0] = (char)(((color & Rmask) >> Rshift) << Rloss);
					data[1] = (char)(((color & Gmask) >> Gshift) << Gloss);
					data[2] = (char)(((color & Bmask) >> Bshift) << Bloss);
					data[3] = (char)(Amask ? (((color & Amask) >> Ashift) << Aloss) : 255);
					data += 4;
				}
			}break;
		case 4:
			for(h=0; h<surf->h; ++h)
			{
				Uint32* ptr = (Uint32*)((Uint8*)surf->pixels + (h*surf->pitch));
				for(w=0; w<surf->w; ++w)
				{
					color = *ptr++;
					data[0] = (char)(((color & Rmask) >> Rshift) << Rloss);
					data[1] = (char)(((color & Gmask) >> Gshift) << Rloss);
					data[2] = (char)(((color & Bmask) >> Bshift) << Rloss);
					data[3] = (char)(Amask ? (((color & Amask) >> Ashift) << Rloss) : 255);
					data += 4;
				}
			}break;
		}
		PySurface_Unlock(surfobj);
	}
	else
		return RAISE(PyExc_ValueError, "Unrecognized type of format");

	return string;
}



    /*DOC*/ static char doc_fromstring[] =
    /*DOC*/    "pygame.image.fromstring(size, format, string) -> Surface\n"
    /*DOC*/    "create a surface from a raw string buffer\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will create a new Surface from a copy of raw data in\n"
    /*DOC*/    "a string. This can be used to transfer images from other\n"
    /*DOC*/    "libraries like PIL's fromstring(). \n"
    /*DOC*/    "\n"
    /*DOC*/    "The format argument is a string representing which type of\n"
    /*DOC*/    "string data you need. It can be one of the following, \"P\"\n"
    /*DOC*/    "for 8bit palette indices. \"RGB\" for 24bit RGB data, \"RGBA\"\n"
    /*DOC*/    "for 32bit RGB and alpha, or \"RGBX\" for 32bit padded RGB colors.\n"
    /*DOC*/    "\n"
    /*DOC*/    "These flags are a subset of the formats supported the PIL\n"
    /*DOC*/    "Python Image Library. Note that the \"P\" format only create\n"
    /*DOC*/    "an 8bit surface, but the colormap will be all black.\n"
    /*DOC*/ ;

PyObject* image_fromstring(PyObject* self, PyObject* arg)
{
	PyObject *string;
	char *format, *data, *pixels;
	SDL_Surface *surf = NULL;
	int w, h, len;
	int loopw, looph;

	if(!PyArg_ParseTuple(arg, "(ii)sO!", &w, &h, &format, &PyString_Type, &string))
		return NULL;

	if(w < 1 || h < 1)
		return RAISE(PyExc_ValueError, "Resolution must be positive values");

	PyString_AsStringAndSize(string, &data, &len);

	if(!strcmp(format, "P"))
	{
		if(len != w*h)
			return RAISE(PyExc_ValueError, "String length does not equal format and resolution size");
		surf = SDL_CreateRGBSurface(0, w, h, 8, 0, 0, 0, 0);
		if(!surf)
			return RAISE(PyExc_SDLError, SDL_GetError());
		SDL_LockSurface(surf);
		pixels = (char*)surf->pixels;
		for(looph=0; looph<h; ++looph)
			memcpy(pixels+looph*surf->pitch, data+looph*w, w);
		SDL_UnlockSurface(surf);
	}
	else if(!strcmp(format, "RGB"))
	{
		if(len != w*h*3)
			return RAISE(PyExc_ValueError, "String length does not equal format and resolution size");
		surf = SDL_CreateRGBSurface(0, w, h, 24, 0xFF<<16, 0xFF<<8, 0xFF, 0);
		if(!surf)
			return RAISE(PyExc_SDLError, SDL_GetError());
		SDL_LockSurface(surf);
		pixels = (char*)surf->pixels;
		for(looph=0; looph<h; ++looph)
		{
			Uint8* pix = (Uint8*)(pixels+looph*surf->pitch);
			for(loopw=0; loopw<w; ++loopw)
			{
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
				pix[2] = data[0]; pix[1] = data[1]; pix[0] = data[2];
#else
				pix[0] = data[0]; pix[1] = data[1]; pix[2] = data[2];
#endif
				pix += 3;
				data += 3;
			}
		}
		SDL_UnlockSurface(surf);
	}
	else if(!strcmp(format, "RGBA") || !strcmp(format, "RGBX"))
	{
		int alphamult = !strcmp(format, "RGBA");
		if(len != w*h*4)
			return RAISE(PyExc_ValueError, "String length does not equal format and resolution size");
		surf = SDL_CreateRGBSurface(0, w, h, 32, 0xFF<<16, 0xFF<<8, 0xFF,
					(!strcmp(format, "RGBA")) ? 0xFF<<24 : 0);
		if(!surf)
			return RAISE(PyExc_SDLError, SDL_GetError());
		SDL_LockSurface(surf);
		pixels = (char*)surf->pixels;
		for(looph=0; looph<h; ++looph)
		{
			Uint32* pix = (Uint32*)(pixels+looph*surf->pitch);
			for(loopw=0; loopw<w; ++loopw)
			{
				*pix++ = data[0]<<16 | data[1]<<8 | data[2] | (data[3]*alphamult) << 24;
				data += 4;
			}
		}
		SDL_UnlockSurface(surf);
	}
	else
		return RAISE(PyExc_ValueError, "Unrecognized type of format");

	if(!surf)
		return NULL;
	return PySurface_New(surf);
}

static PyMethodDef image_builtins[] =
{
	{ "load_basic", image_load_basic, 1, doc_load },
	{ "save", image_save, 1, doc_save },
	{ "get_extended", image_get_extended, 1, doc_get_extended },

	{ "tostring", image_tostring, 1, doc_tostring },
	{ "fromstring", image_fromstring, 1, doc_fromstring },

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

