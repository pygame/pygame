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
static int SaveTGA(SDL_Surface *surface, char *file, int rle);
static int SaveTGA_RW(SDL_Surface *surface, SDL_RWops *out, int rle);
static SDL_Surface* opengltosdl(void);


#define DATAROW(data, row, width, height, flipped) \
			((flipped) ? (((char*)data)+(height-row-1)*width) : (((char*)data)+row*width))


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
    /*DOC*/    "\n"
    /*DOC*/    "If the image format supports colorkeys and pixel alphas, the\n"
    /*DOC*/    "load() function will properly load and configure these types\n"
    /*DOC*/    "of transparency.\n"
    /*DOC*/ ;

static PyObject* image_load_basic(PyObject* self, PyObject* arg)
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




static SDL_Surface* opengltosdl()
{
        /*we need to get ahold of the pyopengl glReadPixels function*/
        /*we use pyopengl's so we don't need to link with opengl at compiletime*/
        PyObject *pyopengl, *readpixels = NULL;
        int typeflag=0, formatflag=0;
        SDL_Surface *surf;
        Uint32 rmask, gmask, bmask;
        int i;
        unsigned char *pixels;
        PyObject *data;

        surf = SDL_GetVideoSurface();

        pyopengl = PyImport_ImportModule("OpenGL.GL");
        if(pyopengl)
        {
                PyObject* dict = PyModule_GetDict(pyopengl);
                if(dict)
                {
                        PyObject *o;
                        o = PyDict_GetItemString(dict, "GL_RGB");
                        if(!o) {Py_DECREF(pyopengl); return NULL;}
                        formatflag = PyInt_AsLong(o);
                        o = PyDict_GetItemString(dict, "GL_UNSIGNED_BYTE");
                        if(!o) {Py_DECREF(pyopengl); return NULL;}
                        typeflag = PyInt_AsLong(o);
                        readpixels = PyDict_GetItemString(dict, "glReadPixels");
                        if(!readpixels) {Py_DECREF(pyopengl); return NULL;}
                }
                Py_DECREF(pyopengl);
        }
        else
        {
            RAISE(PyExc_ImportError, "Cannot import PyOpenGL");
            return NULL;
        }

        data = PyObject_CallFunction(readpixels, "iiiiii",
                                0, 0, surf->w, surf->h, formatflag, typeflag);
        if(!data)
        {
                RAISE(PyExc_SDLError, "glReadPixels returned NULL");
                return NULL;
        }
        pixels = (unsigned char*)PyString_AsString(data);

        if(SDL_BYTEORDER == SDL_LIL_ENDIAN)
        {
            rmask=0x000000FF; gmask=0x0000FF00; bmask=0x00FF0000;
        }
        else
        {
            rmask=0x00FF0000; gmask=0x0000FF00; bmask=0x000000FF;
        }
        surf = SDL_CreateRGBSurface(SDL_SWSURFACE, surf->w, surf->h, 24,
                    rmask, gmask, bmask, 0);
        if(!surf)
        {
                Py_DECREF(data);
                RAISE(PyExc_SDLError, SDL_GetError());
                return NULL;
        }

        for(i=0; i<surf->h; ++i)
                memcpy(((char *) surf->pixels) + surf->pitch * i, pixels + 3*surf->w * (surf->h-i-1), surf->w*3);

        Py_DECREF(data);
        return surf;
}




    /*DOC*/ static char doc_save[] =
    /*DOC*/    "pygame.image.save(Surface, file) -> None\n"
    /*DOC*/    "save surface data\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will save your surface as a BMP or TGA image. The given\n"
    /*DOC*/    "file argument can be either a filename or a python file-like\n"
    /*DOC*/    "object. This will also work under OPENGL display modes.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The image will default to save with the TGA format. If the\n"
    /*DOC*/    "filename has the BMP extension, it will use the BMP format.\n"
    /*DOC*/ ;

PyObject* image_save(PyObject* self, PyObject* arg)
{
	PyObject* surfobj, *file;
	SDL_Surface *surf;
	SDL_Surface *temp = NULL;
	int result;

	if(!PyArg_ParseTuple(arg, "O!O", &PySurface_Type, &surfobj, &file))
		return NULL;
	surf = PySurface_AsSurface(surfobj);

	if(surf->flags & SDL_OPENGL)
	{
                temp = surf = opengltosdl();
                if(!surf)
                    return NULL;
	}
	else
		PySurface_Prep(surfobj);

	if(PyString_Check(file) || PyUnicode_Check(file))
	{
                int namelen;
		char* name;
		if(!PyArg_ParseTuple(arg, "O|s", &file, &name))
			return NULL;
                namelen = strlen(name);
		Py_BEGIN_ALLOW_THREADS
                if(name[namelen-1]=='p' || name[namelen-1]=='P')
		    result = SDL_SaveBMP(surf, name);
                else
                    result = SaveTGA(surf, name, 1);
		Py_END_ALLOW_THREADS
	}
	else
	{
		SDL_RWops* rw;
		if(!(rw = RWopsFromPython(file)))
			return NULL;
/*		result = SDL_SaveBMP_RW(surf, rw, 1);*/
		result = SaveTGA_RW(surf, rw, 1);
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
    /*DOC*/    "returns true if SDL_image formats are available\n"
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
    /*DOC*/    "pygame.image.tostring(Surface, format, flipped=0) -> string\n"
    /*DOC*/    "create a raw string buffer of the surface data\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will copy the image data into a large string buffer.\n"
    /*DOC*/    "This can be used to transfer images to other libraries like\n"
    /*DOC*/    "PIL's fromstring() and PyOpenGL's glTexImage2D(). \n"
    /*DOC*/    "\n"
    /*DOC*/    "The flipped argument will cause the output string to have\n"
    /*DOC*/    "it's contents flipped vertically.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The format argument is a string representing which type of\n"
    /*DOC*/    "string data you need. It can be one of the following, \"P\"\n"
    /*DOC*/    "for 8bit palette indices. \"RGB\" for 24bit RGB data, \"RGBA\"\n"
    /*DOC*/    "for 32bit RGB and alpha, or \"RGBX\" for 32bit padded RGB colors.\n"
    /*DOC*/    "\"ARGB\" is a popular format for big endian platforms.\n"
    /*DOC*/    "\n"
    /*DOC*/    "These flags are a subset of the formats supported the PIL\n"
    /*DOC*/    "Python Image Library. Note that the \"P\" format only will\n"
    /*DOC*/    "work for 8bit Surfaces.\n"
    /*DOC*/    "\n"
    /*DOC*/    "If you ask for the \"RGBA\" format and the image only has\n"
    /*DOC*/    "colorkey data. An alpha channel will be created from the\n"
    /*DOC*/    "colorkey values.\n"
    /*DOC*/ ;

PyObject* image_tostring(PyObject* self, PyObject* arg)
{
	PyObject *surfobj, *string=NULL;
	char *format, *data, *pixels;
	SDL_Surface *surf, *temp=NULL;
	int w, h, color, len, flipped=0;
	int Rmask, Gmask, Bmask, Amask, Rshift, Gshift, Bshift, Ashift, Rloss, Gloss, Bloss, Aloss;
	int hascolorkey, colorkey;

	if(!PyArg_ParseTuple(arg, "O!s|i", &PySurface_Type, &surfobj, &format, &flipped))
		return NULL;
	surf = PySurface_AsSurface(surfobj);
	if(surf->flags & SDL_OPENGL)
	{
                temp = surf = opengltosdl();
                if(!surf)
                    return NULL;
	}

	Rmask = surf->format->Rmask; Gmask = surf->format->Gmask;
	Bmask = surf->format->Bmask; Amask = surf->format->Amask;
	Rshift = surf->format->Rshift; Gshift = surf->format->Gshift;
	Bshift = surf->format->Bshift; Ashift = surf->format->Ashift;
	Rloss = surf->format->Rloss; Gloss = surf->format->Gloss;
	Bloss = surf->format->Bloss; Aloss = surf->format->Aloss;
	hascolorkey = (surf->flags & SDL_SRCCOLORKEY) && !Amask;
	colorkey = surf->format->colorkey;

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
			memcpy(DATAROW(data, h, surf->w, surf->h, flipped), pixels+(h*surf->pitch), surf->w);
		PySurface_Unlock(surfobj);
	}
	else if(!strcmp(format, "RGB"))
	{
		string = PyString_FromStringAndSize(NULL, surf->w*surf->h*3);
		if(!string)
			return NULL;
		PyString_AsStringAndSize(string, &data, &len);

		if(!temp)
                    PySurface_Lock(surfobj);
		pixels = (char*)surf->pixels;
		switch(surf->format->BytesPerPixel)
		{
		case 1:
			for(h=0; h<surf->h; ++h)
			{
				Uint8* ptr = (Uint8*)DATAROW(surf->pixels, h, surf->pitch, surf->h, flipped);
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
				Uint16* ptr = (Uint16*)DATAROW(surf->pixels, h, surf->pitch, surf->h, flipped);
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
				Uint8* ptr = (Uint8*)DATAROW(surf->pixels, h, surf->pitch, surf->h, flipped);
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
				Uint32* ptr = (Uint32*)DATAROW(surf->pixels, h, surf->pitch, surf->h, flipped);
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
		if(!temp)
                    PySurface_Unlock(surfobj);
	}
	else if(!strcmp(format, "RGBX") || !strcmp(format, "RGBA"))
	{
		if(strcmp(format, "RGBA"))
			hascolorkey = 0;

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
				Uint8* ptr = (Uint8*)DATAROW(surf->pixels, h, surf->pitch, surf->h, flipped);
				for(w=0; w<surf->w; ++w)
				{
					color = *ptr++;
					data[0] = (char)surf->format->palette->colors[color].r;
					data[1] = (char)surf->format->palette->colors[color].g;
					data[2] = (char)surf->format->palette->colors[color].b;
					data[3] = hascolorkey ? (char)(color!=colorkey)*255 : (char)255;
					data += 4;
				}
			}break;
		case 2:
			for(h=0; h<surf->h; ++h)
			{
				Uint16* ptr = (Uint16*)DATAROW(surf->pixels, h, surf->pitch, surf->h, flipped);
				for(w=0; w<surf->w; ++w)
				{
					color = *ptr++;
					data[0] = (char)(((color & Rmask) >> Rshift) << Rloss);
					data[1] = (char)(((color & Gmask) >> Gshift) << Gloss);
					data[2] = (char)(((color & Bmask) >> Bshift) << Bloss);
					data[3] = hascolorkey ? (char)(color!=colorkey)*255 :
								(char)(Amask ? (((color & Amask) >> Ashift) << Aloss) : 255);
					data += 4;
				}
			}break;
		case 3:
			for(h=0; h<surf->h; ++h)
			{
				Uint8* ptr = (Uint8*)DATAROW(surf->pixels, h, surf->pitch, surf->h, flipped);
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
					data[3] = hascolorkey ? (char)(color!=colorkey)*255 :
								(char)(Amask ? (((color & Amask) >> Ashift) << Aloss) : 255);
					data += 4;
				}
			}break;
		case 4:
			for(h=0; h<surf->h; ++h)
			{
				Uint32* ptr = (Uint32*)DATAROW(surf->pixels, h, surf->pitch, surf->h, flipped);
				for(w=0; w<surf->w; ++w)
				{
					color = *ptr++;
					data[0] = (char)(((color & Rmask) >> Rshift) << Rloss);
					data[1] = (char)(((color & Gmask) >> Gshift) << Rloss);
					data[2] = (char)(((color & Bmask) >> Bshift) << Rloss);
					data[3] = hascolorkey ? (char)(color!=colorkey)*255 :
								(char)(Amask ? (((color & Amask) >> Ashift) << Rloss) : 255);
					data += 4;
				}
			}break;
		}
		PySurface_Unlock(surfobj);
	}
	else if(!strcmp(format, "ARGB"))
	{
		hascolorkey = 0;

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
				Uint8* ptr = (Uint8*)DATAROW(surf->pixels, h, surf->pitch, surf->h, flipped);
				for(w=0; w<surf->w; ++w)
				{
					color = *ptr++;
					data[1] = (char)surf->format->palette->colors[color].r;
					data[2] = (char)surf->format->palette->colors[color].g;
					data[3] = (char)surf->format->palette->colors[color].b;
					data[0] = hascolorkey ? (char)(color!=colorkey)*255 : (char)255;
					data += 4;
				}
			}break;
		case 2:
			for(h=0; h<surf->h; ++h)
			{
				Uint16* ptr = (Uint16*)DATAROW(surf->pixels, h, surf->pitch, surf->h, flipped);
				for(w=0; w<surf->w; ++w)
				{
					color = *ptr++;
					data[1] = (char)(((color & Rmask) >> Rshift) << Rloss);
					data[2] = (char)(((color & Gmask) >> Gshift) << Gloss);
					data[3] = (char)(((color & Bmask) >> Bshift) << Bloss);
					data[0] = hascolorkey ? (char)(color!=colorkey)*255 :
								(char)(Amask ? (((color & Amask) >> Ashift) << Aloss) : 255);
					data += 4;
				}
			}break;
		case 3:
			for(h=0; h<surf->h; ++h)
			{
				Uint8* ptr = (Uint8*)DATAROW(surf->pixels, h, surf->pitch, surf->h, flipped);
				for(w=0; w<surf->w; ++w)
				{
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
					color = ptr[0] + (ptr[1]<<8) + (ptr[2]<<16);
#else
					color = ptr[2] + (ptr[1]<<8) + (ptr[0]<<16);
#endif
					ptr += 3;
					data[1] = (char)(((color & Rmask) >> Rshift) << Rloss);
					data[2] = (char)(((color & Gmask) >> Gshift) << Gloss);
					data[3] = (char)(((color & Bmask) >> Bshift) << Bloss);
					data[0] = hascolorkey ? (char)(color!=colorkey)*255 :
								(char)(Amask ? (((color & Amask) >> Ashift) << Aloss) : 255);
					data += 4;
				}
			}break;
		case 4:
			for(h=0; h<surf->h; ++h)
			{
				Uint32* ptr = (Uint32*)DATAROW(surf->pixels, h, surf->pitch, surf->h, flipped);
				for(w=0; w<surf->w; ++w)
				{
					color = *ptr++;
					data[1] = (char)(((color & Rmask) >> Rshift) << Rloss);
					data[2] = (char)(((color & Gmask) >> Gshift) << Rloss);
					data[3] = (char)(((color & Bmask) >> Bshift) << Rloss);
					data[0] = hascolorkey ? (char)(color!=colorkey)*255 :
								(char)(Amask ? (((color & Amask) >> Ashift) << Rloss) : 255);
					data += 4;
				}
			}break;
		}
		PySurface_Unlock(surfobj);
	}
	else
        {
                if(temp) SDL_FreeSurface(temp);
		return RAISE(PyExc_ValueError, "Unrecognized type of format");
        }

        if(temp) SDL_FreeSurface(temp);
	return string;
}



    /*DOC*/ static char doc_fromstring[] =
    /*DOC*/    "pygame.image.fromstring(string, size, format, flipped=0) -> Surface\n"
    /*DOC*/    "create a surface from a raw string buffer\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will create a new Surface from a copy of raw data in\n"
    /*DOC*/    "a string. This can be used to transfer images from other\n"
    /*DOC*/    "libraries like PIL's fromstring(). \n"
    /*DOC*/    "\n"
    /*DOC*/    "The flipped argument should be set to true if the image in\n"
    /*DOC*/    "the string is.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The format argument is a string representing which type of\n"
    /*DOC*/    "string data you need. It can be one of the following, \"P\"\n"
    /*DOC*/    "for 8bit palette indices. \"RGB\" for 24bit RGB data, \"RGBA\"\n"
    /*DOC*/    "for 32bit RGB and alpha, or \"RGBX\" for 32bit padded RGB colors.\n"
    /*DOC*/    "\"ARGB\" is a popular format for big endian platforms.\n"
    /*DOC*/    "\n"
    /*DOC*/    "These flags are a subset of the formats supported the PIL\n"
    /*DOC*/    "Python Image Library. Note that the \"P\" format only create\n"
    /*DOC*/    "an 8bit surface, but the colormap will be all black.\n"
    /*DOC*/ ;

PyObject* image_fromstring(PyObject* self, PyObject* arg)
{
	PyObject *string;
	char *format, *data;
	SDL_Surface *surf = NULL;
	int w, h, len, flipped=0;
	int loopw, looph;

	if(!PyArg_ParseTuple(arg, "O!(ii)s|i", &PyString_Type, &string, &w, &h, &format, &flipped))
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
		for(looph=0; looph<h; ++looph)
			memcpy(((char*)surf->pixels)+looph*surf->pitch, DATAROW(data, looph, w, h, flipped), w);
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
		for(looph=0; looph<h; ++looph)
		{
			Uint8* pix = (Uint8*)DATAROW(surf->pixels, looph, surf->pitch, h, flipped);
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
		surf = SDL_CreateRGBSurface((alphamult?SDL_SRCALPHA:0), w, h, 32,
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                                        0xFF, 0xFF<<8, 0xFF<<16, (alphamult?0xFF<<24:0));
#else
                                        0xFF<<24, 0xFF<<16, 0xFF<<8, (alphamult?0xFF:0));
#endif
		if(!surf)
			return RAISE(PyExc_SDLError, SDL_GetError());
		SDL_LockSurface(surf);
		for(looph=0; looph<h; ++looph)
		{
			Uint32* pix = (Uint32*)DATAROW(surf->pixels, looph, surf->pitch, h, flipped);
			for(loopw=0; loopw<w; ++loopw)
			{
                                *pix++ = *((Uint32*)data);
                                data += 4;
			}
		}
		SDL_UnlockSurface(surf);
	}
	else if(!strcmp(format, "ARGB"))
	{
                if(len != w*h*4)
			return RAISE(PyExc_ValueError, "String length does not equal format and resolution size");
		surf = SDL_CreateRGBSurface(SDL_SRCALPHA, w, h, 32,
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                                        0xFF<<24, 0xFF, 0xFF<<8, 0xFF<<16);
#else
                                        0xFF, 0xFF<<24, 0xFF<<16, 0xFF<<8);
#endif
		if(!surf)
			return RAISE(PyExc_SDLError, SDL_GetError());
		SDL_LockSurface(surf);
		for(looph=0; looph<h; ++looph)
		{
			Uint32* pix = (Uint32*)DATAROW(surf->pixels, looph, surf->pitch, h, flipped);
			for(loopw=0; loopw<w; ++loopw)
			{
                                *pix++ = *((Uint32*)data);
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



/*******************************************************/
/* tga code by Mattias Engdegård, in the public domain */
/*******************************************************/

struct TGAheader {
    Uint8 infolen;		/* length of info field */
    Uint8 has_cmap;		/* 1 if image has colormap, 0 otherwise */
    Uint8 type;

    Uint8 cmap_start[2];	/* index of first colormap entry */
    Uint8 cmap_len[2];		/* number of entries in colormap */
    Uint8 cmap_bits;		/* bits per colormap entry */

    Uint8 yorigin[2];		/* image origin (ignored here) */
    Uint8 xorigin[2];
    Uint8 width[2];		/* image size */
    Uint8 height[2];
    Uint8 pixel_bits;		/* bits/pixel */
    Uint8 flags;
};

enum tga_type {
    TGA_TYPE_INDEXED = 1,
    TGA_TYPE_RGB = 2,
    TGA_TYPE_BW = 3,
    TGA_TYPE_RLE = 8		/* additive */
};


#define TGA_INTERLEAVE_MASK	0xc0
#define TGA_INTERLEAVE_NONE	0x00
#define TGA_INTERLEAVE_2WAY	0x40
#define TGA_INTERLEAVE_4WAY	0x80

#define TGA_ORIGIN_MASK		0x30
#define TGA_ORIGIN_LEFT		0x00
#define TGA_ORIGIN_RIGHT	0x10
#define TGA_ORIGIN_LOWER	0x00
#define TGA_ORIGIN_UPPER	0x20

/* read/write unaligned little-endian 16-bit ints */
#define LE16(p) ((p)[0] + ((p)[1] << 8))
#define SETLE16(p, v) ((p)[0] = (v), (p)[1] = (v) >> 8)

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#define TGA_RLE_MAX 128		/* max length of a TGA RLE chunk */
/* return the number of bytes in the resulting buffer after RLE-encoding
   a line of TGA data */
static int rle_line(Uint8 *src, Uint8 *dst, int w, int bpp)
{
    int x = 0;
    int out = 0;
    int raw = 0;
    while(x < w) {
	Uint32 pix;
	int x0 = x;
	memcpy(&pix, src + x * bpp, bpp);
	x++;
	while(x < w && memcmp(&pix, src + x * bpp, bpp) == 0
	      && x - x0 < TGA_RLE_MAX)
	    x++;
	/* use a repetition chunk iff the repeated pixels would consume
	   two bytes or more */
	if((x - x0 - 1) * bpp >= 2 || x == w) {
	    /* output previous raw chunks */
	    while(raw < x0) {
		int n = MIN(TGA_RLE_MAX, x0 - raw);
		dst[out++] = n - 1;
		memcpy(dst + out, src + raw * bpp, n * bpp);
		out += n * bpp;
		raw += n;
	    }

	    if(x - x0 > 0) {
		/* output new repetition chunk */
		dst[out++] = 0x7f + x - x0;
		memcpy(dst + out, &pix, bpp);
		out += bpp;
	    }
	    raw = x;
	}
    }
    return out;
}

/*
 * Save a surface to an output stream in TGA format.
 * 8bpp surfaces are saved as indexed images with 24bpp palette, or with
 *     32bpp palette if colourkeying is used.
 * 15, 16, 24 and 32bpp surfaces are saved as 24bpp RGB images,
 * or as 32bpp RGBA images if alpha channel is used.
 *
 * RLE compression is not used in the output file.
 *
 * Returns -1 upon error, 0 if success
 */
static int SaveTGA_RW(SDL_Surface *surface, SDL_RWops *out, int rle)
{
    SDL_Surface *linebuf = NULL;
    int alpha = 0;
    int ckey = -1;
    struct TGAheader h;
    int srcbpp;
    unsigned surf_flags;
    unsigned surf_alpha;
    Uint32 rmask, gmask, bmask, amask;
    SDL_Rect r;
    int bpp;
    Uint8 *rlebuf = NULL;

    h.infolen = 0;
    SETLE16(h.cmap_start, 0);

    srcbpp = surface->format->BitsPerPixel;
    if(srcbpp < 8) {
	SDL_SetError("cannot save <8bpp images as TGA");
	return -1;
    }

    if(srcbpp == 8) {
	h.has_cmap = 1;
	h.type = TGA_TYPE_INDEXED;
	if(surface->flags & SDL_SRCCOLORKEY) {
	    ckey = surface->format->colorkey;
	    h.cmap_bits = 32;
	} else
	    h.cmap_bits = 24;
	SETLE16(h.cmap_len, surface->format->palette->ncolors);
	h.pixel_bits = 8;
	rmask = gmask = bmask = amask = 0;
    } else {
	h.has_cmap = 0;
	h.type = TGA_TYPE_RGB;
	h.cmap_bits = 0;
	SETLE16(h.cmap_len, 0);
	if(surface->format->Amask) {
	    alpha = 1;
	    h.pixel_bits = 32;
	} else
	    h.pixel_bits = 24;
	if(SDL_BYTEORDER == SDL_BIG_ENDIAN) {
	    int s = alpha ? 0 : 8;
	    amask = 0x000000ff >> s;
	    rmask = 0x0000ff00 >> s;
	    gmask = 0x00ff0000 >> s;
	    bmask = 0xff000000 >> s;
	} else {
	    amask = alpha ? 0xff000000 : 0;
	    rmask = 0x00ff0000;
	    gmask = 0x0000ff00;
	    bmask = 0x000000ff;
	}
    }
    bpp = h.pixel_bits >> 3;
    if(rle)
	    h.type += TGA_TYPE_RLE;

    SETLE16(h.yorigin, 0);
    SETLE16(h.xorigin, 0);
    SETLE16(h.width, surface->w);
    SETLE16(h.height, surface->h);
    h.flags = TGA_ORIGIN_UPPER | (alpha ? 8 : 0);

    if(!SDL_RWwrite(out, &h, sizeof(h), 1))
	return -1;

    if(h.has_cmap) {
	int i;
	SDL_Palette *pal = surface->format->palette;
	Uint8 entry[4];
	for(i = 0; i < pal->ncolors; i++) {
	    entry[0] = pal->colors[i].b;
	    entry[1] = pal->colors[i].g;
	    entry[2] = pal->colors[i].r;
	    entry[3] = (i == ckey) ? 0 : 0xff;
	    if(!SDL_RWwrite(out, entry, h.cmap_bits >> 3, 1))
		return -1;
	}
    }

    linebuf = SDL_CreateRGBSurface(SDL_SWSURFACE, surface->w, 1, h.pixel_bits,
				   rmask, gmask, bmask, amask);
    if(!linebuf)
	return -1;
    if(h.has_cmap)
	SDL_SetColors(linebuf, surface->format->palette->colors, 0,
		      surface->format->palette->ncolors);
    if(rle) {
	rlebuf = malloc(bpp * surface->w + 1 + surface->w / TGA_RLE_MAX);
	if(!rlebuf) {
	    SDL_SetError("out of memory");
	    goto error;
	}
    }

    /* Temporarily remove colourkey and alpha from surface so copies are
       opaque */
    surf_flags = surface->flags & (SDL_SRCALPHA | SDL_SRCCOLORKEY);
    surf_alpha = surface->format->alpha;
    if(surf_flags & SDL_SRCALPHA)
	SDL_SetAlpha(surface, 0, 255);
    if(surf_flags & SDL_SRCCOLORKEY)
	SDL_SetColorKey(surface, 0, surface->format->colorkey);

    r.x = 0;
    r.w = surface->w;
    r.h = 1;
    for(r.y = 0; r.y < surface->h; r.y++) {
	int n;
	void *buf;
	if(SDL_BlitSurface(surface, &r, linebuf, NULL) < 0)
	    break;
	if(rle) {
	    buf = rlebuf;
	    n = rle_line(linebuf->pixels, rlebuf, surface->w, bpp);
	} else {
	    buf = linebuf->pixels;
	    n = surface->w * bpp;
	}
	if(!SDL_RWwrite(out, buf, n, 1))
	    break;
    }

    /* restore flags */
    if(surf_flags & SDL_SRCALPHA)
	SDL_SetAlpha(surface, SDL_SRCALPHA, (Uint8)surf_alpha);
    if(surf_flags & SDL_SRCCOLORKEY)
	SDL_SetColorKey(surface, SDL_SRCCOLORKEY, surface->format->colorkey);

error:
    free(rlebuf);
    SDL_FreeSurface(linebuf);
    return 0;
}

static int SaveTGA(SDL_Surface *surface, char *file, int rle)
{
    SDL_RWops *out = SDL_RWFromFile(file, "wb");
    int ret;
    if(!out)
	return -1;
    ret = SaveTGA_RW(surface, out, rle);
    SDL_RWclose(out);
    return ret;
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

