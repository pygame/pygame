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

#include"pygame.h"
#include<Numeric/arrayobject.h>
#include<SDL_byteorder.h>



    /*DOC*/ static char doc_pixels3d[] =
    /*DOC*/    "pygame.surfarray.pixels3d(Surface) -> Array\n"
    /*DOC*/    "get a 3d reference array to a surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "This returns a new noncontigous 3d array that\n"
    /*DOC*/    "directly effects a Surface's contents. Think of it\n"
    /*DOC*/    "as a 2d image array with an RGB array for each\n"
    /*DOC*/    "pixel value.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will only work for 24 and 32 bit surfaces,\n"
    /*DOC*/    "where the RGB components can be accessed without\n"
    /*DOC*/    "requiring any masking.\n"
    /*DOC*/    "\n"
    /*DOC*/    "You'll need the surface to be locked if that is\n"
    /*DOC*/    "required. Also be aware that between unlocking and\n"
    /*DOC*/    "relocking a surface, the pixel data can be moved,\n"
    /*DOC*/    "so don't hang onto this array after you have\n"
    /*DOC*/    "unlocked the surface.\n"
    /*DOC*/ ;

static PyObject* pixels3d(PyObject* self, PyObject* arg)
{
	int dim[3];
	PyObject* array;
	SDL_Surface* surf;
	char* startpixel;
	int pixelstep;

	if(!PyArg_ParseTuple(arg, "O!", &PySurface_Type, &array))
		return NULL;
	surf = PySurface_AsSurface(array);

	if(surf->format->BytesPerPixel <= 2 || surf->format->BytesPerPixel > 4)
		return RAISE(PyExc_ValueError, "unsupport bit depth for 3D reference array");

	/*must discover information about how data is packed*/
	if(SDL_BYTEORDER == SDL_LIL_ENDIAN) /*intel-style*/
	{
		if(surf->format->Rmask == 0xff<<16 && 
					surf->format->Gmask == 0xff<<8 &&
					surf->format->Bmask == 0xff)
		{
			pixelstep = -1;
			startpixel = ((char*)surf->pixels)+2;
		}
		else if(surf->format->Bmask == 0xff<<16 && 
					surf->format->Gmask == 0xff<<8 &&
					surf->format->Rmask == 0xff)
		{
			pixelstep = 1;
			startpixel = ((char*)surf->pixels);
		}
		else
			return RAISE(PyExc_ValueError, "unsupport colormasks for 3D reference array");
	}
	else /*mips-style*/
	{
		if(surf->format->Rmask == 0xff<<16 && 
					surf->format->Gmask == 0xff<<8 &&
					surf->format->Bmask == 0xff)
		{
			pixelstep = 1;
			startpixel = ((char*)surf->pixels);
		}
		else if(surf->format->Bmask == 0xff<<16 && 
					surf->format->Gmask == 0xff<<8 &&
					surf->format->Rmask == 0xff)
		{
			pixelstep = -1;
			startpixel = ((char*)surf->pixels)+2;
		}
		else
			return RAISE(PyExc_ValueError, "unsupport colormasks 3D reference array");
	}

	dim[0] = surf->w;
	dim[1] = surf->h;
	dim[2] = 3; /*could be 4 if alpha in the house*/
	array = PyArray_FromDimsAndData(3, dim, PyArray_UBYTE, startpixel);
	if(array)
	{
		((PyArrayObject*)array)->flags = OWN_DIMENSIONS|OWN_STRIDES|SAVESPACE;
		((PyArrayObject*)array)->strides[2] = pixelstep;
		((PyArrayObject*)array)->strides[1] = surf->pitch;
		((PyArrayObject*)array)->strides[0] = surf->format->BytesPerPixel;
		((PyArrayObject*)array)->base = array;
		Py_INCREF(array);
	}
	return array;
}



    /*DOC*/ static char doc_pixels2d[] =
    /*DOC*/    "pygame.surfarray.pixels2d(Surface) -> Array\n"
    /*DOC*/    "get a 2d reference array to a surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "This returns a new noncontigous 2d array that\n"
    /*DOC*/    "directly effects a Surface's contents. Think of it\n"
    /*DOC*/    "as a 2d image array with a mapped pixel value at\n"
    /*DOC*/    "each index.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will not work on 24bit surfaces, since there\n"
    /*DOC*/    "is no native 24bit data type to access the pixel\n"
    /*DOC*/    "values.\n"
    /*DOC*/    "\n"
    /*DOC*/    "You'll need the surface to be locked if that is\n"
    /*DOC*/    "required. Also be aware that between unlocking and\n"
    /*DOC*/    "relocking a surface, the pixel data can be moved,\n"
    /*DOC*/    "so don't hang onto this array after you have\n"
    /*DOC*/    "unlocked the surface.\n"
    /*DOC*/ ;

static PyObject* pixels2d(PyObject* self, PyObject* arg)
{
	int types[] = {PyArray_UBYTE, PyArray_SHORT, 0, PyArray_INT};
	int dim[3];
	int type;
	PyObject* array;
	SDL_Surface* surf;

	if(!PyArg_ParseTuple(arg, "O!", &PySurface_Type, &array))
		return NULL;
	surf = PySurface_AsSurface(array);


	if(surf->format->BytesPerPixel == 3 || surf->format->BytesPerPixel < 1 || surf->format->BytesPerPixel > 4)
	{
		PyErr_SetString(PyExc_ValueError, "unsupport bit depth for 2D reference array");
		return NULL;
	}

	dim[0] = surf->w;
	dim[1] = surf->h;
	type = types[surf->format->BytesPerPixel-1];
	array = PyArray_FromDimsAndData(2, dim, type, (char*)surf->pixels);
	if(array)
	{
		((PyArrayObject*)array)->strides[1] = surf->pitch;
		((PyArrayObject*)array)->strides[0] = surf->format->BytesPerPixel;
		((PyArrayObject*)array)->flags = OWN_DIMENSIONS|OWN_STRIDES;
	}
	return array;
}



    /*DOC*/ static char doc_array2d[] =
    /*DOC*/    "pygame.surfarray.array2d(Surface) -> Array\n"
    /*DOC*/    "get a 2d array copied from a surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "This returns a new contigous 2d array. Think of it\n"
    /*DOC*/    "as a 2d image array with a mapped pixel value at\n"
    /*DOC*/    "each index.\n"
    /*DOC*/    "\n"
    /*DOC*/    "You'll need the surface to be locked if that is\n"
    /*DOC*/    "required. Once the array is created you can unlock\n"
    /*DOC*/    "the surface.\n"
    /*DOC*/ ;

PyObject* array2d(PyObject* self, PyObject* arg)
{
	int dim[2], loopy;
	int* data;
	PyObject* array;
	SDL_Surface* surf;

	if(!PyArg_ParseTuple(arg, "O!", &PySurface_Type, &array))
		return NULL;
	surf = PySurface_AsSurface(array);

	dim[0] = surf->w;
	dim[1] = surf->h;

	if(surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
		return RAISE(PyExc_ValueError, "unsupport bit depth for surface array");

	array = PyArray_FromDims(2, dim, PyArray_INT);
	if(!array) return NULL;

	data = (int*)((PyArrayObject*)array)->data;
	
	switch(surf->format->BytesPerPixel)
	{
	case 1:
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			char* pix = (char*)(((char*)surf->pixels)+loopy*surf->pitch);
			char* end = (char*)(((char*)pix)+surf->w*surf->format->BytesPerPixel);
			while(pix <= end)
				*data++ = *pix++;
		}break;
	case 2:
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			short* pix = (short*)(((char*)surf->pixels)+loopy*surf->pitch);
			short* end = (short*)(((char*)pix)+surf->w*surf->format->BytesPerPixel);
			while(pix <= end)
				*data++ = *pix++;
		}break;
	case 3:
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			char* pix = (char*)(((char*)surf->pixels)+loopy*surf->pitch);
			char* end = (char*)(((char*)pix)+surf->w*surf->format->BytesPerPixel);
			while(pix <= end)
			{
				*data++ = (*(int*)pix) >> 8;
				pix += 3;
			}
		}break;
	default: /*case 4*/
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			int* pix = (int*)(((char*)surf->pixels)+loopy*surf->pitch);
			int* end = (int*)(((char*)pix)+surf->w*surf->format->BytesPerPixel);
			while(pix <= end)
				*data++ = *pix++;
		}break;
	}

	return array;
}



    /*DOC*/ static char doc_array3d[] =
    /*DOC*/    "pygame.surfarray.array3d(Surface) -> Array\n"
    /*DOC*/    "get a 3d array copied from a surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "This returns a new contigous 3d array. Think of it\n"
    /*DOC*/    "as a 2d image array with an RGB array for each\n"
    /*DOC*/    "pixel value.\n"
    /*DOC*/    "\n"
    /*DOC*/    "You'll need the surface to be locked if that is\n"
    /*DOC*/    "required. Once the array is created you can unlock\n"
    /*DOC*/    "the surface.\n"
    /*DOC*/ ;

PyObject* array3d(PyObject* self, PyObject* arg)
{
	int dim[3], loopy;
	Uint8* data;
	PyObject* array;
	SDL_Surface* surf;
	SDL_PixelFormat* format;
	int Rmask, Gmask, Bmask, Rshift, Gshift, Bshift;

	if(!PyArg_ParseTuple(arg, "O!", &PySurface_Type, &array))
		return NULL;
	surf = PySurface_AsSurface(array);

	format = surf->format;
	dim[0] = surf->w;
	dim[1] = surf->h;
	dim[2] = 3;

	if(surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
		return RAISE(PyExc_ValueError, "unsupport bit depth for surface array");

	array = PyArray_FromDims(3, dim, PyArray_UBYTE);
	if(!array) return NULL;

	data = (Uint8*)((PyArrayObject*)array)->data;
	Rmask = format->Rmask; Gmask = format->Gmask; Bmask = format->Bmask;
	Rshift = format->Rshift; Gshift = format->Gshift; Bshift = format->Bshift;
	
	switch(surf->format->BytesPerPixel)
	{
	case 1:
		return RAISE(PyExc_ValueError, "colormaps unsupported");
	case 2:
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			short* pix = (short*)(((char*)surf->pixels)+loopy*surf->pitch);
			short* end = (short*)(((char*)pix)+surf->w*surf->format->BytesPerPixel);
			while(pix <= end)
			{
				short color = *pix++;
				*data++ = (color&Rmask)>>Rshift;
				*data++ = (color&Gmask)>>Gshift;
				*data++ = (color&Bmask)>>Bshift;
			}
		}break;
	case 3:
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			char* pix = (char*)(((char*)surf->pixels)+loopy*surf->pitch);
			char* end = (char*)(((char*)pix)+surf->w*surf->format->BytesPerPixel);
			while(pix <= end)
			{
				int color = (*(int*)pix++) >> 8;
				*data++ = (color&Rmask)>>Rshift;
				*data++ = (color&Gmask)>>Gshift;
				*data++ = (color&Bmask)>>Bshift;
			}
		}break;
	default: /*case 4*/
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			int* pix = (int*)(((char*)surf->pixels)+loopy*surf->pitch);
			int* end = (int*)(((char*)pix)+surf->w*surf->format->BytesPerPixel);
			while(pix <= end)
			{
				int color = *pix++;
				*data++ = (color&Rmask)>>Rshift;
				*data++ = (color&Gmask)>>Gshift;
				*data++ = (color&Bmask)>>Bshift;
			}
		}break;
	}

	return array;
}




    /*DOC*/ static char doc_map_array[] =
    /*DOC*/    "pygame.surfarray.map_array(surf, array3d) -> array2d\n"
    /*DOC*/    "map an array with RGB values into mapped colors\n"
    /*DOC*/    "\n"
    /*DOC*/    "Create a new array with the RGB pixel values of a\n"
    /*DOC*/    "3d array into mapped color values in a 2D array.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Just so you know, this can also map a 2D array\n"
    /*DOC*/    "with RGB values into a 1D array of mapped color\n"
    /*DOC*/    "values\n"
    /*DOC*/ ;

PyObject* map_array(PyObject* self, PyObject* arg)
{
	int* data;
	PyObject *surfobj, *arrayobj, *newarray;
	SDL_Surface* surf;
	SDL_PixelFormat* format;
	PyArrayObject* array;
	int loopx, loopy;
	int stridex, stridey, stridez, stridez2, sizex, sizey;
	int dims[2];

	if(!PyArg_ParseTuple(arg, "O!O!", &PySurface_Type, &surfobj, &PyArray_Type, &arrayobj))
		return NULL;
	surf = PySurface_AsSurface(surfobj);
	format = surf->format;
	array = (PyArrayObject*)arrayobj;

	if(!array->nd || array->dimensions[array->nd-1] != 3)
		return RAISE(PyExc_ValueError, "array must be a 3d array of 3-value color data\n");

	if(surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
		return RAISE(PyExc_ValueError, "unsupport bit depth for surface array");

	switch(array->nd)
	{
	case 3: /*image of colors*/
		dims[0] = array->dimensions[0];
		dims[1] = array->dimensions[1];
		newarray = PyArray_FromDims(2, dims, PyArray_INT);
		if(!newarray) return NULL;
		data = (int*)((PyArrayObject*)newarray)->data;
		stridex = array->strides[0];
		stridey = array->strides[1];
		stridez = array->strides[2];
		sizex = array->dimensions[0];
		sizey = array->dimensions[1];
		break;
	case 2: /*list of colors*/
		dims[0] = array->dimensions[0];
		newarray = PyArray_FromDims(1, dims, PyArray_INT);
		if(!newarray) return NULL;
		data = (int*)((PyArrayObject*)newarray)->data;
		stridex = 0;
		stridey = array->strides[0];
		stridez = array->strides[1];
		sizex = 1;
		sizey = array->dimensions[0];
#if 0 /*kinda like a scalar here, use normal map_rgb*/
	case 1: /*single color*/
		dims[0] = 1;
		newarray = PyArray_FromDims(1, dims, PyArray_INT);
		if(!newarray) return NULL;
		data = (int*)((PyArrayObject*)newarray)->data;
		stridex = 0;
		stridey = 0;
		stridez = array->strides[0];
		sizex = 1;
		sizey = 1;
#endif
	default:
		return RAISE(PyExc_ValueError, "unsupported array shape");
	}
	stridez2 = stridez*2;


	switch(array->descr->elsize)
	{
	case sizeof(char):
		for(loopx = 0; loopx < sizex; ++loopx)
		{
			char* col = array->data + stridex * loopx;
			for(loopy = 0; loopy < sizey; ++loopy)
			{
				char* pix = col + stridey * loopy;
				*data++ =	(*((unsigned char*)(pix)) >>
								format->Rloss << format->Rshift) |
							(*((unsigned char*)(pix+stridez)) >>
								format->Gloss << format->Gshift) |
							(*((unsigned char*)(pix+stridez2)) >>
								format->Bloss << format->Bshift);
			}
		}break;
	case sizeof(short):
		for(loopx = 0; loopx < sizex; ++loopx)
		{
			char* col = array->data + stridex * loopx;
			for(loopy = 0; loopy < sizey; ++loopy)
			{
				char* pix = col + stridey * loopy;
				*data++ =	(*((unsigned short*)(pix)) >>
								format->Rloss << format->Rshift) |
							(*((unsigned short*)(pix+stridez)) >>
								format->Gloss << format->Gshift) |
							(*((unsigned short*)(pix+stridez2)) >>
								format->Bloss << format->Bshift);
			}
		}break;
	case sizeof(int):
		for(loopx = 0; loopx < sizex; ++loopx)
		{
			char* col = array->data + stridex * loopx;
			for(loopy = 0; loopy < sizey; ++loopy)
			{
				char* pix = col + stridey * loopy;
				*data++ =	(*((int*)(pix)) >>
								format->Rloss << format->Rshift) |
							(*((int*)(pix+stridez)) >>
								format->Gloss << format->Gshift) |
							(*((int*)(pix+stridez2)) >>
								format->Bloss << format->Bshift);
			}
		}break;
	default: 
		Py_DECREF(newarray);
		return RAISE(PyExc_ValueError, "unsupported bytesperpixel for array\n");
	}

	return newarray;
}



    /*DOC*/ static char doc_blit_array[] =
    /*DOC*/    "pygame.surfarray.blit_array(surf, array) -> None\n"
    /*DOC*/    "quickly transfer an array to a Surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Transfer an array of any type (3d or 2d) onto a\n"
    /*DOC*/    "Surface. The array must be the same dimensions as\n"
    /*DOC*/    "the destination Surface. While you can assign the\n"
    /*DOC*/    "values of an array to the pixel referenced arrays,\n"
    /*DOC*/    "using this blit method will usually be quicker\n"
    /*DOC*/    "because of it's smarter handling of noncontiguous\n"
    /*DOC*/    "arrays. Plus it allows you to blit from any image\n"
    /*DOC*/    "array type to any surface format in one step, no\n"
    /*DOC*/    "conversions.\n"
    /*DOC*/ ;

PyObject* blit_array(PyObject* self, PyObject* arg)
{
	PyObject *surfobj, *arrayobj;
	SDL_Surface* surf;
	SDL_PixelFormat* format;
	PyArrayObject* array;
	int loopx, loopy;
	int stridex, stridey, stridez, stridez2, sizex, sizey;
	int Rloss, Gloss, Bloss, Rshift, Gshift, Bshift;

	if(!PyArg_ParseTuple(arg, "O!O!", &PySurface_Type, &surfobj, &PyArray_Type, &arrayobj))
		return NULL;
	surf = PySurface_AsSurface(surfobj);
	format = surf->format;
	array = (PyArrayObject*)arrayobj;

	if(!(array->nd == 2 || (array->nd == 3 && array->dimensions[2] == 3)))
		return RAISE(PyExc_ValueError, "must be a valid 2d or 3d array\n");

	if(surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
		return RAISE(PyExc_ValueError, "unsupport bit depth for surface");

	stridex = array->strides[0];
	stridey = array->strides[1];
	if(array->nd == 3)
	{
		stridez = array->strides[2];
		stridez2 = stridez*2;
	}
	sizex = array->dimensions[0];
	sizey = array->dimensions[1];
	Rloss = format->Rloss; Gloss = format->Gloss; Bloss = format->Bloss;
	Rshift = format->Rshift; Gshift = format->Gshift; Bshift = format->Bshift;

	if(sizex != surf->w || sizey != surf->h)
		return RAISE(PyExc_ValueError, "array must match surface dimensions");

	switch(surf->format->BytesPerPixel)
	{
	case 1:
	case 2:
	case 3:
		return RAISE(PyExc_RuntimeError, "unsupported bit depth for image");
	case 4:
		{
			if(array->nd == 2)
			{
				if(array->descr->elsize == sizeof(char))
				{
					for(loopy = 0; loopy < sizey; ++loopy)
					{
						int* imgrow = (int*)(((char*)surf->pixels)+loopy*surf->pitch);
						char* datarow = array->data + stridey * loopy;
						for(loopx = 0; loopx < sizex; ++loopx)
						{
							int* pix = imgrow + loopx;
							char* data = datarow + stridex * loopx;
							*(imgrow + loopx) = (int)*(unsigned char*)(datarow + stridex * loopx);
						}
					}
				}
				else if(array->descr->elsize == sizeof(int))
				{
					for(loopy = 0; loopy < sizey; ++loopy)
					{
						int* imgrow = (int*)(((char*)surf->pixels)+loopy*surf->pitch);
						char* datarow = array->data + stridey * loopy;
						for(loopx = 0; loopx < sizex; ++loopx)
						{
							int* pix = imgrow + loopx;
							char* data = datarow + stridex * loopx;
							*(imgrow + loopx) = (int)*(int*)(datarow + stridex * loopx);
						}
					}
				}
			}
			else
			{
				switch(array->descr->elsize)
				{
				case sizeof(char):
					for(loopy = 0; loopy < sizey; ++loopy)
					{
						int* data = (int*)(((char*)surf->pixels) + surf->pitch * loopy);
						char* pix = array->data + stridey * loopy;
						for(loopx = 0; loopx < sizex; ++loopx)
						{
							*data++ = (*((unsigned char*)(pix)) >> Rloss << Rshift) |
									((*((unsigned char*)(pix+stridez))) >> Gloss << Gshift) |
									((*((unsigned char*)(pix+stridez2))) >> Bloss << Bshift);
							pix += stridex;
						}
					}break;
				case sizeof(int):
					for(loopy = 0; loopy < sizey; ++loopy)
					{
						int* data = (int*)(((char*)surf->pixels) + surf->pitch * loopy);
						char* pix = array->data + stridey * loopy;
						for(loopx = 0; loopx < sizex; ++loopx)
						{
							*data++ = (*((unsigned int*)(pix)) >> Rloss << Rshift) |
									((*((unsigned int*)(pix+stridez))) >> Gloss << Gshift) |
									((*((unsigned int*)(pix+stridez2))) >> Bloss << Bshift);
							pix += stridex;
						}
					}break;
				default: 
					return RAISE(PyExc_ValueError, "unsupported datatype for array\n");
				}
			}
		}break;
	}

	RETURN_NONE;
}


static PyMethodDef surfarray_builtins[] =
{
	{ "pixels2d", pixels2d, 1, doc_pixels2d },
	{ "pixels3d", pixels3d, 1, doc_pixels3d },
	{ "array2d", array2d, 1, doc_array2d },
	{ "array3d", array3d, 1, doc_array3d },
	{ "map_array", map_array, 1, doc_map_array },
/*	{ "unmap_array", unmap_array, 1, doc_unmap_array },*/
	{ "blit_array", blit_array, 1, doc_blit_array },

	{ NULL, NULL }
};






    /*DOC*/ static char doc_pygame_surfarray_MODULE[] =
    /*DOC*/    "Contains routines for mixing numeric arrays with\n"
    /*DOC*/    "surfaces\n"
    /*DOC*/ ;

void initsurfarray()
{
	PyObject *module, *dict;

    /* create the module */
	module = Py_InitModule3("surfarray", surfarray_builtins, doc_pygame_surfarray_MODULE);
	dict = PyModule_GetDict(module);

	/*imported needed apis*/
	import_pygame_base();
	import_pygame_surface();
	import_array();
}



