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
    /*DOC*/    "where the RGB components can be accessed as 8-bit\n"
    /*DOC*/    "components.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function will lock the given surface, and it\n"
    /*DOC*/    "will remained locked for as long as the pixel array\n"
    /*DOC*/    "exists\n"
    /*DOC*/ ;

static PyObject* pixels3d(PyObject* self, PyObject* arg)
{
	int dim[3];
	PyObject* array, *surfobj;
	SDL_Surface* surf;
	char* startpixel;
	int pixelstep;
	const int lilendian = (SDL_BYTEORDER == SDL_LIL_ENDIAN);
	PyObject* lifelock;

	if(!PyArg_ParseTuple(arg, "O!", &PySurface_Type, &surfobj))
		return NULL;
	surf = PySurface_AsSurface(surfobj);

	if(surf->format->BytesPerPixel <= 2 || surf->format->BytesPerPixel > 4)
		return RAISE(PyExc_ValueError, "unsupport bit depth for 3D reference array");

	lifelock = PySurface_LockLifetime(surfobj);
	if(!lifelock) return NULL;

	/*must discover information about how data is packed*/
	if(surf->format->Rmask == 0xff<<16 &&
				surf->format->Gmask == 0xff<<8 &&
				surf->format->Bmask == 0xff)
	{
		pixelstep = (lilendian ? -1 : 1);
		startpixel = ((char*)surf->pixels) + (lilendian ? 2 : 0);
	}
	else if(surf->format->Bmask == 0xff<<16 &&
				surf->format->Gmask == 0xff<<8 &&
				surf->format->Rmask == 0xff)
	{
		pixelstep = (lilendian ? 1 : -1);
		startpixel = ((char*)surf->pixels) + (lilendian ? 0 : 2);
	}
	else
		return RAISE(PyExc_ValueError, "unsupport colormasks for 3D reference array");
	if(!lilendian && surf->format->BytesPerPixel == 4)
	    ++startpixel;

	/*create the referenced array*/
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
		((PyArrayObject*)array)->base = lifelock;
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
    /*DOC*/    "This function will lock the given surface, and it\n"
    /*DOC*/    "will remained locked for as long as the pixel array\n"
    /*DOC*/    "exists\n"
    /*DOC*/ ;

static PyObject* pixels2d(PyObject* self, PyObject* arg)
{
	int types[] = {PyArray_UBYTE, PyArray_SHORT, 0, PyArray_INT};
	int dim[3];
	int type;
	PyObject *array, *surfobj;
	SDL_Surface* surf;
	PyObject* lifelock;

	if(!PyArg_ParseTuple(arg, "O!", &PySurface_Type, &surfobj))
		return NULL;
	surf = PySurface_AsSurface(surfobj);


	if(surf->format->BytesPerPixel == 3 || surf->format->BytesPerPixel < 1 || surf->format->BytesPerPixel > 4)
		return RAISE(PyExc_ValueError, "unsupport bit depth for 2D reference array");

	lifelock = PySurface_LockLifetime(surfobj);
	if(!lifelock) return NULL;

	dim[0] = surf->w;
	dim[1] = surf->h;
	type = types[surf->format->BytesPerPixel-1];
	array = PyArray_FromDimsAndData(2, dim, type, (char*)surf->pixels);
	if(array)
	{
		((PyArrayObject*)array)->strides[1] = surf->pitch;
		((PyArrayObject*)array)->strides[0] = surf->format->BytesPerPixel;
		((PyArrayObject*)array)->flags = OWN_DIMENSIONS|OWN_STRIDES;
		((PyArrayObject*)array)->base = lifelock;
	}
	return array;
}


    /*DOC*/ static char doc_pixels_alpha[] =
    /*DOC*/    "pygame.surfarray.pixels_alpha(Surface) -> Array\n"
    /*DOC*/    "get a reference array to a surface alpha data\n"
    /*DOC*/    "\n"
    /*DOC*/    "This returns a new noncontigous array that directly\n"
    /*DOC*/    "effects a Surface's alpha contents.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will only work for 32bit surfaces with a pixel\n"
    /*DOC*/    "alpha channel enabled.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function will lock the given surface, and it\n"
    /*DOC*/    "will remained locked for as long as the pixel array\n"
    /*DOC*/    "exists\n"
    /*DOC*/ ;

static PyObject* pixels_alpha(PyObject* self, PyObject* arg)
{
	int dim[3];
	PyObject *array, *surfobj;
	PyObject* lifelock;
	SDL_Surface* surf;
	char* startpixel;
	const int lilendian = (SDL_BYTEORDER == SDL_LIL_ENDIAN);

	if(!PyArg_ParseTuple(arg, "O!", &PySurface_Type, &surfobj))
		return NULL;
	surf = PySurface_AsSurface(surfobj);

	if(surf->format->BytesPerPixel != 4)
		return RAISE(PyExc_ValueError, "unsupport bit depth for alpha array");

	lifelock = PySurface_LockLifetime(surfobj);
	if(!lifelock) return NULL;

	/*must discover information about how data is packed*/
	if(surf->format->Amask == 0xff<<24)
		startpixel = ((char*)surf->pixels) + (lilendian ? 3 : 0);
	else if(surf->format->Amask == 0xff)
		startpixel = ((char*)surf->pixels) + (lilendian ? 0 : 3);
	else
		return RAISE(PyExc_ValueError, "unsupport colormasks for alpha reference array");

	dim[0] = surf->w;
	dim[1] = surf->h;
	array = PyArray_FromDimsAndData(2, dim, PyArray_UBYTE, startpixel);
	if(array)
	{
		((PyArrayObject*)array)->strides[1] = surf->pitch;
		((PyArrayObject*)array)->strides[0] = surf->format->BytesPerPixel;
		((PyArrayObject*)array)->flags = OWN_DIMENSIONS|OWN_STRIDES;
		((PyArrayObject*)array)->base = lifelock;
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
    /*DOC*/    "This function will temporarily lock the surface.\n"
    /*DOC*/ ;

PyObject* array2d(PyObject* self, PyObject* arg)
{
	int dim[2], loopy;
	Uint8* data;
	PyObject *surfobj, *array;
	SDL_Surface* surf;
	int stridex, stridey;

	if(!PyArg_ParseTuple(arg, "O!", &PySurface_Type, &surfobj))
		return NULL;
	surf = PySurface_AsSurface(surfobj);

	dim[0] = surf->w;
	dim[1] = surf->h;

	if(surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
		return RAISE(PyExc_ValueError, "unsupport bit depth for surface array");
	array = PyArray_FromDims(2, dim, PyArray_INT);
	if(!array) return NULL;

	stridex = ((PyArrayObject*)array)->strides[0];
	stridey = ((PyArrayObject*)array)->strides[1];

	if(!PySurface_Lock(surfobj)) return NULL;

	switch(surf->format->BytesPerPixel)
	{
	case 1:
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			Uint8* pix = (Uint8*)(((char*)surf->pixels)+loopy*surf->pitch);
			Uint8* end = (Uint8*)(((char*)pix)+surf->w);
			data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
			while(pix < end)
			{
				*(Uint32*)data = *pix++;
				data += stridex;
			}
		}break;
	case 2:
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			Uint16* pix = (Uint16*)(((char*)surf->pixels)+loopy*surf->pitch);
			Uint16* end = (Uint16*)(((char*)pix)+surf->w*2);
			data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
			while(pix < end)
			{
				*(Uint32*)data = *pix++;
				data += stridex;
			}
		}break;
	case 3:
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			Uint8* pix = (Uint8*)(((char*)surf->pixels)+loopy*surf->pitch);
			Uint8* end = pix+surf->w*3;
			data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
			while(pix < end)
			{
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
				*(Uint32*)data = pix[0] + (pix[1]<<8) + (pix[2]<<16);
#else
				*(Uint32*)data = pix[2] + (pix[1]<<8) + (pix[0]<<16);
#endif
				pix += 3;
				data += stridex;
			}
		}break;
	default: /*case 4*/
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			Uint32* pix = (Uint32*)(((char*)surf->pixels)+loopy*surf->pitch);
			Uint32* end = (Uint32*)(((char*)pix)+surf->w*4);
			data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
			while(pix < end)
			{
				*(Uint32*)data = *pix++;
				data += stridex;
			}
		}break;
	}

	if(!PySurface_Unlock(surfobj)) return NULL;
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
    /*DOC*/    "This function will temporarily lock the surface.\n"
    /*DOC*/ ;

PyObject* array3d(PyObject* self, PyObject* arg)
{
	int dim[3], loopy;
	Uint8* data;
	PyObject *array, *surfobj;
	SDL_Surface* surf;
	SDL_PixelFormat* format;
	int Rmask, Gmask, Bmask, Rshift, Gshift, Bshift, Rloss, Gloss, Bloss;
	int stridex, stridey;
	SDL_Color* palette;

	if(!PyArg_ParseTuple(arg, "O!", &PySurface_Type, &surfobj))
		return NULL;
	surf = PySurface_AsSurface(surfobj);

	format = surf->format;
	dim[0] = surf->w;
	dim[1] = surf->h;
	dim[2] = 3;
	Rmask = format->Rmask; Gmask = format->Gmask; Bmask = format->Bmask;
	Rshift = format->Rshift; Gshift = format->Gshift; Bshift = format->Bshift;
	Rloss = format->Rloss; Gloss = format->Gloss; Bloss = format->Bloss;

	if(surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
		return RAISE(PyExc_ValueError, "unsupport bit depth for surface array");

	array = PyArray_FromDims(3, dim, PyArray_UBYTE);
	if(!array) return NULL;

	stridex = ((PyArrayObject*)array)->strides[0];
	stridey = ((PyArrayObject*)array)->strides[1];

	if(!PySurface_Lock(surfobj)) return NULL;

	switch(surf->format->BytesPerPixel)
	{
	case 1:
		if(!format->palette)
		{
			if(!PySurface_Unlock(surfobj)) return NULL;
			return RAISE(PyExc_RuntimeError, "8bit surface has no palette");
		}
		palette = format->palette->colors;
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			Uint8* pix = (Uint8*)(((char*)surf->pixels)+loopy*surf->pitch);
			Uint8* end = (Uint8*)(((char*)pix)+surf->w*surf->format->BytesPerPixel);
			data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
			while(pix < end)
			{
				SDL_Color* c = palette + (*pix++);
				data[0] = c->r;
				data[1] = c->g;
				data[2] = c->b;
				data += stridex;
			}
		}break;
	case 2:
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			Uint16* pix = (Uint16*)(((char*)surf->pixels)+loopy*surf->pitch);
			Uint16* end = (Uint16*)(((char*)pix)+surf->w*surf->format->BytesPerPixel);
			data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
			while(pix < end)
			{
				Uint32 color = *pix++;
				data[0] = ((color&Rmask)>>Rshift)<<Rloss;
				data[1] = ((color&Gmask)>>Gshift)<<Gloss;
				data[2] = ((color&Bmask)>>Bshift)<<Bloss;
				data += stridex;
			}
		}break;
	case 3:
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			Uint8* pix = (Uint8*)(((char*)surf->pixels)+loopy*surf->pitch);
			Uint8* end = (Uint8*)(((char*)pix)+surf->w*surf->format->BytesPerPixel);
			data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
			while(pix < end)
			{
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
				Uint32 color = (pix[0]) + (pix[1]<<8) + (pix[2]<<16); pix += 3;
#else
				Uint32 color = (pix[2]) + (pix[1]<<8) + (pix[0]<<16); pix += 3;
#endif
				data[0] = ((color&Rmask)>>Rshift)/*<<Rloss*/; /*assume no loss on 24bit*/
				data[1] = ((color&Gmask)>>Gshift)/*<<Gloss*/;
				data[2] = ((color&Bmask)>>Bshift)/*<<Bloss*/;
				data += stridex;
			}
		}break;
	default: /*case 4*/
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			Uint32* pix = (Uint32*)(((char*)surf->pixels)+loopy*surf->pitch);
			Uint32* end = (Uint32*)(((char*)pix)+surf->w*surf->format->BytesPerPixel);
			data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
			while(pix < end)
			{
				Uint32 color = *pix++;
				data[0] = ((color&Rmask)>>Rshift)/*<<Rloss*/; /*assume no loss on 32bit*/
				data[1] = ((color&Gmask)>>Gshift)/*<<Gloss*/;
				data[2] = ((color&Bmask)>>Bshift)/*<<Bloss*/;
				data += stridex;
			}
		}break;
	}

	if(!PySurface_Unlock(surfobj)) return NULL;
	return array;
}




    /*DOC*/ static char doc_array_alpha[] =
    /*DOC*/    "pygame.surfarray.array_alpha(Surface) -> Array\n"
    /*DOC*/    "get an array with a surface pixel alpha values\n"
    /*DOC*/    "\n"
    /*DOC*/    "This returns a new contigous 2d array with the\n"
    /*DOC*/    "alpha values of an image as unsigned bytes. If the\n"
    /*DOC*/    "surface has no alpha, an array of all opaque values\n"
    /*DOC*/    "is returned.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function will temporarily lock the surface.\n"
    /*DOC*/ ;

PyObject* array_alpha(PyObject* self, PyObject* arg)
{
	int dim[2], loopy;
	Uint8* data;
	Uint32 color;
	PyObject *array, *surfobj;
	SDL_Surface* surf;
	int stridex, stridey;
	int Ashift, Amask, Aloss;

	if(!PyArg_ParseTuple(arg, "O!", &PySurface_Type, &surfobj))
		return NULL;
	surf = PySurface_AsSurface(surfobj);

	dim[0] = surf->w;
	dim[1] = surf->h;

	if(surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
		return RAISE(PyExc_ValueError, "unsupport bit depth for alpha array");

	array = PyArray_FromDims(2, dim, PyArray_UBYTE);
	if(!array) return NULL;

	Amask = surf->format->Amask;
	Ashift = surf->format->Ashift;
	Aloss = surf->format->Aloss;

	if(!Amask || surf->format->BytesPerPixel==1) /*no pixel alpha*/
	{
		memset(((PyArrayObject*)array)->data, 255, surf->w * surf->h);
		return array;
	}

	stridex = ((PyArrayObject*)array)->strides[0];
	stridey = ((PyArrayObject*)array)->strides[1];

	if(!PySurface_Lock(surfobj)) return NULL;

	switch(surf->format->BytesPerPixel)
	{
	case 2:
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			Uint16* pix = (Uint16*)(((char*)surf->pixels)+loopy*surf->pitch);
			Uint16* end = (Uint16*)(((char*)pix)+surf->w*2);
			data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
			while(pix < end)
			{
				color = *pix++;
				*data = (color & Amask) >> Ashift << Aloss;
				data += stridex;
			}
		}break;
	case 3:
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			Uint8* pix = (Uint8*)(((char*)surf->pixels)+loopy*surf->pitch);
			Uint8* end = pix+surf->w*3;
			data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
			while(pix < end)
			{
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
				color = pix[0] + (pix[1]<<8) + (pix[2]<<16);
#else
				color = pix[2] + (pix[1]<<8) + (pix[0]<<16);
#endif
				*data = (color & Amask) >> Ashift << Aloss;
				pix += 3;
				data += stridex;
			}
		}break;
	default: /*case 4*/
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			Uint32* pix = (Uint32*)(((char*)surf->pixels)+loopy*surf->pitch);
			Uint32* end = (Uint32*)(((char*)pix)+surf->w*4);
			data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
			while(pix < end)
			{
				color = *pix++;
				*data = (color & Amask) >> Ashift /*<< Aloss*/; /*assume no loss in 32bit*/
				data += stridex;
			}
		}break;
	}

	if(!PySurface_Unlock(surfobj)) return NULL;
	return array;
}



    /*DOC*/ static char doc_array_colorkey[] =
    /*DOC*/    "pygame.surfarray.array_colorkey(Surface) -> Array\n"
    /*DOC*/    "get an array with a surface colorkey values\n"
    /*DOC*/    "\n"
    /*DOC*/    "This returns a new contigous 2d array with the\n"
    /*DOC*/    "colorkey values of an image as unsigned bytes. If the\n"
    /*DOC*/    "surface has no colorkey, an array of all opaque values\n"
    /*DOC*/    "is returned. Otherwise the array is either 0's or 255's.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function will temporarily lock the surface.\n"
    /*DOC*/ ;

PyObject* array_colorkey(PyObject* self, PyObject* arg)
{
	int dim[2], loopy;
	Uint8* data;
	Uint32 color, colorkey;
	PyObject *array, *surfobj;
	SDL_Surface* surf;
	int stridex, stridey;

	if(!PyArg_ParseTuple(arg, "O!", &PySurface_Type, &surfobj))
		return NULL;
	surf = PySurface_AsSurface(surfobj);

	dim[0] = surf->w;
	dim[1] = surf->h;

	if(surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
		return RAISE(PyExc_ValueError, "unsupport bit depth for colorkey array");

	array = PyArray_FromDims(2, dim, PyArray_UBYTE);
	if(!array) return NULL;

	colorkey = surf->format->colorkey;
	if(!(surf->flags & SDL_SRCCOLORKEY)) /*no pixel alpha*/
	{
		memset(((PyArrayObject*)array)->data, 255, surf->w * surf->h);
		return array;
	}

	stridex = ((PyArrayObject*)array)->strides[0];
	stridey = ((PyArrayObject*)array)->strides[1];

	if(!PySurface_Lock(surfobj)) return NULL;

	switch(surf->format->BytesPerPixel)
	{
	case 1:
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			Uint8* pix = (Uint8*)(((char*)surf->pixels)+loopy*surf->pitch);
			Uint8* end = (Uint8*)(((char*)pix)+surf->w);
			data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
			while(pix < end)
			{
				color = *pix++;
				*data = (color != colorkey) * 255;
				data += stridex;
			}
		}break;
	case 2:
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			Uint16* pix = (Uint16*)(((char*)surf->pixels)+loopy*surf->pitch);
			Uint16* end = (Uint16*)(((char*)pix)+surf->w*2);
			data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
			while(pix < end)
			{
				color = *pix++;
				*data = (color != colorkey) * 255;
				data += stridex;
			}
		}break;
	case 3:
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			Uint8* pix = (Uint8*)(((char*)surf->pixels)+loopy*surf->pitch);
			Uint8* end = pix+surf->w*3;
			data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
			while(pix < end)
			{
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
				color = pix[0] + (pix[1]<<8) + (pix[2]<<16);
#else
				color = pix[2] + (pix[1]<<8) + (pix[0]<<16);
#endif
				*data = (color != colorkey) * 255;
				pix += 3;
				data += stridex;
			}
		}break;
	default: /*case 4*/
		for(loopy = 0; loopy < surf->h; ++loopy)
		{
			Uint32* pix = (Uint32*)(((char*)surf->pixels)+loopy*surf->pitch);
			Uint32* end = (Uint32*)(((char*)pix)+surf->w*4);
			data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
			while(pix < end)
			{
				color = *pix++;
				*data = (color != colorkey) * 255;
				data += stridex;
			}
		}break;
	}

	if(!PySurface_Lock(surfobj)) return NULL;
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
		break;
#if 1 /*kinda like a scalar here, use normal map_rgb*/
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
		break;
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


#if 0
/* not really fast enough to warrant this, using minimum(maximum()) is same */
    /*DOC*/ static char XXX_clamp_array[] =
    /*DOC*/    "pygame.surfarray.clamp_array(array3d, min=0, max=255) -> None\n"
    /*DOC*/    "will clamp all integer values in an array between 0 and 255\n"
    /*DOC*/    "\n"
    /*DOC*/    "Given an array of integer values, this will make sure\n"
    /*DOC*/    "no values are outside the range between 0 and 255.\n"
    /*DOC*/    "This will modify the array in-place.\n"
    /*DOC*/    "\n"
    /*DOC*/    "You can specify the minimum and maximum values for clamping,\n"
    /*DOC*/    "but they default to 0 and 255, which is most useful.\n"
    /*DOC*/ ;

PyObject* clamp_array(PyObject* self, PyObject* arg)
{
	PyObject *arrayobj;
	PyArrayObject* array;
	int loopx, loopy, loopz;
	int stridex, stridey, stridez, sizex, sizey, sizez;
	int minval = 0, maxval = 255;

	if(!PyArg_ParseTuple(arg, "O!|ii", &PyArray_Type, &arrayobj, &minval, &maxval))
		return NULL;
	array = (PyArrayObject*)arrayobj;

	switch(array->nd)
	{
	case 3:
		stridex = array->strides[0];
		stridey = array->strides[1];
		stridez = array->strides[2];
		sizex = array->dimensions[0];
		sizey = array->dimensions[1];
		sizez = array->dimensions[2];
		break;
	case 2:
		stridex = 0;
		stridey = array->strides[0];
		stridez = array->strides[1];
		sizex = 1;
		sizey = array->dimensions[0];
		sizez = array->dimensions[1];
		break;
	case 1:
		stridex = 0;
		stridey = 0;
		stridez = array->strides[0];
		sizex = 1;
		sizey = 1;
		sizez = array->dimensions[0];
		break;
	default:
		return RAISE(PyExc_ValueError, "unsupported dimensions for array");
	}


	switch(array->descr->elsize)
	{
	case sizeof(char):
		for(loopx = 0; loopx < sizex; ++loopx)
		{
			char* col = array->data + stridex * loopx;
			for(loopy = 0; loopy < sizey; ++loopy)
			{
				char* row = col + stridey * loopy;
				for(loopz = 0; loopz < sizez; ++loopz)
				{
					char* data = (char*)row;
					if(*data < minval) *data = minval;
					else if(*data > maxval) *data = maxval;
					row += sizez;
				}
			}
		}break;
	case sizeof(short):
		for(loopx = 0; loopx < sizex; ++loopx)
		{
			char* col = array->data + stridex * loopx;
			for(loopy = 0; loopy < sizey; ++loopy)
			{
				char* row = col + stridey * loopy;
				for(loopz = 0; loopz < sizez; ++loopz)
				{
					short* data = (short*)row;
					if(*data < minval) *data = minval;
					else if(*data > maxval) *data = maxval;
					row += sizez;
				}
			}
		}break;
	case sizeof(int):
		for(loopx = 0; loopx < sizex; ++loopx)
		{
			char* col = array->data + stridex * loopx;
			for(loopy = 0; loopy < sizey; ++loopy)
			{
				char* row = col + stridey * loopy;
				for(loopz = 0; loopz < sizez; ++loopz)
				{
					int* data = (int*)row;
					if(*data < minval) *data = minval;
					else if(*data > maxval) *data = maxval;
					row += sizez;
				}
			}
		}break;
	}

	RETURN_NONE
}
#endif




/*macros used to blit arrays*/


#define COPYMACRO_2D(DST, SRC) \
     for(loopy = 0; loopy < sizey; ++loopy) { \
         DST* imgrow = (DST*)(((char*)surf->pixels)+loopy*surf->pitch); \
         Uint8* datarow = (Uint8*)array->data + stridey * loopy; \
         for(loopx = 0; loopx < sizex; ++loopx) \
             *(imgrow + loopx) = (DST)*(SRC*)(datarow + stridex * loopx); \
     }


#define COPYMACRO_2D_24(SRC) \
     for(loopy = 0; loopy < sizey-1; ++loopy) { \
         Uint8* imgrow = ((Uint8*)surf->pixels)+loopy*surf->pitch; \
         Uint8* datarow = (Uint8*)array->data + stridey * loopy; \
         for(loopx = 0; loopx < sizex; ++loopx) \
             *(int*)(imgrow + loopx*3) = (int)*(SRC*)(datarow + stridex * loopx)<<8; \
     }{ \
     char* imgrow = ((char*)surf->pixels)+loopy*surf->pitch; \
     char* datarow = array->data + stridey * loopy; \
     for(loopx = 0; loopx < sizex-1; ++loopx) \
         *(int*)(imgrow + loopx*3) = ((int)*(SRC*)(datarow + stridex * loopx))<<8; \
     }


#define COPYMACRO_3D(DST, SRC) \
     for(loopy = 0; loopy < sizey; ++loopy) { \
         DST* data = (DST*)(((char*)surf->pixels) + surf->pitch * loopy); \
         char* pix = array->data + stridey * loopy; \
         for(loopx = 0; loopx < sizex; ++loopx) { \
             *data++ = (DST)(*(SRC*)(pix) >> Rloss << Rshift) | \
                     (*(SRC*)(pix+stridez) >> Gloss << Gshift) | \
                     (*(SRC*)(pix+stridez2) >> Bloss << Bshift); \
             pix += stridex; \
     }    }


#define COPYMACRO_3D_24(SRC) \
     for(loopy = 0; loopy < sizey; ++loopy) { \
         Uint8* data = ((Uint8*)surf->pixels) + surf->pitch * loopy; \
         Uint8* pix = (Uint8*)array->data + stridey * loopy; \
         for(loopx = 0; loopx < sizex; ++loopx) { \
             *data++ = (Uint8)*(SRC*)(pix+stridez2); \
             *data++ = (Uint8)*(SRC*)(pix+stridez); \
             *data++ = (Uint8)*(SRC*)(pix); \
             pix += stridex; \
     }    }



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
    /*DOC*/    "internal conversions.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function will temporarily lock the surface.\n"
    /*DOC*/ ;

PyObject* blit_array(PyObject* self, PyObject* arg)
{
	PyObject *surfobj, *arrayobj;
	SDL_Surface* surf;
	SDL_PixelFormat* format;
	PyArrayObject* array;
	int loopx, loopy;
	int stridex, stridey, stridez=0, stridez2=0, sizex, sizey;
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
	if(!PySurface_Lock(surfobj)) return NULL;

	switch(surf->format->BytesPerPixel)
	{
	case 1:
		if(array->nd == 2) {
			switch(array->descr->elsize) {
				case sizeof(Uint8):  COPYMACRO_2D(Uint8, Uint8)  break;
				case sizeof(Uint16): COPYMACRO_2D(Uint8, Uint16)  break;
				case sizeof(Uint32): COPYMACRO_2D(Uint8, Uint32)  break;
				default:
					if(!PySurface_Unlock(surfobj)) return NULL;
					return RAISE(PyExc_ValueError, "unsupported datatype for array\n");
			}
		}
		break;
	case 2:
		if(array->nd == 2) {
			switch(array->descr->elsize) {
				case sizeof(Uint8):  COPYMACRO_2D(Uint16, Uint8)  break;
				case sizeof(Uint16): COPYMACRO_2D(Uint16, Uint16)  break;
				case sizeof(Uint32): COPYMACRO_2D(Uint16, Uint32)  break;
				default:
					if(!PySurface_Unlock(surfobj)) return NULL;
					return RAISE(PyExc_ValueError, "unsupported datatype for array\n");
			}
		} else {
			switch(array->descr->elsize) {
				case sizeof(Uint8): COPYMACRO_3D(Uint16, Uint8)  break;
				case sizeof(Uint16):COPYMACRO_3D(Uint16, Uint16)  break;
				case sizeof(Uint32):COPYMACRO_3D(Uint16, Uint32)  break;
				default:
					if(!PySurface_Unlock(surfobj)) return NULL;
					return RAISE(PyExc_ValueError, "unsupported datatype for array\n");
			}
		}
		break;
	case 3:
		if(array->nd == 2) {
			switch(array->descr->elsize) {
				case sizeof(Uint8):  COPYMACRO_2D_24(Uint8)  break;
				case sizeof(Uint16): COPYMACRO_2D_24(Uint16)  break;
				case sizeof(Uint32): COPYMACRO_2D_24(Uint32)  break;
				default:
					if(!PySurface_Unlock(surfobj)) return NULL;
					return RAISE(PyExc_ValueError, "unsupported datatype for array\n");
			}
		} else {
			switch(array->descr->elsize) {
				case sizeof(Uint8): COPYMACRO_3D_24(Uint8)  break;
				case sizeof(Uint16):COPYMACRO_3D_24(Uint16)  break;
				case sizeof(Uint32):COPYMACRO_3D_24(Uint32)  break;
				default:
					if(!PySurface_Unlock(surfobj)) return NULL;
					return RAISE(PyExc_ValueError, "unsupported datatype for array\n");
			}
		}
		break;
	case 4:
		if(array->nd == 2) {
			switch(array->descr->elsize) {
				case sizeof(Uint8):  COPYMACRO_2D(Uint32, Uint8)  break;
				case sizeof(Uint16): COPYMACRO_2D(Uint32, Uint16)  break;
				case sizeof(Uint32): COPYMACRO_2D(Uint32, Uint32)  break;
			default:
					if(!PySurface_Unlock(surfobj)) return NULL;
					return RAISE(PyExc_ValueError, "unsupported datatype for array\n");
			}
		} else {
			switch(array->descr->elsize) {
				case sizeof(Uint8): COPYMACRO_3D(Uint32, Uint8)  break;
				case sizeof(Uint16):COPYMACRO_3D(Uint32, Uint16)  break;
				case sizeof(Uint32):COPYMACRO_3D(Uint32, Uint32)  break;
				default:
					if(!PySurface_Unlock(surfobj)) return NULL;
					return RAISE(PyExc_ValueError, "unsupported datatype for array\n");
			}
		}
		break;
	default:
		if(!PySurface_Unlock(surfobj)) return NULL;
		return RAISE(PyExc_RuntimeError, "unsupported bit depth for image");
	}

	if(!PySurface_Unlock(surfobj)) return NULL;
	RETURN_NONE
}


    /*DOC*/ static char doc_make_surface[] =
    /*DOC*/    "pygame.surfarray.make_surface(array) -> Surface\n"
    /*DOC*/    "create a new Surface from array data\n"
    /*DOC*/    "\n"
    /*DOC*/    "Create a new software surface that closely resembles\n"
    /*DOC*/    "the data and format of the image array data.\n"
    /*DOC*/ ;

PyObject* make_surface(PyObject* self, PyObject* arg)
{
	PyObject *arrayobj, *surfobj, *args;
	SDL_Surface* surf;
	PyArrayObject* array;
	int sizex, sizey, bitsperpixel;
        Uint32 rmask, gmask, bmask;

	if(!PyArg_ParseTuple(arg, "O!", &PyArray_Type, &arrayobj))
		return NULL;
        array = (PyArrayObject*)arrayobj;

	if(!(array->nd == 2 || (array->nd == 3 && array->dimensions[2] == 3)))
		return RAISE(PyExc_ValueError, "must be a valid 2d or 3d array\n");
        if(array->descr->type_num > PyArray_LONG)
                return RAISE(PyExc_ValueError, "Invalid array datatype for surface");

        if(array->nd == 2)
        {
            bitsperpixel = 8;
            rmask = gmask = bmask = 0;
        }
        else
        {
            bitsperpixel = 32;
            rmask = 0xFF<<16; gmask = 0xFF<<8; bmask = 0xFF;
        }
        sizex = array->dimensions[0];
        sizey = array->dimensions[1];

        surf = SDL_CreateRGBSurface(0, sizex, sizey, bitsperpixel, rmask, gmask, bmask, 0);
        if(!surf)
                return RAISE(PyExc_SDLError, SDL_GetError());
        surfobj = PySurface_New(surf);
        if(!surfobj)
        {
            SDL_FreeSurface(surf);
            return NULL;
        }

        args = Py_BuildValue("(OO)", surfobj, array);
        if(!args)
        {
            Py_DECREF(surfobj);
            return NULL;
        }
        blit_array(NULL, args);
        Py_DECREF(args);

        if(PyErr_Occurred())
        {
            Py_DECREF(surfobj);
            return NULL;
        }
        return surfobj;
}



static PyMethodDef surfarray_builtins[] =
{
	{ "pixels2d", pixels2d, 1, doc_pixels2d },
	{ "pixels3d", pixels3d, 1, doc_pixels3d },
	{ "pixels_alpha", pixels_alpha, 1, doc_pixels_alpha },
	{ "array2d", array2d, 1, doc_array2d },
	{ "array3d", array3d, 1, doc_array3d },
	{ "array_alpha", array_alpha, 1, doc_array_alpha },
	{ "array_colorkey", array_colorkey, 1, doc_array_colorkey },
	{ "map_array", map_array, 1, doc_map_array },
/*	{ "unmap_array", unmap_array, 1, doc_unmap_array },*/
	{ "blit_array", blit_array, 1, doc_blit_array },
/*	{ "clamp_array", clamp_array, 1, doc_clamp_array }, not quick enough to be worthwhile :[ */
        { "make_surface", make_surface, 1, doc_make_surface },

	{ NULL, NULL }
};





    /*DOC*/ static char doc_pygame_surfarray_MODULE[] =
    /*DOC*/    "Contains routines for mixing numeric arrays with\n"
    /*DOC*/    "surfaces. You can create arrays that directly reference\n"
    /*DOC*/    "the pixel data of an image. Sometimes this can be limited\n"
    /*DOC*/    "to the pixel format of the Surface, so you can also create\n"
    /*DOC*/    "independent copies from any format.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The image arrays are indexes 'X' axis first. This is different\n"
    /*DOC*/    "than traditional C memory access, where images are often indexed\n"
    /*DOC*/    "with the 'Y' axis first. All this means is to access pixel values\n"
    /*DOC*/    "in the array, you index them as 'X, Y' pairs. myarray[10,20] will\n"
    /*DOC*/    "provide you the pixel at 10, 20 in the image. If you prefer to\n"
    /*DOC*/    "work with the traditional framebuffer indices, use the arrays\n"
    /*DOC*/    "'transpose()' method to create the alternate view of the pixel\n"
    /*DOC*/    "data.\n"
    /*DOC*/    "\n"
    /*DOC*/ ;

PYGAME_EXPORT
void initsurfarray(void)
{
	PyObject *module, *dict;

    /* create the module */
	module = Py_InitModule3("surfarray", surfarray_builtins, doc_pygame_surfarray_MODULE);
	dict = PyModule_GetDict(module);

	/*imported needed apis*/
	import_pygame_base();
	import_pygame_surface();
	import_array();
    /*needed for Numeric in python2.3*/
        PyImport_ImportModule("Numeric");
}



