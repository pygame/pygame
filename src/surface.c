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
 *  PyGAME Surface module
 */
#define PYGAMEAPI_SURFACE_INTERNAL
#include "pygame.h"


staticforward PyTypeObject PySurface_Type;
static PyObject* PySurface_New(SDL_Surface* info);
#define PySurface_Check(x) ((x)->ob_type == &PySurface_Type)



/* surface object methods */


    /*DOC*/ static char doc_surf_get_at[] =
    /*DOC*/    "Surface.get_at([x, y]) -> int\n"
    /*DOC*/    "get a pixel color\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the mapped pixel color at the coordinates\n"
    /*DOC*/    "given point.\n"
    /*DOC*/ ;

static PyObject* surf_get_at(PyObject* self, PyObject* arg)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	SDL_PixelFormat* format = surf->format;
	Uint8* pixels = (Uint8*)surf->pixels;
	int x, y;

	if(!PyArg_ParseTuple(arg, "(ii)", &x, &y))
		return NULL;

	if(x < 0 || x >= surf->w || y < 0 || y >= surf->h)
		return RAISE(PyExc_IndexError, "buffer index out of range");

	if(!pixels)
		return RAISE(PyExc_SDLError, "Surface must be locked for pixel access");

	switch(format->BytesPerPixel)
	{
		case 1:
		{
			Uint8 col = *((Uint8*)pixels + y * surf->pitch + x);
			return PyInt_FromLong(col);
		}
		break;
		case 2:
		{
			Uint16 col = *((Uint16*)(pixels + y * surf->pitch) + x);
			return PyInt_FromLong(col); 
		}
		break;
		case 3:
		{
			Uint32 col;
			Uint8* byte_buf;
			
			byte_buf = ((Uint8*)(pixels + y * surf->pitch) + x * 3);
			col =
				*(byte_buf + (format->Rshift >> 3)) << format->Rshift |
				*(byte_buf + (format->Gshift >> 3)) << format->Gshift |
				*(byte_buf + (format->Bshift >> 3)) << format->Bshift;
			return PyInt_FromLong(col);   
		}
		break;
		case 4:
		{
			Uint32 col = *((Uint32*)(pixels + y * surf->pitch) + x);
			return PyInt_FromLong(col); 
		}
		break;
	}
	return RAISE(PyExc_RuntimeError, "Unable to determine color depth.");
}



    /*DOC*/ static char doc_surf_set_at[] =
    /*DOC*/    "Surface.set_at([x, y], pixel) -> None\n"
    /*DOC*/    "set pixel at given position\n"
    /*DOC*/    "\n"
    /*DOC*/    "Assigns a mapped pixel color to the image at the give position.\n"
    /*DOC*/ ;

static PyObject* surf_set_at(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	SDL_PixelFormat* format = surf->format;
	Uint8* pixels = (Uint8*)surf->pixels;
	int x, y;
	Uint32 color;
	
	if(!PyArg_ParseTuple(args, "(ii)i", &x, &y, &color))
		return NULL;

	if(x < 0 || x >= surf->w || y < 0 || y >= surf->h)
	{
		printf("%d,%d  -  %d,%d\n",x,y, surf->w, surf->h);
		PyErr_SetString(PyExc_IndexError, "buffer index out of range");
		return NULL;
	}

	if(!pixels)
		return RAISE(PyExc_SDLError, "Surface must be locked for pixel access");

	switch(format->BytesPerPixel)
	{
		case 1:
		{
			*((Uint8*)pixels + y * surf->pitch + x) = (Uint8)color;
		}
		break;
		case 2:
		{
			*((Uint16*)(pixels + y * surf->pitch) + x) = (Uint16)color;
		}
		break;
		case 3:
		{
			Uint8* byte_buf = (Uint8*)(pixels + y * surf->pitch) + x * 3;
			Uint8 r, g, b;

			r = (color & format->Rmask) >> format->Rshift;
			g = (color & format->Gmask) >> format->Gshift;
			b = (color & format->Bmask) >> format->Bshift;

			*(byte_buf + (format->Rshift >> 3)) = r;
			*(byte_buf + (format->Gshift >> 3)) = g;
			*(byte_buf + (format->Bshift >> 3)) = b;	
		}
		break;
		case 4:
		{
			*((Uint32*)(pixels + y * surf->pitch) + x) = color;
		}
		break;
		default:
			return RAISE(PyExc_SDLError, "Unable to determine color depth.");
	}

	RETURN_NONE
}



    /*DOC*/ static char doc_surf_map_rgb[] =
    /*DOC*/    "Surface.map_rgb([r, g, b]) -> int\n"
    /*DOC*/    "convert RGB into a mapped color\n"
    /*DOC*/    "\n"
    /*DOC*/    "Uses the Surface format to convert RGB into a mapped color value.\n"
    /*DOC*/    "Note that this will work if the RGB is passed as three arguments\n"
    /*DOC*/    "instead of a sequence.\n"
    /*DOC*/ ;

static PyObject* surf_map_rgb(PyObject* self,PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	Uint8 r, g, b;

	if(PyTuple_Size(args)==1)
	{
		if(!PyArg_ParseTuple(args, "(bbb)", &r, &g, &b))
			return NULL;
	}
	else if(!PyArg_ParseTuple(args, "bbb", &r, &g, &b))
		return NULL;

	return PyInt_FromLong(SDL_MapRGB(surf->format, r, g, b));
}



    /*DOC*/ static char doc_surf_unmap_rgb[] =
    /*DOC*/    "Surface.unmap_rgb(color) -> r, g, b\n"
    /*DOC*/    "convert mapped color into RGB\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function returns the RGB components for a mapped color\n"
    /*DOC*/    "value.\n"
    /*DOC*/ ;

static PyObject* surf_unmap_rgb(PyObject* self,PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	Uint32 col;
	Uint8 r, g, b;
	
	if(!PyArg_ParseTuple(args, "i", &col))
		return NULL;

	SDL_GetRGB(col,surf->format, &r, &g, &b);	

	return Py_BuildValue("(bbb)", r, g, b);
}


    /*DOC*/ static char doc_surf_map_rgba[] =
    /*DOC*/    "Surface.map_rgba([r, g, b, a]) -> int\n"
    /*DOC*/    "convert RGBA into a mapped color\n"
    /*DOC*/    "\n"
    /*DOC*/    "Uses the Surface format to convert RGBA into a mapped color\n"
    /*DOC*/    "value. It is safe to call this on a surface with no pixel alpha.\n"
    /*DOC*/    "The alpha will simply be ignored.\n"
    /*DOC*/ ;

static PyObject* surf_map_rgba(PyObject* self,PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	Uint8 r, g, b, a;

	if(PyTuple_Size(args)==1)
	{
		if(!PyArg_ParseTuple(args, "(bbbb)", &r, &g, &b, &a))
			return NULL;
	}
	else if(!PyArg_ParseTuple(args, "bbbb", &r, &g, &b, &a))
		return NULL;

	return PyInt_FromLong(SDL_MapRGBA(surf->format, r, g, b, a));
}



    /*DOC*/ static char doc_surf_unmap_rgba[] =
    /*DOC*/    "Surface.unmap_rgba(color) -> r, g, b, a\n"
    /*DOC*/    "convert mapped color into RGBA\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function returns the RGB components for a mapped color\n"
    /*DOC*/    "value. For surfaces with no alpha, the alpha will always be 255.\n"
    /*DOC*/ ;

static PyObject* surf_unmap_rgba(PyObject* self,PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	Uint32 col;
	Uint8 r, g, b, a;
	
	if(!PyArg_ParseTuple(args, "i", &col))
		return NULL;

	SDL_GetRGBA(col,surf->format, &r, &g, &b, &a);	

	return Py_BuildValue("(bbb)", r, g, b, a);
}



    /*DOC*/ static char doc_surf_lock[] =
    /*DOC*/    "Surface.lock() -> None\n"
    /*DOC*/    "locks Surface for pixel access\n"
    /*DOC*/    "\n"
    /*DOC*/    "On accelerated surfaces, it is usually required to lock the\n"
    /*DOC*/    "surface before you can access the pixel values. To be safe, it is\n"
    /*DOC*/    "always a good idea to lock the surface before entering a block of\n"
    /*DOC*/    "code that changes or accesses the pixel values. The surface must\n"
    /*DOC*/    "not be locked when performing other pyGame functions on it like\n"
    /*DOC*/    "fill and blit.\n"
    /*DOC*/    "\n"
    /*DOC*/    "You can doublecheck to really make sure a lock is needed by\n"
    /*DOC*/    "calling the mustlock() member. This should not be needed, since\n"
    /*DOC*/    "it is usually recommended to lock anyways and work with all\n"
    /*DOC*/    "surface types. If the surface does not need to be locked, the\n"
    /*DOC*/    "operation will return quickly with minute overhead.\n"
    /*DOC*/    "\n"
    /*DOC*/    "On some platforms a necessary lock can shut off some parts of the\n"
    /*DOC*/    "system. This is not a problem unless you leave surfaces locked\n"
    /*DOC*/    "for long periouds of time. Only keep the surface locked when you\n"
    /*DOC*/    "need the pixel access. At the same time, it is not a good too\n"
    /*DOC*/    "repeatedly lock and unlock the surface inside tight loops. It is\n"
    /*DOC*/    "fine to leave the surface locked while needed, just don't be\n"
    /*DOC*/    "lazy.\n"
    /*DOC*/ ;

static PyObject* surf_lock(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	if(SDL_LockSurface(surf) == -1)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE
}



    /*DOC*/ static char doc_surf_unlock[] =
    /*DOC*/    "Surface.unlock() -> None\n"
    /*DOC*/    "locks Surface for pixel access\n"
    /*DOC*/    "\n"
    /*DOC*/    "After a surface has been locked, you will need to unlock it when\n"
    /*DOC*/    "you are done.\n"
    /*DOC*/    "\n"
    /*DOC*/    "You can doublecheck to really make sure a lock is needed by\n"
    /*DOC*/    "calling the mustlock() member. This should not be needed, since\n"
    /*DOC*/    "it is usually recommended to lock anyways and work with all\n"
    /*DOC*/    "surface types. If the surface does not need to be locked, the\n"
    /*DOC*/    "operation will return quickly with minute overhead.\n"
    /*DOC*/ ;

static PyObject* surf_unlock(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	SDL_UnlockSurface(surf);
	RETURN_NONE
}



    /*DOC*/ static char doc_surf_mustlock[] =
    /*DOC*/    "Surface.mustlock() -> bool\n"
    /*DOC*/    "check if the surface needs locking\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns true if the surface really does need locking to gain\n"
    /*DOC*/    "pixel access. Usually the overhead of checking before locking\n"
    /*DOC*/    "outweight the overhead of just locking any surface before access.\n"
    /*DOC*/ ;

static PyObject* surf_mustlock(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	return PyInt_FromLong(SDL_MUSTLOCK(surf));
}



    /*DOC*/ static char doc_surf_get_palette[] =
    /*DOC*/    "Surface.get_palette() -> [[r, g, b], ...]\n"
    /*DOC*/    "get the palette\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will return the an array of all the color indexes in the\n"
    /*DOC*/    "Surface's palette.\n"
    /*DOC*/ ;

static PyObject* surf_get_palette(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	SDL_Palette* pal = surf->format->palette;
	PyObject* list;
	int i;
		
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	if(!pal)
		return RAISE(PyExc_SDLError, "Surface is not palettized\n");

	list = PyTuple_New(pal->ncolors);
	if(!list)
		return NULL;

	for(i = 0;i < pal->ncolors;i++)
	{
		PyObject* color;
		SDL_Color* c = &pal->colors[i];

		color = Py_BuildValue("(bbb)", c->r, c->g, c->b);
		if(!color)
		{
			Py_DECREF(list);
			return NULL;
		}

		PyTuple_SET_ITEM(list, i, color);
	}

	return list;
}



    /*DOC*/ static char doc_surf_get_palette_at[] =
    /*DOC*/    "Surface.get_palette_at(index) -> r, g, b\n"
    /*DOC*/    "get a palette entry\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will retreive an individual color entry from the Surface's\n"
    /*DOC*/    "palette.\n"
    /*DOC*/ ;

static PyObject* surf_get_palette_at(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	SDL_Palette* pal = surf->format->palette;
	SDL_Color* c;
	int index;

	if(!PyArg_ParseTuple(args, "i", &index))
		return NULL;

	if(!pal)
		return RAISE(PyExc_SDLError, "Surface is not palettized\n");
	if(index >= pal->ncolors || index < 0)
		return RAISE(PyExc_IndexError, "index out of bounds");

	c = &pal->colors[index];
	return Py_BuildValue("(bbb)", c->r, c->g, c->b);
}



    /*DOC*/ static char doc_surf_set_palette[] =
    /*DOC*/    "Surface.set_palette([[r, g, b], ...]) -> None\n"
    /*DOC*/    "set the palette\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will replace the entire palette with color\n"
    /*DOC*/    "information you provide.\n"
    /*DOC*/ ;

static PyObject* surf_set_palette(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	SDL_Palette* pal = surf->format->palette;
	SDL_Color* colors;
	PyObject* list, *item;
	int i, len;
	short r, g, b;
	
	if(!PyArg_ParseTuple(args, "O", &list))
		return NULL;
	if(!PySequence_Check(list))
		return RAISE(PyExc_ValueError, "Argument must be a sequence type");


	if(!pal)
		return RAISE(PyExc_SDLError, "Surface is not palettized\n");

	len = min(pal->ncolors, PySequence_Length(list));

	colors = (SDL_Color*)malloc(len * sizeof(SDL_Color));
	if(!colors)
		return NULL;
	
	for(i = 0; i < len; i++)
	{
		item = PySequence_GetItem(list, i);

		if(!PySequence_Check(item) || PySequence_Length(item) != 3)
		{
			Py_DECREF(item);
			free((char*)colors);
			return RAISE(PyExc_TypeError, "takes a sequence of sequence of RGB");
		}
		if(!ShortFromObjIndex(item, 0, &r) || !ShortFromObjIndex(item, 1, &g) || !ShortFromObjIndex(item, 2, &b))
			return RAISE(PyExc_TypeError, "RGB sequence must contain numeric values");

		colors[i].r = (unsigned char)r;
		colors[i].g = (unsigned char)g;
		colors[i].b = (unsigned char)b;
	
		Py_DECREF(item);
	}

	SDL_SetColors(surf, colors, 0, len);

	free((char*)colors);

	RETURN_NONE
}



    /*DOC*/ static char doc_surf_set_palette_at[] =
    /*DOC*/    "Surface.set_palette_at(index, [r, g, b]) -> None\n"
    /*DOC*/    "set a palette entry\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function sets the palette color at a specific entry.\n"
    /*DOC*/ ;

static PyObject* surf_set_palette_at(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	SDL_Palette* pal = surf->format->palette;
	SDL_Color color;
	int index;
	Uint8 r, g, b;

	if(!PyArg_ParseTuple(args, "i(bbb)", &index, &r, &g, &b))
		return NULL;

	if(!pal)
	{
		PyErr_SetString(PyExc_SDLError, "Surface is not palettized\n");
		return NULL;
	}

	if(index >= pal->ncolors || index <= 0)
	{
		PyErr_SetString(PyExc_IndexError, "index out of bounds");
		return NULL;
	}

	color.r = r;
	color.g = g;
	color.b = b;

	SDL_SetColors(surf, &color, index, 1);

	RETURN_NONE
}



    /*DOC*/ static char doc_surf_set_colorkey[] =
    /*DOC*/    "Surface.set_colorkey([color, [flags]]) -> None\n"
    /*DOC*/    "change colorkey information\n"
    /*DOC*/    "\n"
    /*DOC*/    "Set the colorkey for the surface by passing a mapped color value\n"
    /*DOC*/    "as the color argument. If no arguments are passed, colorkeying\n"
    /*DOC*/    "will be disabled for this surface.\n"
    /*DOC*/    "\n"
    /*DOC*/    "If your image is nonchanging and will be used repeatedly, you\n"
    /*DOC*/    "will probably want to pass the RLEACCEL flag to the call. This\n"
    /*DOC*/    "will take a short time to compile your surface, and increase the\n"
    /*DOC*/    "blitting speed.\n"
    /*DOC*/ ;

static PyObject* surf_set_colorkey(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	Uint32 flags = 0, key = 0;

	if(!PyArg_ParseTuple(args, "|ii", &key, &flags))
		return NULL;
	
	if(PyTuple_Size(args) > 0)
		flags |= SDL_SRCCOLORKEY;

	if(SDL_SetColorKey(surf, flags, key) == -1)
		return RAISE(PyExc_SDLError, SDL_GetError());
	RETURN_NONE
}


    /*DOC*/ static char doc_surf_get_colorkey[] =
    /*DOC*/    "Surface.get_colorkey() -> color\n"
    /*DOC*/    "query colorkey\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the current mapped color value being used for\n"
    /*DOC*/    "colorkeying. If colorkeying is not enabled for this surface, it\n"
    /*DOC*/    "returns None\n"
    /*DOC*/ ;

static PyObject* surf_get_colorkey(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;
	
	if(surf->flags&SDL_SRCCOLORKEY)
		return PyInt_FromLong(surf->format->colorkey);

	RETURN_NONE
}


    /*DOC*/ static char doc_surf_set_alpha[] =
    /*DOC*/    "Surface.set_alpha([alpha, [flags]]) -> None\n"
    /*DOC*/    "change alpha information\n"
    /*DOC*/    "\n"
    /*DOC*/    "Set the overall transparency for the surface. If no alpha is\n"
    /*DOC*/    "passed, alpha blending is disabled for the surface. An alpha of 0\n"
    /*DOC*/    "is fully transparent, an alpha of 255 is fully opaque.\n"
    /*DOC*/    "\n"
    /*DOC*/    "If your surface has a pixel alpha channel, it will override the\n"
    /*DOC*/    "overall surface transparency. You'll need to change the actual\n"
    /*DOC*/    "pixel transparency to make changes.\n"
    /*DOC*/    "\n"
    /*DOC*/    "If your image is nonchanging and will be used repeatedly, you\n"
    /*DOC*/    "will probably want to pass the RLEACCEL flag to the call. This\n"
    /*DOC*/    "will take a short time to compile your surface, and increase the\n"
    /*DOC*/    "blitting speed.\n"
    /*DOC*/ ;

static PyObject* surf_set_alpha(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	Uint32 flags = 0;
	Uint8 alpha = 0;

	if(!PyArg_ParseTuple(args, "|bi", &alpha, &flags))
		return NULL;

	if(PyTuple_Size(args) > 0)
		flags |= SDL_SRCALPHA;

	if(SDL_SetAlpha(surf, flags, alpha) == -1)
		return RAISE(PyExc_SDLError, SDL_GetError());
	RETURN_NONE
}


    /*DOC*/ static char doc_surf_get_alpha[] =
    /*DOC*/    "Surface.get_alpha() -> alpha\n"
    /*DOC*/    "query alpha information\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the current alpha value for the Surface. If transparency\n"
    /*DOC*/    "is disabled for the Surface, it returns None.\n"
    /*DOC*/ ;

static PyObject* surf_get_alpha(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;
	
	if(surf->flags&SDL_SRCALPHA)
		return PyInt_FromLong(surf->format->alpha);

	RETURN_NONE
}


    /*DOC*/ static char doc_surf_convert[] =
    /*DOC*/    "Surface.convert([src_surface]) -> Surface\n"
    /*DOC*/    "new copy of surface with different format\n"
    /*DOC*/    "\n"
    /*DOC*/    "Creates a new copy of the surface with the desired pixel format.\n"
    /*DOC*/    "Surfaces with the same pixel format will blit much faster than\n"
    /*DOC*/    "those with mixed formats. The pixel format of the new surface\n"
    /*DOC*/    "will match the format given as the argument. If no surface is\n"
    /*DOC*/    "given, the new surface will have the same pixel format as the\n"
    /*DOC*/    "current display.\n"
    /*DOC*/ ;

static PyObject* surf_convert(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	PySurfaceObject* srcsurf = NULL;
	SDL_Surface* src;
	SDL_Surface* newsurf;
	
	if(!PyArg_ParseTuple(args, "|O!", &PySurface_Type, &srcsurf))
		return NULL;

	if(srcsurf)
	{
		src = PySurface_AsSurface(srcsurf);
		newsurf = SDL_ConvertSurface(surf, src->format, src->flags);
	}
	else
		newsurf = SDL_DisplayFormat(surf);

	return PySurface_New(newsurf);
}



    /*DOC*/ static char doc_surf_convert_alpha[] =
    /*DOC*/    "Surface.convert_alpha([src_surface]) -> Surface\n"
    /*DOC*/    "new copy of surface with different format and per pixel alpha\n"
    /*DOC*/    "\n"
    /*DOC*/    "Creates a new copy of the surface with the desired pixel format.\n"
    /*DOC*/    "The new surface will be in a format suited for quick blitting to\n"
    /*DOC*/    "the given format with per pixel alpha. If no surface is given,\n"
    /*DOC*/    "the new surface will be optimized for blittint to the current\n"
    /*DOC*/    "display.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Unlike the convert() method, the pixel format for the new image\n"
    /*DOC*/    "will not be exactly the same as the requested source, but it will\n"
    /*DOC*/    "be optimized for fast alpha blitting to the destination.\n"
    /*DOC*/ ;

static PyObject* surf_convert_alpha(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	PySurfaceObject* srcsurf = NULL;
	SDL_Surface* newsurf, *src;
	
	if(!PyArg_ParseTuple(args, "|O!", &PySurface_Type, &srcsurf))
		return NULL;

	if(srcsurf)
	{
		/*hmm, we have to figure this out, not all depths have good support for alpha*/
		src = PySurface_AsSurface(srcsurf);
		newsurf = SDL_DisplayFormatAlpha(surf);
	}
	else
		newsurf = SDL_DisplayFormatAlpha(surf);

	return PySurface_New(newsurf);
}


    /*DOC*/ static char doc_surf_set_clip[] =
    /*DOC*/    "Surface.set_clip([rectstyle])) -> None\n"
    /*DOC*/    "assign destination clipping rectangle\n"
    /*DOC*/    "\n"
    /*DOC*/    "Assigns the destination clipping rectangle for the Surface. When\n"
    /*DOC*/    "blit or fill operations are performed on the Surface, they are\n"
    /*DOC*/    "restricted to the inside of the clipping rectangle. If no\n"
    /*DOC*/    "rectangle is passed, the clipping region is set to the entire\n"
    /*DOC*/    "Surface area. The rectangle you pass will be clipped to the area of\n"
    /*DOC*/    "the Surface.\n"
    /*DOC*/ ;

static PyObject* surf_set_clip(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	GAME_Rect *rect=NULL, temp;
	int result;

	if(PyTuple_Size(args))
	{
		rect = GameRect_FromObject(args, &temp);
		if(!rect)
			return RAISE(PyExc_ValueError, "invalid rectstyle object");
	}
		
	result = SDL_SetClipRect(surf, (SDL_Rect*)rect);
	if(result == -1)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE
}



    /*DOC*/ static char doc_surf_get_clip[] =
    /*DOC*/    "Surface.get_clip() -> rect\n"
    /*DOC*/    "query the clipping area\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the current destination clipping area being used by the\n"
    /*DOC*/    "Surface. If the clipping area is not set, it will return a\n"
    /*DOC*/    "rectangle containing the full Surface area.\n"
    /*DOC*/ ;

static PyObject* surf_get_clip(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	return PyRect_New((GAME_Rect*)&surf->clip_rect);
}



    /*DOC*/ static char doc_surf_fill[] =
    /*DOC*/    "Surface.fill(color, [rectstyle])) -> Rect\n"
    /*DOC*/    "fill areas of a Surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Fills the specified area of the Surface with the mapped color\n"
    /*DOC*/    "value. If no destination rectangle is supplied, it will fill the\n"
    /*DOC*/    "entire Surface.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The fill is subject to be clipped by the active clipping\n"
    /*DOC*/    "rectangle. The return value contains the actual area filled.\n"
    /*DOC*/ ;

static PyObject* surf_fill(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	GAME_Rect *rect, temp;
	PyObject* r = NULL;
	Uint32 color;
	int result;
	
	if(!PyArg_ParseTuple(args, "i|O", &color, &r))
		return NULL;

	if(!r)
	{
		rect = &temp;
		temp.x = temp.y = (short)0;
		temp.w = surf->w;
		temp.h = surf->h;
	}
	else if(!(rect = GameRect_FromObject(r, &temp)))
		return RAISE(PyExc_ValueError, "invalid rectstyle object");

	result = SDL_FillRect(surf, (SDL_Rect*)rect, color);
	if(result == -1)
		return RAISE(PyExc_SDLError, SDL_GetError());
	return PyRect_New(rect);
}


/*clip blit source rect to source image*/
static void screencroprect(GAME_Rect* r, int w, int h)
{
	if(r->x >= w || r->y >= h)
		r->x = r->y = r->w = r->h = 0;
	else
	{
		if(r->x < 0) r->x = 0;
		if(r->y < 0) r->y = 0;
		if(r->x + r->w >= w) r->w = (w-1)-r->x;
		if(r->y + r->h >= h) r->h = (h-1)-r->y;
	}
}



    /*DOC*/ static char doc_surf_blit[] =
    /*DOC*/    "Surface.blit(source, destoffset, [sourcerect]) -> Rect\n"
    /*DOC*/    "copy a one Surface to another.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The blitting will transfer one surface to another. It will\n"
    /*DOC*/    "respect any special modes like colorkeying and alpha. If hardware\n"
    /*DOC*/    "support is available, it will be used. The given source is the\n"
    /*DOC*/    "Surface to copy from. The destoffset is a 2-number-sequence that\n"
    /*DOC*/    "specifies where on the destination Surface the blit happens.\n"
    /*DOC*/    "When sourcerect isn't supplied, the blit will copy the\n"
    /*DOC*/    "entire source surface. If you would like to copy only a portion\n"
    /*DOC*/    "of the source, use the sourcerect argument to control\n"
    /*DOC*/    "what area is copied.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The blit is subject to be clipped by the active clipping\n"
    /*DOC*/    "rectangle. The return value contains the actual area blitted.\n"
    /*DOC*/ ;

static PyObject* surf_blit(PyObject* self, PyObject* args)
{
	SDL_Surface* src, *dest = PySurface_AsSurface(self);
	GAME_Rect* src_rect, temp;
	PyObject* srcobject, *argrect = NULL;
	int dx, dy, result;
	SDL_Rect dest_rect;

	if(!PyArg_ParseTuple(args, "O!(ii)|O", &PySurface_Type, &srcobject, &dx, &dy, &argrect))
		return NULL;
	src = PySurface_AsSurface(srcobject);

	if(argrect)
	{
		if(!(src_rect = GameRect_FromObject(argrect, &temp)))
			return RAISE(PyExc_TypeError, "Invalid rectstyle argument");
	}
	else
	{
		temp.x = temp.y = 0;
		temp.w = src->w;
		temp.h = src->h;
		src_rect = &temp;
	}

	dest_rect.x = (short)dx;
	dest_rect.y = (short)dy;
	dest_rect.w = (unsigned short)src_rect->w;
	dest_rect.h = (unsigned short)src_rect->h;

	result = SDL_BlitSurface(src, (SDL_Rect*)src_rect, dest, &dest_rect);
	if(result == -1)
		return RAISE(PyExc_SDLError, SDL_GetError());

	return PyRect_New((GAME_Rect*)&dest_rect);
}


    /*DOC*/ static char doc_surf_get_flags[] =
    /*DOC*/    "Surface.get_flags() -> flags\n"
    /*DOC*/    "query the surface width\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the current state flags for the surface.\n"
    /*DOC*/ ;

static PyObject* surf_get_flags(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	return PyInt_FromLong(surf->flags);
}



    /*DOC*/ static char doc_surf_get_pitch[] =
    /*DOC*/    "Surface.get_pitch() -> pitch\n"
    /*DOC*/    "query the surface pitch\n"
    /*DOC*/    "\n"
    /*DOC*/    "The surface pitch is the number of bytes used in each\n"
    /*DOC*/    "scanline. This function should rarely needed, mainly for\n"
    /*DOC*/    "any special-case debugging.\n"
    /*DOC*/ ;

static PyObject* surf_get_pitch(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	return PyInt_FromLong(surf->pitch);
}



    /*DOC*/ static char doc_surf_get_size[] =
    /*DOC*/    "Surface.get_size() -> x, y\n"
    /*DOC*/    "query the surface size\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the width and height of the Surface.\n"
    /*DOC*/ ;

static PyObject* surf_get_size(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	return Py_BuildValue("(ii)", surf->w, surf->h);
}



    /*DOC*/ static char doc_surf_get_width[] =
    /*DOC*/    "Surface.get_width() -> width\n"
    /*DOC*/    "query the surface width\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the width of the Surface.\n"
    /*DOC*/ ;

static PyObject* surf_get_width(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	return PyInt_FromLong(surf->w);
}



    /*DOC*/ static char doc_surf_get_height[] =
    /*DOC*/    "Surface.get_height() -> height\n"
    /*DOC*/    "query the surface height\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the height of the Surface.\n"
    /*DOC*/ ;

static PyObject* surf_get_height(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	return PyInt_FromLong(surf->h);
}



    /*DOC*/ static char doc_surf_get_rect[] =
    /*DOC*/    "Surface.get_rect() -> rect\n"
    /*DOC*/    "get a rectangle covering the entire surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns a new rectangle covering the entire surface.\n"
    /*DOC*/    "This rectangle will always start at 0, 0 with a width.\n"
    /*DOC*/    "and height the same size as the image.\n"
    /*DOC*/ ;

static PyObject* surf_get_rect(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	return PyRect_New4(0, 0, (short)surf->w, (short)surf->h);
}



    /*DOC*/ static char doc_surf_get_bitsize[] =
    /*DOC*/    "Surface.get_bitsize() -> int\n"
    /*DOC*/    "query size of pixel\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the number of bits used to represent each pixel. This\n"
    /*DOC*/    "value may not exactly fill the number of bytes used per pixel.\n"
    /*DOC*/    "For example a 15 bit Surface still requires a full 2 bytes.\n"
    /*DOC*/ ;

static PyObject* surf_get_bitsize(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	return PyInt_FromLong(surf->format->BitsPerPixel);
}


    /*DOC*/ static char doc_surf_get_bytesize[] =
    /*DOC*/    "Surface.get_bytesize() -> int\n"
    /*DOC*/    "query size of pixel\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the number of bytes used to store each pixel.\n"
    /*DOC*/ ;

static PyObject* surf_get_bytesize(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	return PyInt_FromLong(surf->format->BytesPerPixel);
}


    /*DOC*/ static char doc_surf_get_masks[] =
    /*DOC*/    "Surface.get_masks() -> redmask, greenmask, bluemask, alphamask\n"
    /*DOC*/    "get mapping bitmasks for each colorplane\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the bitmasks for each color plane. The bitmask is used to\n"
    /*DOC*/    "isolate each colorplane value from a mapped color value. A value\n"
    /*DOC*/    "of zero means that colorplane is not used (like alpha)\n"
    /*DOC*/ ;

static PyObject* surf_get_masks(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	return Py_BuildValue("(iiii)", surf->format->Rmask, surf->format->Gmask,
				surf->format->Bmask, surf->format->Amask);
}


    /*DOC*/ static char doc_surf_get_shifts[] =
    /*DOC*/    "Surface.get_shifts() -> redshift, greenshift, blueshift,\n"
    /*DOC*/    "alphashift\n"
    /*DOC*/    "get mapping shifts for each colorplane\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the bitshifts used for each color plane. The shift is\n"
    /*DOC*/    "determine how many bits left-shifted a colorplane value is in a\n"
    /*DOC*/    "mapped color value.\n"
    /*DOC*/ ;

static PyObject* surf_get_shifts(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	return Py_BuildValue("(iiii)", surf->format->Rshift, surf->format->Gshift,
				surf->format->Bshift, surf->format->Ashift);
}


    /*DOC*/ static char doc_surf_get_losses[] =
    /*DOC*/    "Surface.get_losses() -> redloss, greenloss, blueloss, alphaloss\n"
    /*DOC*/    "get mapping losses for each colorplane\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the bitloss for each color plane. The loss is the number\n"
    /*DOC*/    "of bits removed for each colorplane from a full 8 bits of\n"
    /*DOC*/    "resolution. A value of 8 usually indicates that colorplane is not\n"
    /*DOC*/    "used (like the alpha)\n"
    /*DOC*/ ;

static PyObject* surf_get_losses(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	return Py_BuildValue("(iiii)", surf->format->Rloss, surf->format->Gloss,
				surf->format->Bloss, surf->format->Aloss);
}



    /*DOC*/ static char doc_surf_save[] =
    /*DOC*/    "Surface.save(file) -> None\n"
    /*DOC*/    "save surface as BMP data\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will save your surface in the BMP format. The given file\n"
    /*DOC*/    "argument can be either a filename or a python file-like object\n"
    /*DOC*/    "to save the BMP image to.\n"
    /*DOC*/ ;

static PyObject* surf_save(PyObject* self, PyObject* arg)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	PyObject* file;
	SDL_RWops *rw;
	int result;
	if(!PyArg_ParseTuple(arg, "O", &file))
		return NULL;

	VIDEO_INIT_CHECK();

	if(PyString_Check(file))
	{
		char* name = PyString_AsString(file);
		Py_BEGIN_ALLOW_THREADS
		result = SDL_SaveBMP(surf, name);
		Py_END_ALLOW_THREADS
	}
	else
	{
		if(!(rw = RWopsFromPython(file)))
			return NULL;
		Py_BEGIN_ALLOW_THREADS
		result = SDL_SaveBMP_RW(surf, rw, 1);
		Py_END_ALLOW_THREADS
	}

	if(result == -1)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE;
}




static struct PyMethodDef surface_methods[] =
{
	{"get_at",			surf_get_at,		1, doc_surf_get_at },
	{"set_at",			surf_set_at,		1, doc_surf_set_at },

	{"map_rgb",			surf_map_rgb,		1, doc_surf_map_rgb },
	{"unmap_rgb",		surf_unmap_rgb,		1, doc_surf_unmap_rgb },
	{"map_rgba",		surf_map_rgba,		1, doc_surf_map_rgba },
	{"unmap_rgba",		surf_unmap_rgba,	1, doc_surf_unmap_rgba },

	{"get_palette",		surf_get_palette,	1, doc_surf_get_palette },
	{"get_palette_at",	surf_get_palette_at,1, doc_surf_get_palette_at },
	{"set_palette",		surf_set_palette,	1, doc_surf_set_palette },
	{"set_palette_at",	surf_set_palette_at,1, doc_surf_set_palette_at },

	{"lock",			surf_lock,			1, doc_surf_lock },
	{"unlock",			surf_unlock,		1, doc_surf_unlock },
	{"mustlock",		surf_mustlock,		1, doc_surf_mustlock },

	{"set_colorkey",	surf_set_colorkey,	1, doc_surf_set_colorkey },
	{"get_colorkey",	surf_get_colorkey,	1, doc_surf_get_colorkey },
	{"set_alpha",		surf_set_alpha,		1, doc_surf_set_alpha },
	{"get_alpha",		surf_get_alpha,		1, doc_surf_get_alpha },

	{"convert",			surf_convert,		1, doc_surf_convert },
	{"convert_alpha",	surf_convert_alpha,	1, doc_surf_convert_alpha },

	{"set_clip",		surf_set_clip,		1, doc_surf_set_clip },
	{"get_clip",		surf_get_clip,		1, doc_surf_get_clip },

	{"fill",			surf_fill,			1, doc_surf_fill },
	{"blit",			surf_blit,			1, doc_surf_blit },

	{"get_flags",		surf_get_flags,		1, doc_surf_get_flags },
	{"get_size",		surf_get_size,		1, doc_surf_get_size },
	{"get_width",		surf_get_width,		1, doc_surf_get_width },
	{"get_height",		surf_get_height,	1, doc_surf_get_height },
	{"get_rect",		surf_get_rect,		1, doc_surf_get_rect },
	{"get_pitch",		surf_get_pitch,		1, doc_surf_get_pitch },
	{"get_bitsize",		surf_get_bitsize,	1, doc_surf_get_bitsize },
	{"get_bytesize",	surf_get_bytesize,	1, doc_surf_get_bytesize },
	{"get_masks",		surf_get_masks,		1, doc_surf_get_masks },
	{"get_shifts",		surf_get_shifts,	1, doc_surf_get_shifts },
	{"get_losses",		surf_get_losses,	1, doc_surf_get_losses },

	{NULL,		NULL}
};



/* surface object internals */

static void surface_dealloc(PyObject* self)
{
	PySurfaceObject* surf = (PySurfaceObject*)self;

	if(SDL_WasInit(SDL_INIT_VIDEO))
		SDL_FreeSurface(surf->surf);
	PyMem_DEL(self);	
}



static PyObject *surface_getattr(PyObject *self, char *name)
{
	return Py_FindMethod(surface_methods, (PyObject *)self, name);
}


PyObject* surface_str(PyObject* self)
{
	char str[1024];
	SDL_Surface* surf = PySurface_AsSurface(self);
	const char* type;

	type = (surf->flags&SDL_HWSURFACE)?"HW":"SW";

	sprintf(str, "<Surface(%dx%dx%d %s)>", surf->w, surf->h, surf->format->BitsPerPixel, type);

	return PyString_FromString(str);
}

    /*DOC*/ static char doc_Surface_MODULE[] =
    /*DOC*/    "Surface objects represent a simple memory buffer of pixels.\n"
    /*DOC*/    "Surface objects can reside in system memory, or in special\n"
    /*DOC*/    "hardware memory, which can be hardware accelerated. Surfaces that\n"
    /*DOC*/    "are 8 bits per pixel use a colormap to represent their color\n"
    /*DOC*/    "values. All Surfaces with higher bits per pixel use a packed\n"
    /*DOC*/    "pixels to store their color values.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Surfaces can have many extra attributes like alpha planes,\n"
    /*DOC*/    "colorkeys, source rectangle clipping. These functions mainly\n"
    /*DOC*/    "effect how the Surface is blitted to other Surfaces. The blit\n"
    /*DOC*/    "routines will attempt to use hardware acceleration when possible,\n"
    /*DOC*/    "otherwise will use highly optimized software blitting methods.\n"
    /*DOC*/    "\n"
    /*DOC*/    "There is support for pixel access for the Surfaces. Pixel access\n"
    /*DOC*/    "on hardware surfaces is slow and not recommended. Pixels can be\n"
    /*DOC*/    "accessed using the get_at() and set_at() functions. These methods\n"
    /*DOC*/    "are fine for simple access, but will be considerably slow when\n"
    /*DOC*/    "doing of pixel work with them. If you plan on doing a lot of\n"
    /*DOC*/    "pixel level work, it is recommended to use the pygame.surfarray\n"
    /*DOC*/    "module, which can treat the surfaces like large multidimensional\n"
    /*DOC*/    "arrays (and it's quite quick). Some surfaces need to be locked\n"
    /*DOC*/    "before they can be used. Surfaces with flags like HWSURFACE and\n"
    /*DOC*/    "RLEACCEL generally require calls to lock() and unlock()\n"
    /*DOC*/    "surrounding pixel access. It is safe to lock() and unlock()\n"
    /*DOC*/    "surfaces that do not require locking. Nonetheless, you can check\n"
    /*DOC*/    "to see if a Surface really needs to be locked with the mustlock()\n"
    /*DOC*/    "function.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The packed pixel values are a single interger with the red,\n"
    /*DOC*/    "green, and blue components packed in to match the current\n"
    /*DOC*/    "bitdepth for the Surface. You generally don't need to care how\n"
    /*DOC*/    "this is done, the map_rgb() and unmap_rgb() will convert back and\n"
    /*DOC*/    "forth between the packed pixel values for you. Information on how\n"
    /*DOC*/    "the pixels are packed can be retreived from the get_masks(),\n"
    /*DOC*/    "get_losses() and get_shifts() routines.\n"
    /*DOC*/ ;
#if 0 /*extra help, only for docs*/
    /*DOC*/ static char doc_Surface_EXTRA[] =
    /*DOC*/    "Here is the quick breakdown of how packed pixels work (don't worry if\n"
    /*DOC*/    "you don't quite understand this, it is only here for informational\n"
    /*DOC*/    "purposes, it is not needed). Each colorplane mask can be used to\n"
    /*DOC*/    "isolate the values for a colorplane from the packed pixel color.\n"
    /*DOC*/    "Therefore PACKED_COLOR & RED_MASK == REDPLANE. Note that the\n"
    /*DOC*/    "REDPLANE is not exactly the red color value, but it is the red\n"
    /*DOC*/    "color value bitwise left shifted a certain amount. The losses and\n"
    /*DOC*/    "masks can be used to convert back and forth between each\n"
    /*DOC*/    "colorplane and the actual color for that plane. Here are the\n"
    /*DOC*/    "final formulas used be map and unmap (not exactly, heh).\n"
    /*DOC*/    "PACKED_COLOR = RED>>losses[0]<<shifts[0] |\n"
    /*DOC*/    "      GREEN>>losses[1]<<shifts[1] | BLUE>>losses[2]<<shifts[2]\n"
    /*DOC*/    "RED = PACKED_COLOR & masks[0] >> shifts[0] << losses[0]\n"
    /*DOC*/    "GREEN = PACKED_COLOR & masks[1] >> shifts[1] << losses[1]\n"
    /*DOC*/    "BLUE = PACKED_COLOR & masks[2] >> shifts[2] << losses[2]\n"
    /*DOC*/    "There is also an alpha channel for some Surfaces. The alpha\n"
    /*DOC*/    "channel works this same exact way, and the map_rgba() and\n"
    /*DOC*/    "unmap_rgba() functions can be used to do the conversion for you.\n"
#endif


static PyTypeObject PySurface_Type =
{
	PyObject_HEAD_INIT(NULL)
	0,						/*size*/
	"Surface",				/*name*/
	sizeof(PySurfaceObject),/*basic size*/
	0,						/*itemsize*/
	surface_dealloc,		/*dealloc*/
	0,						/*print*/
	surface_getattr,		/*getattr*/
	NULL,					/*setattr*/
	NULL,					/*compare*/
	surface_str,			/*repr*/
	NULL,					/*as_number*/
	NULL,					/*as_sequence*/
	NULL,					/*as_mapping*/
	(hashfunc)NULL,			/*hash*/
	(ternaryfunc)NULL,		/*call*/
	(reprfunc)NULL,			/*str*/
	0L,0L,0L,0L,
	doc_Surface_MODULE /* Documentation string */
};


static PyObject* PySurface_New(SDL_Surface* s)
{
	PySurfaceObject* surf;

	if(!s) return RAISE(PyExc_SDLError, SDL_GetError());

	surf = PyObject_NEW(PySurfaceObject, &PySurface_Type);
	if(surf)
		surf->surf = s;

	return (PyObject*)surf;
}



/* surface module functions */

    /*DOC*/ static char doc_new_surface[] =
    /*DOC*/    "pygame.new_surface(size, [flags, [depth|Surface, [masks]]]) ->\n"
    /*DOC*/    "Surface\n"
    /*DOC*/    "create a new Surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Creates a new surface object. Size is a 2-int-sequence containing\n"
    /*DOC*/    "width and height. Depth is the number of bits used per pixel. If\n"
    /*DOC*/    "omitted, depth will use the current display depth. Masks is a\n"
    /*DOC*/    "four item sequence containing the bitmask for r,g,b, and a. If\n"
    /*DOC*/    "omitted, masks will default to the usual values for the given\n"
    /*DOC*/    "bitdepth. Flags is a mix of the following flags: SWSURFACE,\n"
    /*DOC*/    "HWSURFACE, ASYNCBLIT, SRCCOLORKEY, or SRCALPHA. (flags = 0 is the\n"
    /*DOC*/    "same as SWSURFACE). depth and masks can be substituted for\n"
    /*DOC*/    "another surface object which will create the new surface with the\n"
    /*DOC*/    "same format as the given one. When using default masks, alpha\n"
    /*DOC*/    "will always be ignored. Note, if you pass SRCOLORKEY and/or\n"
    /*DOC*/    "SRCALPHA, the surface won't immediately have these features\n"
    /*DOC*/    "enabled. SDL will use these flags to help optimize the surface\n"
    /*DOC*/    "for use with the blitters. Also, for a plain software surface, 0\n"
    /*DOC*/    "can be used for the flag. A plain hardware surface can just use 1\n"
    /*DOC*/    "for the flag.\n"
    /*DOC*/ ;

static PyObject* new_surface(PyObject* self, PyObject* arg)
{
	Uint32 flags = 0;
	int width, height;
	PyObject *depth=NULL, *masks=NULL;
	short bpp;
	Uint32 Rmask, Gmask, Bmask, Amask;
	SDL_Surface* surface;

	if(!PyArg_ParseTuple(arg, "(ii)|iOO", &width, &height, &flags, &depth, &masks))
		return NULL;

	VIDEO_INIT_CHECK();
		
	if(depth && masks) /*all info supplied, most errorchecking needed*/
	{
		if(PySurface_Check(depth))
			return RAISE(PyExc_ValueError, "cannot pass surface for depth and color masks");
		if(!ShortFromObj(depth, &bpp))
			return RAISE(PyExc_ValueError, "invalid bits per pixel depth argument");
		if(!PySequence_Check(masks) || PySequence_Length(masks)!=4)
			return RAISE(PyExc_ValueError, "masks argument must be sequence of four numbers");
		if(!UintFromObjIndex(masks, 0, &Rmask) || !UintFromObjIndex(masks, 1, &Gmask) ||
					!UintFromObjIndex(masks, 2, &Bmask) || !UintFromObjIndex(masks, 3, &Amask))
			return RAISE(PyExc_ValueError, "invalid mask values in masks sequence");
	}
	else if(depth && PyNumber_Check(depth))/*use default masks*/
	{
		if(!ShortFromObj(depth, &bpp))
			return RAISE(PyExc_ValueError, "invalid bits per pixel depth argument");
		Amask = 0;
		switch(bpp)
		{
		case 8:
			Rmask = 0xFF >> 6 << 5; Gmask = 0xFF >> 5 << 2; Bmask = 0xFF >> 6; break;
		case 12:
			Rmask = 0xFF >> 4 << 8; Gmask = 0xFF >> 4 << 4; Bmask = 0xFF >> 4; break;
		case 15:
			Rmask = 0xFF >> 3 << 10; Gmask = 0xFF >> 3 << 5; Bmask = 0xFF >> 3; break;
		case 16:
			Rmask = 0xFF >> 3 << 11; Gmask = 0xFF >> 2 << 5; Bmask = 0xFF >> 3; break;
		case 24:
		case 32:
			Rmask = 0xFF << 16; Gmask = 0xFF << 8; Bmask = 0xFF; break;
		default:
			return RAISE(PyExc_ValueError, "no standard masks exist for given bitdepth");
		}
	}
	else /*no depth or surface*/
	{
		SDL_PixelFormat* pix;
		if(depth && PySurface_Check(depth))
			pix = ((PySurfaceObject*)depth)->surf->format;
		else if(SDL_GetVideoSurface())
			pix = SDL_GetVideoSurface()->format;
		else
			pix = SDL_GetVideoInfo()->vfmt;
		bpp = pix->BitsPerPixel;
		Rmask = pix->Rmask;
		Gmask = pix->Gmask;
		Bmask = pix->Bmask;
		Amask = pix->Amask;
	}

	surface = SDL_CreateRGBSurface(flags, width, height, bpp, Rmask, Gmask, Bmask, Amask);
	return PySurface_New(surface);
}




static PyMethodDef surface_builtins[] =
{
	{ "new_surface", new_surface, 1, doc_new_surface },
	{ NULL, NULL }
};



    /*DOC*/ static char doc_pygame_surface_MODULE[] =
    /*DOC*/    "The surface module doesn't have much in the line of functions. It\n"
    /*DOC*/    "does have the Surface object, and one routine to create new\n"
    /*DOC*/    "surfaces, pygame.surface().\n"
    /*DOC*/ ;

void initsurface()
{
	PyObject *module, *dict, *apiobj;
	static void* c_api[PYGAMEAPI_SURFACE_NUMSLOTS];

	PyType_Init(PySurface_Type);


    /* create the module */
	module = Py_InitModule3("surface", surface_builtins, doc_pygame_surface_MODULE);
	dict = PyModule_GetDict(module);

	/* export the c api */
	c_api[0] = &PySurface_Type;
	c_api[1] = PySurface_New;
	apiobj = PyCObject_FromVoidPtr(c_api, NULL);
	PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);

	/*imported needed apis*/
	import_pygame_base();
	import_pygame_rect();
}

