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
 *  pygame Surface module
 */
#define PYGAMEAPI_SURFACE_INTERNAL
#include "pygame.h"


staticforward PyTypeObject PySurface_Type;
static PyObject* PySurface_New(SDL_Surface* info);
#define PySurface_Check(x) ((x)->ob_type == &PySurface_Type)
extern int pygame_AlphaBlit(SDL_Surface *src, SDL_Rect *srcrect,
                        SDL_Surface *dst, SDL_Rect *dstrect);

#if PYTHON_API_VERSION >= 1011 /*this is the python-2.2 constructor*/
static PyObject* surface_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
#endif



/* surface object methods */


    /*DOC*/ static char doc_surf_get_at[] =
    /*DOC*/    "Surface.get_at(position) -> RGBA\n"
    /*DOC*/    "get a pixel color\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the RGB color values at a given pixel. If the\n"
    /*DOC*/    "Surface has no per-pixel alpha, the alpha will be 255 (opaque).\n"
    /*DOC*/    "A pixel outside the surface area will raise an IndexError.\n"
    /*DOC*/    "\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function will need to temporarily lock the surface.\n"
    /*DOC*/ ;

static PyObject* surf_get_at(PyObject* self, PyObject* arg)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	SDL_PixelFormat* format = surf->format;
	Uint8* pixels = (Uint8*)surf->pixels;
	int x, y;
	Uint32 color;
	Uint8* pix;
	Uint8 r, g, b, a;

	if(!PyArg_ParseTuple(arg, "(ii)", &x, &y))
		return NULL;

	if(surf->flags & SDL_OPENGL)
		return RAISE(PyExc_SDLError, "Cannot call on OPENGL Surfaces");

	if(x < 0 || x >= surf->w || y < 0 || y >= surf->h)
		return RAISE(PyExc_IndexError, "pixel index out of range");

	if(format->BytesPerPixel < 1 || format->BytesPerPixel > 4)
		return RAISE(PyExc_RuntimeError, "invalid color depth for surface");

	if(!PySurface_Lock(self)) return NULL;
	pixels = (Uint8*)surf->pixels;

	switch(format->BytesPerPixel)
	{
		case 1:
			color = (Uint32)*((Uint8*)pixels + y * surf->pitch + x);
			break;
		case 2:
			color = (Uint32)*((Uint16*)(pixels + y * surf->pitch) + x);
			break;
		case 3:
			pix = ((Uint8*)(pixels + y * surf->pitch) + x * 3);
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
			color = (pix[0]) + (pix[1]<<8) + (pix[2]<<16);
#else
			color = (pix[2]) + (pix[1]<<8) + (pix[0]<<16);
#endif
			break;
		default: /*case 4:*/
			color = *((Uint32*)(pixels + y * surf->pitch) + x);
			break;
	}
	if(!PySurface_Unlock(self)) return NULL;

	SDL_GetRGBA(color, format, &r, &g, &b, &a);
	return Py_BuildValue("(bbbb)", r, g, b, a);
}



    /*DOC*/ static char doc_surf_set_at[] =
    /*DOC*/    "Surface.set_at(position, RGBA) -> None\n"
    /*DOC*/    "set pixel at given position\n"
    /*DOC*/    "\n"
    /*DOC*/    "Assigns color to the image at the give position. Color can be a\n"
    /*DOC*/    "RGBA sequence or a mapped color integer. Setting the pixel outside\n"
    /*DOC*/    "the clip area or surface area will have no effect.\n"
    /*DOC*/    "\n"
    /*DOC*/    "In some situations just using the fill() function with a one-pixel\n"
    /*DOC*/    "sized rectangle will be quicker. Also the fill function does not\n"
    /*DOC*/    "require the surface to be locked.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function will need to temporarily lock the surface.\n"
    /*DOC*/ ;

static PyObject* surf_set_at(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	SDL_PixelFormat* format = surf->format;
	Uint8* pixels;
	int x, y;
	Uint32 color;
	Uint8 rgba[4];
	PyObject* rgba_obj;
	Uint8* byte_buf;

	if(!PyArg_ParseTuple(args, "(ii)O", &x, &y, &rgba_obj))
		return NULL;

	if(surf->flags & SDL_OPENGL)
		return RAISE(PyExc_SDLError, "Cannot call on OPENGL Surfaces");

	if(format->BytesPerPixel < 1 || format->BytesPerPixel > 4)
		return RAISE(PyExc_RuntimeError, "invalid color depth for surface");

	if(x < surf->clip_rect.x || x >= surf->clip_rect.x + surf->clip_rect.w ||
                    y < surf->clip_rect.y || y >= surf->clip_rect.y + surf->clip_rect.h)
	{
		/*out of clip area*/
                RETURN_NONE
	}

	if(PyInt_Check(rgba_obj))
		color = (Uint32)PyInt_AsLong(rgba_obj);
	else if(RGBAFromObj(rgba_obj, rgba))
		color = SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
	else
		return RAISE(PyExc_TypeError, "invalid color argument");

	if(!PySurface_Lock(self)) return NULL;
	pixels = (Uint8*)surf->pixels;

	switch(format->BytesPerPixel)
	{
		case 1:
			*((Uint8*)pixels + y * surf->pitch + x) = (Uint8)color;
			break;
		case 2:
			*((Uint16*)(pixels + y * surf->pitch) + x) = (Uint16)color;
			break;
		case 3:
			byte_buf = (Uint8*)(pixels + y * surf->pitch) + x * 3;
			*(byte_buf + (format->Rshift >> 3)) = rgba[0];
			*(byte_buf + (format->Gshift >> 3)) = rgba[1];
			*(byte_buf + (format->Bshift >> 3)) = rgba[2];
			break;
		default: /*case 4:*/
			*((Uint32*)(pixels + y * surf->pitch) + x) = color;
			break;
	}

	if(!PySurface_Unlock(self)) return NULL;
	RETURN_NONE
}



    /*DOC*/ static char doc_surf_map_rgb[] =
    /*DOC*/    "Surface.map_rgb(RGBA) -> int\n"
    /*DOC*/    "convert RGB into a mapped color\n"
    /*DOC*/    "\n"
    /*DOC*/    "Uses the Surface format to convert RGBA into a mapped color value.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function is not as needed as normal C code using SDL. The pygame\n"
    /*DOC*/    "functions do not used mapped colors, so there is no need to map them.\n"
   /*DOC*/ ;

static PyObject* surf_map_rgb(PyObject* self,PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	Uint8 rgba[4];
	int color;

	if(!RGBAFromObj(args, rgba))
		return RAISE(PyExc_TypeError, "Invalid RGBA argument");

	color = SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
	return PyInt_FromLong(color);
}



    /*DOC*/ static char doc_surf_unmap_rgb[] =
    /*DOC*/    "Surface.unmap_rgb(color) -> RGBA\n"
    /*DOC*/    "convert mapped color into RGB\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function returns the RGBA components for a mapped color\n"
    /*DOC*/    "value. If Surface has no per-pixel alpha, alpha will be 255 (opaque).\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function is not as needed as normal C code using SDL. The pygame\n"
    /*DOC*/    "functions do not used mapped colors, so there is no need to unmap them.\n"
    /*DOC*/ ;

static PyObject* surf_unmap_rgb(PyObject* self,PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	Uint32 col;
	Uint8 r, g, b, a;

	if(!PyArg_ParseTuple(args, "i", &col))
		return NULL;

	SDL_GetRGBA(col,surf->format, &r, &g, &b, &a);

	return Py_BuildValue("(bbbb)", r, g, b, a);
}



    /*DOC*/ static char doc_surf_lock[] =
    /*DOC*/    "Surface.lock() -> None\n"
    /*DOC*/    "locks Surface for pixel access\n"
    /*DOC*/    "\n"
    /*DOC*/    "On accelerated surfaces, it is usually required to lock the\n"
    /*DOC*/    "surface before you can access the pixel values. To be safe, it is\n"
    /*DOC*/    "always a good idea to lock the surface before entering a block of\n"
    /*DOC*/    "code that changes or accesses the pixel values. The surface must\n"
    /*DOC*/    "not be locked when performing other pygame functions on it like\n"
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
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	if(!PySurface_Lock(self))
		return NULL;

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
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	if(!PySurface_Unlock(self))
		return NULL;

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
	return PyInt_FromLong(SDL_MUSTLOCK(surf) || ((PySurfaceObject*)self)->subsurface);
}


    /*DOC*/ static char doc_surf_get_locked[] =
    /*DOC*/    "Surface.get_locked() -> bool\n"
    /*DOC*/    "check if the surface needs locking\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns true if the surface is currently locked.\n"
    /*DOC*/ ;

static PyObject* surf_get_locked(PyObject* self, PyObject* args)
{
	PySurfaceObject* surf = (PySurfaceObject*)self;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong(surf->lockcount);
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
		return RAISE(PyExc_SDLError, "Surface has no palette to get\n");

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
		return RAISE(PyExc_SDLError, "Surface has no palette to set\n");
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
    /*DOC*/    "\n"
    /*DOC*/    "You can pass an incomplete list of RGB values, and\n"
    /*DOC*/    "this will only change the first colors in the palette.\n"
    /*DOC*/ ;

static PyObject* surf_set_palette(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	SDL_Palette* pal = surf->format->palette;
	SDL_Color* colors;
	PyObject* list, *item;
	int i, len;
	int r, g, b;

	if(!PyArg_ParseTuple(args, "O", &list))
		return NULL;
	if(!PySequence_Check(list))
		return RAISE(PyExc_ValueError, "Argument must be a sequence type");

	if(!pal)
		return RAISE(PyExc_SDLError, "Surface has no palette\n");

	if(!SDL_WasInit(SDL_INIT_VIDEO))
		return RAISE(PyExc_SDLError, "cannot set palette without pygame.display initialized");

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
		if(!IntFromObjIndex(item, 0, &r) || !IntFromObjIndex(item, 1, &g) || !IntFromObjIndex(item, 2, &b))
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

	if(index >= pal->ncolors || index < 0)
	{
		PyErr_SetString(PyExc_IndexError, "index out of bounds");
		return NULL;
	}

	if(!SDL_WasInit(SDL_INIT_VIDEO))
		return RAISE(PyExc_SDLError, "cannot set palette without pygame.display initialized");

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
    /*DOC*/    "as the color argument. If no arguments or None is passed,\n"
    /*DOC*/    "colorkeying will be disabled for this surface.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The color argument can be either a RGBA sequence or a mapped integer.\n"
    /*DOC*/    "\n"
    /*DOC*/    "If your image is nonchanging and will be used repeatedly, you\n"
    /*DOC*/    "will probably want to pass the RLEACCEL flag to the call. This\n"
    /*DOC*/    "will take a short time to compile your surface, and increase the\n"
    /*DOC*/    "blitting speed.\n"
    /*DOC*/ ;

static PyObject* surf_set_colorkey(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	Uint32 flags = 0, color = 0;
	PyObject* rgba_obj = NULL, *intobj = NULL;
	Uint8 rgba[4];
	int result, hascolor=0;

	if(!PyArg_ParseTuple(args, "|Oi", &rgba_obj, &flags))
		return NULL;

	if(surf->flags & SDL_OPENGL)
		return RAISE(PyExc_SDLError, "Cannot call on OPENGL Surfaces");

	if(rgba_obj && rgba_obj!=Py_None)
	{
		if(PyNumber_Check(rgba_obj) && (intobj=PyNumber_Int(rgba_obj)))
		{
			color = (Uint32)PyInt_AsLong(intobj);
			Py_DECREF(intobj);
		}
		else if(RGBAFromObj(rgba_obj, rgba))
			color = SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
		else
			return RAISE(PyExc_TypeError, "invalid color argument");
		hascolor = 1;
	}
	if(hascolor)
		flags |= SDL_SRCCOLORKEY;

	PySurface_Prep(self);
	result = SDL_SetColorKey(surf, flags, color);
	PySurface_Unprep(self);

	if(result == -1)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE
}


    /*DOC*/ static char doc_surf_get_colorkey[] =
    /*DOC*/    "Surface.get_colorkey() -> RGBA\n"
    /*DOC*/    "query colorkey\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the current mapped color value being used for\n"
    /*DOC*/    "colorkeying. If colorkeying is not enabled for this surface, it\n"
    /*DOC*/    "returns None\n"
    /*DOC*/ ;

static PyObject* surf_get_colorkey(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	Uint8 r, g, b, a;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	if(surf->flags & SDL_OPENGL)
		return RAISE(PyExc_SDLError, "Cannot call on OPENGL Surfaces");

	if(!(surf->flags&SDL_SRCCOLORKEY))
		RETURN_NONE

	SDL_GetRGBA(surf->format->colorkey, surf->format, &r, &g, &b, &a);
	return Py_BuildValue("(bbbb)", r, g, b, a);
}


    /*DOC*/ static char doc_surf_set_alpha[] =
    /*DOC*/    "Surface.set_alpha([alpha, [flags]]) -> None\n"
    /*DOC*/    "change alpha information\n"
    /*DOC*/    "\n"
    /*DOC*/    "Set the overall transparency for the surface. If no alpha is\n"
    /*DOC*/    "passed, alpha blending is disabled for the surface. An alpha of 0\n"
    /*DOC*/    "is fully transparent, an alpha of 255 is fully opaque. If no\n"
    /*DOC*/    "arguments or None is passed, this will disable the surface alpha.\n"
    /*DOC*/    "\n"
    /*DOC*/    "If your surface has a pixel alpha channel, it will override the\n"
    /*DOC*/    "overall surface transparency. You'll need to change the actual\n"
    /*DOC*/    "pixel transparency to make changes.\n"
    /*DOC*/    "\n"
    /*DOC*/    "If your image also has pixel alpha values, will be used repeatedly, you\n"
    /*DOC*/    "will probably want to pass the RLEACCEL flag to the call. This\n"
    /*DOC*/    "will take a short time to compile your surface, and increase the\n"
    /*DOC*/    "blitting speed.\n"
    /*DOC*/ ;

static PyObject* surf_set_alpha(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	Uint32 flags = 0;
	PyObject* alpha_obj = NULL, *intobj=NULL;
	Uint8 alpha;
	int result, alphaval=255, hasalpha=0;

	if(!PyArg_ParseTuple(args, "|Oi", &alpha_obj, &flags))
		return NULL;

	if(surf->flags & SDL_OPENGL)
		return RAISE(PyExc_SDLError, "Cannot call on OPENGL Surfaces");

	if(alpha_obj && alpha_obj!=Py_None)
	{
		if(PyNumber_Check(alpha_obj) && (intobj=PyNumber_Int(alpha_obj)))
		{
			alphaval = (int)PyInt_AsLong(intobj);
			Py_DECREF(intobj);
		}
		else
			return RAISE(PyExc_TypeError, "invalid alpha argument");
		hasalpha = 1;
	}
	if(hasalpha)
		flags |= SDL_SRCALPHA;

	if(alphaval>255) alpha = 255;
	else if(alphaval<0) alpha = 0;
	else alpha = (Uint8)alphaval;

	PySurface_Prep(self);
	result = SDL_SetAlpha(surf, flags, alpha);
	PySurface_Unprep(self);

	if(result == -1)
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

	if(surf->flags & SDL_OPENGL)
		return RAISE(PyExc_SDLError, "Cannot call on OPENGL Surfaces");

	if(surf->flags&SDL_SRCALPHA)
		return PyInt_FromLong(surf->format->alpha);

	RETURN_NONE
}


    /*DOC*/ static char doc_surf_convert[] =
    /*DOC*/    "Surface.convert([src_surface] OR depth, [flags] OR masks) -> Surface\n"
    /*DOC*/    "new copy of surface with different format\n"
    /*DOC*/    "\n"
    /*DOC*/    "Creates a new copy of the surface with the desired pixel format.\n"
    /*DOC*/    "Surfaces with the same pixel format will blit much faster than\n"
    /*DOC*/    "those with mixed formats. The pixel format of the new surface\n"
    /*DOC*/    "will match the format given as the argument. If no surface is\n"
    /*DOC*/    "given, the new surface will have the same pixel format as the\n"
    /*DOC*/    "current display.\n"
    /*DOC*/    "\n"
    /*DOC*/    "convert() will also accept bitsize or mask arguments like the\n"
    /*DOC*/    "Surface() constructor function. Either pass an integer bitsize\n"
    /*DOC*/    "or a sequence of color masks to specify the format of surface\n"
    /*DOC*/    "you would like to convert to. When used this way you may also\n"
    /*DOC*/    "pass an optional flags argument (whew).\n"
    /*DOC*/ ;

static PyObject* surf_convert(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	PyObject* final;
	PyObject* argobject=NULL;
	SDL_Surface* src;
	SDL_Surface* newsurf;
	Uint32 flags=-1;

	if(!SDL_WasInit(SDL_INIT_VIDEO))
		return RAISE(PyExc_SDLError, "cannot convert without pygame.display initialized");

	if(!PyArg_ParseTuple(args, "|Oi", &argobject, &flags))
		return NULL;

        if(surf->flags & SDL_OPENGL)
        {
            return RAISE(PyExc_SDLError, "Cannot convert opengl display");
        }

	PySurface_Prep(self);
	if(argobject)
	{
		if(PySurface_Check(argobject))
		{
			src = PySurface_AsSurface(argobject);
			flags = src->flags | (surf->flags & (SDL_SRCCOLORKEY|SDL_SRCALPHA));
			newsurf = SDL_ConvertSurface(surf, src->format, flags);
		}
		else
		{
			int bpp;
			SDL_PixelFormat format;
			memcpy(&format, surf->format, sizeof(format));
			if(IntFromObj(argobject, &bpp))
			{
				int Rmask, Gmask, Bmask, Amask;
				if(flags!=-1 && flags&SDL_SRCALPHA)
				{
					switch(bpp)
					{
					case 16:
						Rmask = 0xF<<8; Gmask = 0xF<<4; Bmask = 0xF; Amask = 0xF<<12; break;
					case 32:
						Rmask = 0xFF<<16; Gmask = 0xFF<<8; Bmask = 0xFF; Amask = 0xFF<<24; break;
					default:
						return RAISE(PyExc_ValueError, "no standard masks exist for given bitdepth with alpha");
					}
				}
				else
				{
					Amask = 0;
					switch(bpp)
					{
					case 8:
						Rmask = 0xFF>>6<<5; Gmask = 0xFF>>5<<2; Bmask = 0xFF>>6; break;
					case 12:
						Rmask = 0xFF>>4<<8; Gmask = 0xFF>>4<<4; Bmask = 0xFF>>4; break;
					case 15:
						Rmask = 0xFF>>3<<10; Gmask = 0xFF>>3<<5; Bmask = 0xFF>>3; break;
					case 16:
						Rmask = 0xFF>>3<<11; Gmask = 0xFF>>2<<5; Bmask = 0xFF>>3; break;
					case 24:
					case 32:
						Rmask = 0xFF << 16; Gmask = 0xFF << 8; Bmask = 0xFF; break;
					default:
						return RAISE(PyExc_ValueError, "nonstandard bit depth given");
					}
				}
				format.Rmask = Rmask; format.Gmask = Gmask;
				format.Bmask = Bmask; format.Amask = Amask;
			}
			else if(PySequence_Check(argobject) && PySequence_Size(argobject)==4)
			{
				Uint32 mask;
				if(!UintFromObjIndex(argobject, 0, &format.Rmask) ||
							!UintFromObjIndex(argobject, 1, &format.Gmask) ||
							!UintFromObjIndex(argobject, 2, &format.Bmask) ||
							!UintFromObjIndex(argobject, 3, &format.Amask))
				{
					PySurface_Unprep(self);
					return RAISE(PyExc_ValueError, "invalid color masks given");
				}
				mask = format.Rmask|format.Gmask|format.Bmask|format.Amask;
				for(bpp=0; bpp<32; ++bpp)
					if(!(mask>>bpp)) break;
			}
			else
			{
				PySurface_Unprep(self);
				return RAISE(PyExc_ValueError, "invalid argument specifying new format to convert to");
			}
			format.BitsPerPixel = (Uint8)bpp;
			format.BytesPerPixel = (bpp+7)/8;
			if(flags == -1)
				flags = surf->flags;
			if(format.Amask)
				flags |= SDL_SRCALPHA;
			newsurf = SDL_ConvertSurface(surf, &format, flags);
		}
	}
	else
	{
		if(SDL_WasInit(SDL_INIT_VIDEO))
			newsurf = SDL_DisplayFormat(surf);
		else
			newsurf = SDL_ConvertSurface(surf, surf->format, surf->flags);
	}
	PySurface_Unprep(self);

	final = PySurface_New(newsurf);
	if(!final)
		SDL_FreeSurface(newsurf);
	return final;
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
	PyObject* final;
	PySurfaceObject* srcsurf = NULL;
	SDL_Surface* newsurf, *src;

	if(!SDL_WasInit(SDL_INIT_VIDEO))
		return RAISE(PyExc_SDLError, "cannot convert without pygame.display initialized");

	if(!PyArg_ParseTuple(args, "|O!", &PySurface_Type, &srcsurf))
		return NULL;

	PySurface_Prep(self);
	if(srcsurf)
	{
		/*hmm, we have to figure this out, not all depths have good support for alpha*/
		src = PySurface_AsSurface(srcsurf);
		newsurf = SDL_DisplayFormatAlpha(surf);
	}
	else
		newsurf = SDL_DisplayFormatAlpha(surf);
	PySurface_Unprep(self);

	final = PySurface_New(newsurf);
	if(!final)
		SDL_FreeSurface(newsurf);
	return final;
}


    /*DOC*/ static char doc_surf_set_clip[] =
    /*DOC*/    "Surface.set_clip([rectstyle]) -> None\n"
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
	PyObject* item;
	GAME_Rect *rect=NULL, temp;
        SDL_Rect sdlrect;
	int result;

	if(PyTuple_Size(args))
	{
		item = PyTuple_GET_ITEM(args, 0);
		if(!(item == Py_None && PyTuple_Size(args) == 1))
		{
		    rect = GameRect_FromObject(args, &temp);
		    if(!rect)
			    return RAISE(PyExc_ValueError, "invalid rectstyle object");
		}
                sdlrect.x = rect->x;
                sdlrect.y = rect->y;
                sdlrect.h = rect->h;
                sdlrect.w = rect->w;
                result = SDL_SetClipRect(surf, &sdlrect);
	}
        else
                result = SDL_SetClipRect(surf, NULL);

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
	return PyRect_New(&surf->clip_rect);
}



    /*DOC*/ static char doc_surf_fill[] =
    /*DOC*/    "Surface.fill(color, [rectstyle])) -> Rect\n"
    /*DOC*/    "fill areas of a Surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Fills the specified area of the Surface with the mapped color\n"
    /*DOC*/    "value. If no destination rectangle is supplied, it will fill the\n"
    /*DOC*/    "entire Surface.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The color argument can be a RGBA sequence or a mapped color integer.\n"
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
	PyObject* rgba_obj;
	Uint8 rgba[4];
        SDL_Rect sdlrect;

	if(!PyArg_ParseTuple(args, "O|O", &rgba_obj, &r))
		return NULL;

	if(surf->flags & SDL_OPENGL)
		return RAISE(PyExc_SDLError, "Cannot call on OPENGL Surfaces");

	if(PyInt_Check(rgba_obj))
		color = (Uint32)PyInt_AsLong(rgba_obj);
	else if(RGBAFromObj(rgba_obj, rgba))
		color = SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
	else
		return RAISE(PyExc_TypeError, "invalid color argument");

	if(!r)
	{
		rect = &temp;
		temp.x = temp.y = 0;
		temp.w = surf->w;
		temp.h = surf->h;
	}
	else if(!(rect = GameRect_FromObject(r, &temp)))
		return RAISE(PyExc_ValueError, "invalid rectstyle object");

	/*we need a fresh copy so our Rect values don't get munged*/
	if(rect != &temp)
	{
		memcpy(&temp, rect, sizeof(temp));
		rect = &temp;
	}

        sdlrect.x = rect->x;
        sdlrect.y = rect->y;
        sdlrect.w = rect->w;
        sdlrect.h = rect->h;

	PySurface_Prep(self);
	result = SDL_FillRect(surf, &sdlrect, color);
	PySurface_Unprep(self);

	if(result == -1)
		return RAISE(PyExc_SDLError, SDL_GetError());
	return PyRect_New(&sdlrect);
}


/*this internal blit function is accessable through the C api*/
int PySurface_Blit(PyObject *dstobj, PyObject *srcobj, SDL_Rect *dstrect, SDL_Rect *srcrect)
{
    SDL_Surface *src = PySurface_AsSurface(srcobj);
    SDL_Surface *dst = PySurface_AsSurface(dstobj);
    SDL_Surface *subsurface = NULL;
    int result, suboffsetx=0, suboffsety=0;
    SDL_Rect orig_clip, sub_clip;
    int didconvert = 0;

    /*passthrough blits to the real surface*/
    if(((PySurfaceObject*)dstobj)->subsurface)
    {
	    PyObject *owner;
	    struct SubSurface_Data *subdata;

	    subdata = ((PySurfaceObject*)dstobj)->subsurface;
	    owner = subdata->owner;
            subsurface = PySurface_AsSurface(owner);
	    suboffsetx = subdata->offsetx;
	    suboffsety = subdata->offsety;

	    while(((PySurfaceObject*)owner)->subsurface)
	    {
		subdata = ((PySurfaceObject*)owner)->subsurface;
    		owner = subdata->owner;
	        subsurface = PySurface_AsSurface(owner);
	    	suboffsetx += subdata->offsetx;
    	    	suboffsety += subdata->offsety;
	    }

	    SDL_GetClipRect(subsurface, &orig_clip);
	    SDL_GetClipRect(dst, &sub_clip);
	    sub_clip.x += suboffsetx;
	    sub_clip.y += suboffsety;
	    SDL_SetClipRect(subsurface, &sub_clip);
	    dstrect->x += suboffsetx;
	    dstrect->y += suboffsety;
	    dst = subsurface;
    }
    else
    {
	    PySurface_Prep(dstobj);
	    subsurface = NULL;
    }

    PySurface_Prep(srcobj);
/*    Py_BEGIN_ALLOW_THREADS */

    /*can't blit alpha to 8bit, crashes SDL*/
    if(dst->format->BytesPerPixel==1 && (src->format->Amask || src->flags&SDL_SRCALPHA))
    {
	    didconvert = 1;
	    src = SDL_DisplayFormat(src);
    }

    /*see if we should handle alpha ourselves*/
    if(dst->format->Amask && (dst->flags&SDL_SRCALPHA) &&
                !(src->format->Amask && !(src->flags&SDL_SRCALPHA)) && /*special case, SDL works*/
                (dst->format->BytesPerPixel == 2 || dst->format->BytesPerPixel==4))
    {
        result = pygame_AlphaBlit(src, srcrect, dst, dstrect);
    }
    else
    {
        result = SDL_BlitSurface(src, srcrect, dst, dstrect);
    }

    if(didconvert)
	    SDL_FreeSurface(src);

/*    Py_END_ALLOW_THREADS */
    if(subsurface)
    {
	    SDL_SetClipRect(subsurface, &orig_clip);
	    dstrect->x -= suboffsetx;
	    dstrect->y -= suboffsety;
    }
    else
	PySurface_Unprep(dstobj);
    PySurface_Unprep(srcobj);

    if(result == -1)
	    RAISE(PyExc_SDLError, SDL_GetError());
    if(result == -2)
	    RAISE(PyExc_SDLError, "Surface was lost");

    return result != 0;
}




    /*DOC*/ static char doc_surf_blit[] =
    /*DOC*/    "Surface.blit(source, destpos, [sourcerect]) -> Rect\n"
    /*DOC*/    "copy a one Surface to another.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The blitting will copy pixels from the source. It will\n"
    /*DOC*/    "respect any special modes like colorkeying and alpha. If hardware\n"
    /*DOC*/    "support is available, it will be used. The given source is the\n"
    /*DOC*/    "Surface to copy from. The destoffset is a 2-number-sequence that\n"
    /*DOC*/    "specifies where on the destination Surface the blit happens (see below).\n"
    /*DOC*/    "When sourcerect isn't supplied, the blit will copy the\n"
    /*DOC*/    "entire source surface. If you would like to copy only a portion\n"
    /*DOC*/    "of the source, use the sourcerect argument to control\n"
    /*DOC*/    "what area is copied.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The blit is subject to be clipped by the active clipping\n"
    /*DOC*/    "rectangle. The return value contains the actual area blitted.\n"
    /*DOC*/    "\n"
    /*DOC*/    "As a shortcut, the destination position can be passed as a\n"
    /*DOC*/    "rectangle. If a rectangle is given, the blit will use the topleft\n"
    /*DOC*/    "corner of the rectangle as the blit destination position. The\n"
    /*DOC*/    "rectangle sizes will be ignored.\n"
#if 0  /* "" */
    /*DOC*/    "\n"
    /*DOC*/    "Blitting surfaces with pixel alphas onto an 8bit destination will\n"
    /*DOC*/    "not use the surface alpha values.\n"
#endif /* "" */
    /*DOC*/ ;

static PyObject* surf_blit(PyObject* self, PyObject* args)
{
	SDL_Surface* src, *dest = PySurface_AsSurface(self);
	GAME_Rect* src_rect, temp;
	PyObject* srcobject, *argpos, *argrect = NULL;
	int dx, dy, result;
	SDL_Rect dest_rect, sdlsrc_rect;
	int sx, sy;

	if(!PyArg_ParseTuple(args, "O!O|O", &PySurface_Type, &srcobject, &argpos, &argrect))
		return NULL;
	src = PySurface_AsSurface(srcobject);

	if(dest->flags & SDL_OPENGL && !(dest->flags&(SDL_OPENGLBLIT&~SDL_OPENGL)))
		return RAISE(PyExc_SDLError, "Cannot blit to OPENGL Surfaces (OPENGLBLIT is ok)");

	if((src_rect = GameRect_FromObject(argpos, &temp)))
	{
		dx = src_rect->x;
		dy = src_rect->y;
	}
	else if(TwoIntsFromObj(argpos, &sx, &sy))
	{
		dx = sx;
		dy = sy;
	}
	else
		return RAISE(PyExc_TypeError, "invalid destination position for blit");

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
        sdlsrc_rect.x = (short)src_rect->x;
        sdlsrc_rect.y = (short)src_rect->y;
        sdlsrc_rect.w = (unsigned short)src_rect->w;
        sdlsrc_rect.h = (unsigned short)src_rect->h;

	result = PySurface_Blit(self, srcobject, &dest_rect, &sdlsrc_rect);
	if(result != 0)
	    return NULL;

	return PyRect_New(&dest_rect);
}


    /*DOC*/ static char doc_surf_get_flags[] =
    /*DOC*/    "Surface.get_flags() -> flags\n"
    /*DOC*/    "query the surface flags\n"
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
	return PyRect_New4(0, 0, surf->w, surf->h);
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


    /*DOC*/ static char doc_surf_subsurface[] =
    /*DOC*/    "Surface.subsurface(rectstyle) -> Surface\n"
    /*DOC*/    "create a new surface that shares pixel data\n"
    /*DOC*/    "\n"
    /*DOC*/    "Creates a new surface that shares pixel data of the given surface.\n"
    /*DOC*/    "Note that only the pixel data is shared. Things like clipping rectangles\n"
    /*DOC*/    "and colorkeys will be unique for the new surface.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The new subsurface will inherit the palette, colorkey, and surface alpha\n"
    /*DOC*/    "values from the base image.\n"
    /*DOC*/    "\n"
    /*DOC*/    "You should not use the RLEACCEL flag for parent surfaces of subsurfaces,\n"
    /*DOC*/    "for the most part it will work, but it will cause a lot of extra work,\n"
    /*DOC*/    "every time you change the subsurface, you must decode and recode the\n"
    /*DOC*/    "RLEACCEL data for the parent surface.\n"
    /*DOC*/    "\n"
    /*DOC*/    "As for using RLEACCEL with the subsurfaces, that will work as you'd\n"
    /*DOC*/    "expect, but changes the the parent Surface will not always take effect\n"
    /*DOC*/    "in the subsurface.\n"
    /*DOC*/ ;

static PyObject* surf_subsurface(PyObject* self, PyObject* args)
{
	SDL_Surface* surf = PySurface_AsSurface(self);
	SDL_PixelFormat* format = surf->format;
	GAME_Rect *rect, temp;
	SDL_Surface* sub;
	PyObject* subobj;
	int pixeloffset;
	char* startpixel;
	struct SubSurface_Data* data;

	if(surf->flags & SDL_OPENGL)
		return RAISE(PyExc_SDLError, "Cannot call on OPENGL Surfaces");

	if(!(rect = GameRect_FromObject(args, &temp)))
		return RAISE(PyExc_ValueError, "invalid rectstyle argument");
	if(rect->x < 0 || rect-> y < 0 || rect->x + rect->w > surf->w || rect->y + rect->h > surf->h)
		return RAISE(PyExc_ValueError, "subsurface rectangle outside surface area");


	PySurface_Lock(self);

	pixeloffset = rect->x * format->BytesPerPixel + rect->y * surf->pitch;
	startpixel = ((char*)surf->pixels) + pixeloffset;

	sub = SDL_CreateRGBSurfaceFrom(startpixel, rect->w, rect->h, format->BitsPerPixel, \
				surf->pitch, format->Rmask, format->Gmask, format->Bmask, format->Amask);

	PySurface_Unlock(self);

	if(!sub)
		return RAISE(PyExc_SDLError, SDL_GetError());

	/*copy the colormap if we need it*/
	if(surf->format->BytesPerPixel == 1 && surf->format->palette)
		SDL_SetPalette(sub, SDL_LOGPAL, surf->format->palette->colors, 0, surf->format->palette->ncolors);
	if(surf->flags & SDL_SRCALPHA)
		SDL_SetAlpha(sub, surf->flags&SDL_SRCALPHA, format->alpha);
	if(surf->flags & SDL_SRCCOLORKEY)
		SDL_SetColorKey(sub, surf->flags&(SDL_SRCCOLORKEY|SDL_RLEACCEL), format->colorkey);


	data = PyMem_New(struct SubSurface_Data, 1);
	if(!data) return NULL;

	subobj = PySurface_New(sub);
	if(!subobj)
	{
		PyMem_Del(data);
		return NULL;
	}
	Py_INCREF(self);
	data->owner = self;
	data->pixeloffset = pixeloffset;
	data->offsetx = rect->x;
	data->offsety = rect->y;
	((PySurfaceObject*)subobj)->subsurface = data;

	return subobj;
}



    /*DOC*/ static char doc_surf_get_offset[] =
    /*DOC*/    "Surface.get_offset() -> x, y\n"
    /*DOC*/    "get offset of subsurface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the X and Y position a subsurface is positioned\n"
    /*DOC*/    "inside its parent. Will return 0,0 for surfaces that are\n"
    /*DOC*/    "not a subsurface.\n"
    /*DOC*/ ;

static PyObject* surf_get_offset(PyObject* self, PyObject* args)
{
    	struct SubSurface_Data *subdata;
    	subdata = ((PySurfaceObject*)self)->subsurface;
	if(!subdata)
    	    	return Py_BuildValue("(ii)", 0, 0);
    	return Py_BuildValue("(ii)", subdata->offsetx, subdata->offsety);
}


    /*DOC*/ static char doc_surf_get_abs_offset[] =
    /*DOC*/    "Surface.get_abs_offset() -> x, y\n"
    /*DOC*/    "get absolute offset of subsurface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the absolute X and Y position a subsurface is positioned\n"
    /*DOC*/    "inside its top level parent. Will return 0,0 for surfaces that are\n"
    /*DOC*/    "not a subsurface.\n"
    /*DOC*/ ;

static PyObject* surf_get_abs_offset(PyObject* self, PyObject* args)
{
    	struct SubSurface_Data *subdata;
	PyObject *owner;
	int offsetx, offsety;

    	subdata = ((PySurfaceObject*)self)->subsurface;
	if(!subdata)
    	    	return Py_BuildValue("(ii)", 0, 0);

	subdata = ((PySurfaceObject*)self)->subsurface;
	owner = subdata->owner;
	offsetx = subdata->offsetx;
	offsety = subdata->offsety;

	while(((PySurfaceObject*)owner)->subsurface)
	{
	    subdata = ((PySurfaceObject*)owner)->subsurface;
    	    owner = subdata->owner;
	    offsetx += subdata->offsetx;
    	    offsety += subdata->offsety;
	}


	return Py_BuildValue("(ii)", offsetx, offsety);
}

    /*DOC*/ static char doc_surf_get_parent[] =
    /*DOC*/    "Surface.get_parent() -> Surface\n"
    /*DOC*/    "get a subsurface parent\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the Surface that is a parent of this subsurface.\n"
    /*DOC*/    "Will return None if this is not a subsurface.\n"
    /*DOC*/ ;

static PyObject* surf_get_parent(PyObject* self, PyObject* args)
{
    	struct SubSurface_Data *subdata;
    	subdata = ((PySurfaceObject*)self)->subsurface;
	if(!subdata)
    	    	RETURN_NONE

    	Py_INCREF(subdata->owner);
	return subdata->owner;
}

    /*DOC*/ static char doc_surf_get_abs_parent[] =
    /*DOC*/    "Surface.get_abs_parent() -> Surface\n"
    /*DOC*/    "get the toplevel surface for a subsurface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the top level Surface for this subsurface. If this is not\n"
    /*DOC*/    "a subsurface it will return a reference to itself. You will always\n"
    /*DOC*/    "get a valid surface from this method.\n"
    /*DOC*/ ;
static PyObject* surf_get_abs_parent(PyObject* self, PyObject* args)
{
    	struct SubSurface_Data *subdata;
	PyObject *owner;

    	subdata = ((PySurfaceObject*)self)->subsurface;
	if(!subdata)
	{
	    Py_INCREF(self);
	    return self;
	}

	subdata = ((PySurfaceObject*)self)->subsurface;
	owner = subdata->owner;

	while(((PySurfaceObject*)owner)->subsurface)
	{
	    subdata = ((PySurfaceObject*)owner)->subsurface;
    	    owner = subdata->owner;
	}

	Py_INCREF(owner);
	return owner;
}




static struct PyMethodDef surface_methods[] =
{
	{"get_at",			surf_get_at,		1, doc_surf_get_at },
	{"set_at",			surf_set_at,		1, doc_surf_set_at },

	{"map_rgb",			surf_map_rgb,		1, doc_surf_map_rgb },
	{"unmap_rgb",		surf_unmap_rgb, 	1, doc_surf_unmap_rgb },

	{"get_palette", 	surf_get_palette,	1, doc_surf_get_palette },
	{"get_palette_at",	surf_get_palette_at,1, doc_surf_get_palette_at },
	{"set_palette", 	surf_set_palette,	1, doc_surf_set_palette },
	{"set_palette_at",	surf_set_palette_at,1, doc_surf_set_palette_at },

	{"lock",			surf_lock,			1, doc_surf_lock },
	{"unlock",			surf_unlock,		1, doc_surf_unlock },
	{"mustlock",		surf_mustlock,		1, doc_surf_mustlock },
	{"get_locked",		surf_get_locked,	1, doc_surf_get_locked },

	{"set_colorkey",	surf_set_colorkey,	1, doc_surf_set_colorkey },
	{"get_colorkey",	surf_get_colorkey,	1, doc_surf_get_colorkey },
	{"set_alpha",		surf_set_alpha, 	1, doc_surf_set_alpha },
	{"get_alpha",		surf_get_alpha, 	1, doc_surf_get_alpha },

	{"convert",			surf_convert,		1, doc_surf_convert },
	{"convert_alpha",	surf_convert_alpha,	1, doc_surf_convert_alpha },

	{"set_clip",		surf_set_clip,		1, doc_surf_set_clip },
	{"get_clip",		surf_get_clip,		1, doc_surf_get_clip },

	{"fill",			surf_fill,			1, doc_surf_fill },
	{"blit",			surf_blit,			1, doc_surf_blit },

	{"get_flags",		surf_get_flags, 	1, doc_surf_get_flags },
	{"get_size",		surf_get_size,		1, doc_surf_get_size },
	{"get_width",		surf_get_width, 	1, doc_surf_get_width },
	{"get_height",		surf_get_height,	1, doc_surf_get_height },
	{"get_rect",		surf_get_rect,		1, doc_surf_get_rect },
	{"get_pitch",		surf_get_pitch, 	1, doc_surf_get_pitch },
	{"get_bitsize", 	surf_get_bitsize,	1, doc_surf_get_bitsize },
	{"get_bytesize",	surf_get_bytesize,	1, doc_surf_get_bytesize },
	{"get_masks",		surf_get_masks, 	1, doc_surf_get_masks },
	{"get_shifts",		surf_get_shifts,	1, doc_surf_get_shifts },
	{"get_losses",		surf_get_losses,	1, doc_surf_get_losses },

	{"subsurface",		surf_subsurface,	1, doc_surf_subsurface },
	{"get_offset",		surf_get_offset,	1, doc_surf_get_offset },
	{"get_abs_offset",	surf_get_abs_offset,	1, doc_surf_get_abs_offset },
	{"get_parent",		surf_get_parent,	1, doc_surf_get_parent },
	{"get_abs_parent",	surf_get_abs_parent,	1, doc_surf_get_abs_parent },

	{NULL,		NULL}
};



/* surface object internals */

static void surface_dealloc(PyObject* self)
{
	PySurfaceObject* surf = (PySurfaceObject*)self;
	struct SubSurface_Data* data = ((PySurfaceObject*)self)->subsurface;
	int flags=0;

	if(PySurface_AsSurface(surf))
	    flags = PySurface_AsSurface(surf)->flags;
	if(!(flags&SDL_HWSURFACE) || SDL_WasInit(SDL_INIT_VIDEO))
	{
	    	/*unsafe to free hardware surfaces without video init*/
		while(surf->lockcount > 0)
			PySurface_Unlock(self);
		SDL_FreeSurface(surf->surf);
	}
	if(data)
	{
		Py_XDECREF(data->owner);
		PyMem_Del(data);
	}

	PyObject_DEL(self);
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

	if(surf)
	{
	    type = (surf->flags&SDL_HWSURFACE)?"HW":"SW";
	    sprintf(str, "<Surface(%dx%dx%d %s)>", surf->w, surf->h, surf->format->BitsPerPixel, type);
	}
	else
	{
	    strcpy(str, "<Surface(Dead Display)>");
	}

	return PyString_FromString(str);
}

#if PYTHON_API_VERSION < 1011 /*PYTHON2.2*/
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
    /*DOC*/    "arrays (and it's quite quick).\n"
    /*DOC*/ ;
#endif
    /*DOC*/ static char doc_Surface[] =
    /*DOC*/    "pygame.Surface(size, [flags, [Surface|depth, [masks]]]) -> Surface\n"
    /*DOC*/    "create a new Surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Creates a new surface object. Size is a 2-int-sequence containing\n"
    /*DOC*/    "width and height. Depth is the number of bits used per pixel. If\n"
    /*DOC*/    "omitted, depth will use the current display depth. Masks is a\n"
    /*DOC*/    "four item sequence containing the bitmask for r,g,b, and a. If\n"
    /*DOC*/    "omitted, masks will default to the usual values for the given\n"
    /*DOC*/    "bitdepth. Flags is a mix of the following flags: SWSURFACE,\n"
    /*DOC*/    "HWSURFACE, ASYNCBLIT, or SRCALPHA. (flags = 0 is the\n"
    /*DOC*/    "same as SWSURFACE). Depth and masks can be substituted for\n"
    /*DOC*/    "another surface object which will create the new surface with the\n"
    /*DOC*/    "same format as the given one. When using default masks, alpha\n"
    /*DOC*/    "will always be ignored unless you pass SRCALPHA as a flag.\n"
    /*DOC*/    "For a plain software surface, 0 can be used for the flag. \n"
    /*DOC*/    "A plain hardware surface can just use 1 for the flag.\n"
    /*DOC*/ ;


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
	(hashfunc)NULL, 		/*hash*/
	(ternaryfunc)NULL,		/*call*/
	(reprfunc)NULL, 		/*str*/
	0L,0L,0L,
#if PYTHON_API_VERSION >= 1011 /*PYTHON2.2*/
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES | Py_TPFLAGS_BASETYPE, /* tp_flags */
	doc_Surface, /* Documentation string */
#else
	0,					/* tp_flags */
	doc_Surface_MODULE, /* Documentation string */
#endif
#if PYTHON_API_VERSION >= 1011 /*PYTHON2.2*/
	0,					/* tp_traverse */
	0,					/* tp_clear */
	0,					/* tp_richcompare */
	0,					/* tp_weaklistoffset */
	0,					/* tp_iter */
	0,					/* tp_iternext */
	0,					/* tp_methods */
	0,					/* tp_members */
	0,					/* tp_getset */
	0,					/* tp_base */
	0,					/* tp_dict */
	0,					/* tp_descr_get */
	0,					/* tp_descr_set */
	0,					/* tp_dictoffset */
	0,					/* tp_init */
	0,					/* tp_alloc */
	surface_new,		/* tp_new */
#endif
};


static PyObject* PySurface_New(SDL_Surface* s)
{
	PySurfaceObject* surf;
	if(!s) return RAISE(PyExc_SDLError, SDL_GetError());

	surf = PyObject_NEW(PySurfaceObject, &PySurface_Type);
	if(surf)
	{
		surf->surf = s;
		surf->subsurface = NULL;
		surf->didlock = 0;
		surf->lockcount = 0;
	}
	return (PyObject*)surf;
}



/* surface module functions */

static PyObject* Surface(PyObject* self, PyObject* arg)
{
	Uint32 flags = 0;
	int width, height;
	PyObject *depth=NULL, *masks=NULL, *final;
	int bpp;
	Uint32 Rmask, Gmask, Bmask, Amask;
	SDL_Surface* surface;
	SDL_PixelFormat default_format;

	if(!PyArg_ParseTuple(arg, "(ii)|iOO", &width, &height, &flags, &depth, &masks))
		return NULL;
	if(depth && masks) /*all info supplied, most errorchecking needed*/
	{
		if(PySurface_Check(depth))
			return RAISE(PyExc_ValueError, "cannot pass surface for depth and color masks");
		if(!IntFromObj(depth, &bpp))
			return RAISE(PyExc_ValueError, "invalid bits per pixel depth argument");
		if(!PySequence_Check(masks) || PySequence_Length(masks)!=4)
			return RAISE(PyExc_ValueError, "masks argument must be sequence of four numbers");
		if(!UintFromObjIndex(masks, 0, &Rmask) || !UintFromObjIndex(masks, 1, &Gmask) ||
					!UintFromObjIndex(masks, 2, &Bmask) || !UintFromObjIndex(masks, 3, &Amask))
			return RAISE(PyExc_ValueError, "invalid mask values in masks sequence");
	}
	else if(depth && PyNumber_Check(depth))/*use default masks*/
	{
		if(!IntFromObj(depth, &bpp))
			return RAISE(PyExc_ValueError, "invalid bits per pixel depth argument");
		if(flags & SDL_SRCALPHA)
		{
			switch(bpp)
			{
			case 16:
				Rmask = 0xF<<8; Gmask = 0xF<<4; Bmask = 0xF; Amask = 0xF<<12; break;
			case 32:
				Rmask = 0xFF<<16; Gmask = 0xFF<<8; Bmask = 0xFF; Amask = 0xFF<<24; break;
			default:
				return RAISE(PyExc_ValueError, "no standard masks exist for given bitdepth with alpha");
			}
		}
		else
		{
			Amask = 0;
			switch(bpp)
			{
			case 8:
				Rmask = 0xFF>>6<<5; Gmask = 0xFF>>5<<2; Bmask = 0xFF>>6; break;
			case 12:
				Rmask = 0xFF>>4<<8; Gmask = 0xFF>>4<<4; Bmask = 0xFF>>4; break;
			case 15:
				Rmask = 0xFF>>3<<10; Gmask = 0xFF>>3<<5; Bmask = 0xFF>>3; break;
			case 16:
				Rmask = 0xFF>>3<<11; Gmask = 0xFF>>2<<5; Bmask = 0xFF>>3; break;
			case 24:
			case 32:
				Rmask = 0xFF<<16; Gmask = 0xFF<<8; Bmask = 0xFF; break;
			default:
				return RAISE(PyExc_ValueError, "nonstandard bit depth given");
			}
		}
	}
	else /*no depth or surface*/
	{
		SDL_PixelFormat* pix;
		if(depth && PySurface_Check(depth))
			pix = ((PySurfaceObject*)depth)->surf->format;
		else if(SDL_GetVideoSurface())
			pix = SDL_GetVideoSurface()->format;
		else if(SDL_WasInit(SDL_INIT_VIDEO))
			pix = SDL_GetVideoInfo()->vfmt;
		else
		{
			pix = &default_format;
			pix->BitsPerPixel = 32; pix->Amask = 0;
			pix->Rmask = 0xFF0000; pix->Gmask = 0xFF00; pix->Bmask = 0xFF;
		}
		bpp = pix->BitsPerPixel;
		Rmask = pix->Rmask;
		Gmask = pix->Gmask;
		Bmask = pix->Bmask;
		Amask = pix->Amask;
	}
	surface = SDL_CreateRGBSurface(flags, width, height, bpp, Rmask, Gmask, Bmask, Amask);
	final = PySurface_New(surface);
	if(!final)
		SDL_FreeSurface(surface);
	return final;
}

#if PYTHON_API_VERSION >= 1011 /*this is the python-2.2 constructor*/
static PyObject* surface_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	return Surface(NULL, args);
}
#endif




static PyMethodDef surface_builtins[] =
{
#if PYTHON_API_VERSION < 1011 /*PYTHON2.2*/
	{ "Surface", Surface, 1, doc_Surface },
#endif
	{ NULL, NULL }
};



    /*DOC*/ static char doc_pygame_surface_MODULE[] =
    /*DOC*/    "The surface module doesn't have much in the line of functions. It\n"
    /*DOC*/    "does have the Surface object, and one routine to create new\n"
    /*DOC*/    "surfaces, pygame.Surface().\n"
    /*DOC*/ ;

PYGAME_EXPORT
void initsurface(void)
{
	PyObject *module, *dict, *apiobj, *lockmodule;
	static void* c_api[PYGAMEAPI_SURFACE_NUMSLOTS];

	PyType_Init(PySurface_Type);

    /* create the module */
	module = Py_InitModule3("surface", surface_builtins, doc_pygame_surface_MODULE);
	dict = PyModule_GetDict(module);

	PyDict_SetItemString(dict, "SurfaceType", (PyObject *)&PySurface_Type);
#if PYTHON_API_VERSION >= 1011 /*this is the python-2.2 constructor*/
	PyDict_SetItemString(dict, "Surface", (PyObject *)&PySurface_Type);
#endif

	/* export the c api */
	c_api[0] = &PySurface_Type;
	c_api[1] = PySurface_New;
	c_api[2] = PySurface_Blit;
	apiobj = PyCObject_FromVoidPtr(c_api, NULL);
	PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
	Py_DECREF(apiobj);
	/*imported needed apis*/
	import_pygame_base();
	import_pygame_rect();

	/*import the surflock module manually*/
	lockmodule = PyImport_ImportModule("pygame.surflock");
	if(lockmodule != NULL)
	{
		PyObject *dict = PyModule_GetDict(lockmodule);
		PyObject *c_api = PyDict_GetItemString(dict, PYGAMEAPI_LOCAL_ENTRY);
		if(PyCObject_Check(c_api))
		{
			int i; void** localptr = (void*)PyCObject_AsVoidPtr(c_api);
			for(i = 0; i < PYGAMEAPI_SURFLOCK_NUMSLOTS; ++i)
				PyGAME_C_API[i + PYGAMEAPI_SURFLOCK_FIRSTSLOT] = localptr[i];
		}
		Py_DECREF(lockmodule);
	}
}

