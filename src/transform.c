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
 *  surface transformations for pygame
 */
#include "pygame.h"




static SDL_Surface* newsurf_fromsurf(SDL_Surface* surf, int width, int height)
{
	SDL_Surface* newsurf;

	if(surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
		return (SDL_Surface*)(RAISE(PyExc_ValueError, "unsupport Surface bit depth for transform"));

	newsurf = SDL_CreateRGBSurface(surf->flags, width, height, surf->format->BitsPerPixel,
				surf->format->Rmask, surf->format->Gmask, surf->format->Bmask, surf->format->Amask);
	if(!newsurf)
		return (SDL_Surface*)(RAISE(PyExc_SDLError, SDL_GetError()));

	/* Copy palette, colorkey, etc info */
	if(surf->format->BytesPerPixel==1 && surf->format->palette)
		SDL_SetColors(newsurf, surf->format->palette->colors, 0, surf->format->palette->ncolors);
	if(surf->flags & SDL_SRCCOLORKEY)
		SDL_SetColorKey(newsurf, (surf->flags&SDL_RLEACCEL)|SDL_SRCCOLORKEY, surf->format->colorkey);

	return newsurf;
}





static void rotate(SDL_Surface *src, SDL_Surface *dst, Uint32 bgcolor, int cx, int cy, int isin, int icos)
{
	int x, y, dx, dy, sdx, sdy;

	Uint8 *srcpix = (Uint8*)src->pixels;
	Uint8 *dstrow = (Uint8*)dst->pixels;

	int pixsize = src->format->BytesPerPixel;
	int srcpitch = src->pitch;
	int dstpitch = dst->pitch;

	int xd = (src->w - dst->w) << 15;
	int yd = (src->h - dst->h) << 15;

	int ax = (cx << 16) - (icos * cx);
	int ay = (cy << 16) - (isin * cx);

	for(y = 0; y < dst->h; y++)
	{
		Uint8 *srcpos, *dstpos = (Uint8*)dstrow;
		dy = cy - y;
		sdx = (ax + (isin * dy)) + xd;
		sdy = (ay - (icos * dy)) + yd;
		for(x = 0; x < dst->w; x++)
		{
			dx = sdx >> 16;
			dy = sdy >> 16;
			if((dx >= 0) && (dy >= 0) && (dx < src->w) && (dy < src->h))
			{
				srcpos = (Uint8*)(srcpix + (dy * srcpitch) + (dx * pixsize));
				*dstpos++ = *srcpos;
			}
			else
				*dstpos++ = bgcolor;
			sdx += icos;
			sdy += isin;
		}
		dstrow += dstpitch;
	}  
}



static void stretch(SDL_Surface *src, SDL_Surface *dst)
{
	int looph, loopw;
	
	Uint8* srcrow = (Uint8*)src->pixels;
	Uint8* dstrow = (Uint8*)dst->pixels;

	int srcpitch = src->pitch;
	int dstpitch = dst->pitch;

	int dstwidth = dst->w;
	int dstheight = dst->h;
	int dstwidth2 = dst->w << 1;
	int dstheight2 = dst->h << 1;

	int srcwidth = src->w;
	int srcheight = src->h;
	int srcwidth2 = src->w << 1;
	int srcheight2 = src->h << 1;

	int w_err, h_err = srcheight2 - dstheight2;


	switch(src->format->BytesPerPixel)
	{
	case 1:
		for(looph = 0; looph < dstheight; ++looph)
		{
			Uint8 *srcpix = (Uint8*)srcrow, *dstpix = (Uint8*)dstrow;
			w_err = srcwidth2 - dstwidth2;
			for(loopw = 0; loopw < dstwidth; ++ loopw)
			{
				*dstpix++ = *srcpix;
				while(w_err >= 0) {++srcpix; w_err -= dstwidth2;}
				w_err += srcwidth2;
			}
			while(h_err >= 0) {srcrow += srcpitch; h_err -= dstheight2;}
			dstrow += dstpitch;
			h_err += srcheight2;
		}break;
	case 2:
		for(looph = 0; looph < dstheight; ++looph)
		{
			Uint16 *srcpix = (Uint16*)srcrow, *dstpix = (Uint16*)dstrow;
			w_err = srcwidth2 - dstwidth2;
			for(loopw = 0; loopw < dstwidth; ++ loopw)
			{
				*dstpix++ = *srcpix;
				while(w_err >= 0) {++srcpix; w_err -= dstwidth2;}
				w_err += srcwidth2;
			}
			while(h_err >= 0) {srcrow += srcpitch; h_err -= dstheight2;}
			dstrow += dstpitch;
			h_err += srcheight2;
		}break;
	case 3:
		for(looph = 0; looph < dstheight; ++looph)
		{
			Uint8 *srcpix = (Uint8*)srcrow, *dstpix = (Uint8*)dstrow;
			w_err = srcwidth2 - dstwidth2;
			for(loopw = 0; loopw < dstwidth; ++ loopw)
			{
				dstpix[0] = srcpix[0]; dstpix[1] = srcpix[1]; dstpix[2] = srcpix[2];
				dstpix += 3;
				while(w_err >= 0) {srcpix+=3; w_err -= dstwidth2;}
				w_err += srcwidth2;
			}
			while(h_err >= 0) {srcrow += srcpitch; h_err -= dstheight2;}
			dstrow += dstpitch;
			h_err += srcheight2;
		}break;
	default: /*case 4:*/
		for(looph = 0; looph < dstheight; ++looph)
		{
			Uint32 *srcpix = (Uint32*)srcrow, *dstpix = (Uint32*)dstrow;
			w_err = srcwidth2 - dstwidth2;
			for(loopw = 0; loopw < dstwidth; ++ loopw)
			{
				*dstpix++ = *srcpix;
				while(w_err >= 0) {++srcpix; w_err -= dstwidth2;}
				w_err += srcwidth2;
			}
			while(h_err >= 0) {srcrow += srcpitch; h_err -= dstheight2;}
			dstrow += dstpitch;
			h_err += srcheight2;
		}break;
	}
}





    /*DOC*/ static char doc_scale[] =
    /*DOC*/    "pygame.transform.scale(Surface, size) -> Surface\n"
    /*DOC*/    "scale a Surface to an arbitrary size\n"
    /*DOC*/    "\n"
    /*DOC*/    "Scale the image to the new resolution.\n"
    /*DOC*/ ;

static PyObject* surf_scale(PyObject* self, PyObject* arg)
{
	PyObject *surfobj;
	SDL_Surface* surf, *newsurf;
	int width, height;

	/*get all the arguments*/
	if(!PyArg_ParseTuple(arg, "O!(ii)", &PySurface_Type, &surfobj, &width, &height))
		return NULL;
	surf = PySurface_AsSurface(surfobj);

	newsurf = newsurf_fromsurf(surf, width, height);
	if(!newsurf) return NULL;

	SDL_LockSurface(newsurf);
	PySurface_Lock(surfobj);

	stretch(surf, newsurf);

	PySurface_Unlock(surfobj);
	SDL_UnlockSurface(newsurf);

	return PySurface_New(newsurf);
}




    /*DOC*/ static char doc_rotate[] =
    /*DOC*/    "pygame.transform.rotate(Surface, angle) -> Surface\n"
    /*DOC*/    "rotate a Surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Rotates the image clockwise with the given angle (in degrees).\n"
    /*DOC*/    "The result size will likely be a different resolution than the\n"
    /*DOC*/    "original.\n"
    /*DOC*/ ;

static PyObject* surf_rotate(PyObject* self, PyObject* arg)
{
	PyObject *surfobj;
	SDL_Surface* surf, *newsurf;
	float angle;

	double radangle, sangle, cangle;
	int dstwidthhalf, dstheighthalf;
	double x, y, cx, cy, sx, sy;
	int nxmax,nymax;
	Uint32 bgcolor;

	/*get all the arguments*/
	if(!PyArg_ParseTuple(arg, "O!f", &PySurface_Type, &surfobj, &angle))
		return NULL;
	surf = PySurface_AsSurface(surfobj);


	if(surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
		return RAISE(PyExc_ValueError, "unsupport Surface bit depth for transform");

	radangle = angle*.01745329251994329;
	sangle = sin(radangle);
	cangle = cos(radangle);
	
	x = surf->w/2;
	y = surf->h/2;
	cx = cangle*x;
	cy = cangle*y;
	sx = sangle*x;
	sy = sangle*y;
	nxmax = (int)ceil(max(max(max(fabs(cx+sy), fabs(cx-sy)), fabs(-cx+sy)), fabs(-cx-sy)));
	nymax = (int)ceil(max(max(max(fabs(sx+cy), fabs(sx-cy)), fabs(-sx+cy)), fabs(-sx-cy)));
	dstwidthhalf = nxmax ? nxmax : 1;
	dstheighthalf = nymax ? nymax : 1;

	newsurf = newsurf_fromsurf(surf, dstwidthhalf*2, dstheighthalf*2);
	if(!newsurf) return NULL;

	/* get the background color */
	if(surf->flags & SDL_SRCCOLORKEY)
		bgcolor = surf->format->colorkey;
	else
	{
		switch(surf->format->BytesPerPixel)
		{
		case 1: bgcolor = *(Uint8*)surf->pixels; break;
		case 2: bgcolor = *(Uint16*)surf->pixels; break;
		case 4: bgcolor = *(Uint32*)surf->pixels; break;
		default: /*case 3:*/
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
			bgcolor = (((Uint8*)surf->pixels)[0]) + (((Uint8*)surf->pixels)[1]<<8) + (((Uint8*)surf->pixels)[2]<<16);
#else
			bgcolor = (((Uint8*)surf->pixels)[2]) + (((Uint8*)surf->pixels)[1]<<8) + (((Uint8*)surf->pixels)[0]<<16);
#endif
		}
	}

	SDL_LockSurface(newsurf);
	PySurface_Lock(surfobj);

	rotate(surf, newsurf, bgcolor, dstwidthhalf, dstheighthalf, (int)(sangle*65536), (int)(cangle*65536));

	PySurface_Unlock(surfobj);
	SDL_UnlockSurface(newsurf);

	return PySurface_New(newsurf);
}








static PyMethodDef transform_builtins[] =
{
	{ "scale", surf_scale, 1, doc_scale },
	{ "rotate", surf_rotate, 1, doc_rotate },

	{ NULL, NULL }
};



    /*DOC*/ static char doc_pygame_transform_MODULE[] =
    /*DOC*/    "Contains routines to transform a Surface image.\n"
    /*DOC*/    "\n"
    /*DOC*/    "All transformation functions take a source Surface and\n"
    /*DOC*/    "return a new copy of that surface in the same format as\n"
    /*DOC*/    "the original.\n"
    /*DOC*/    "\n"
    /*DOC*/    "These routines are not filtered or smoothed.\n"
    /*DOC*/ ;

PYGAME_EXPORT
void inittransform(void)
{
	PyObject *module;
	module = Py_InitModule3("transform", transform_builtins, doc_pygame_transform_MODULE);

	/*imported needed apis*/
	import_pygame_base();
	import_pygame_rect();
	import_pygame_surface();
}


