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
#include <math.h>


void scale2x(SDL_Surface *src, SDL_Surface *dst);



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


static SDL_Surface* rotate90(SDL_Surface *src, int angle)
{
    int numturns = (angle / 90) % 4;
    int dstwidth, dstheight;
    SDL_Surface* dst;
    char *srcpix, *dstpix, *srcrow, *dstrow;
    int srcstepx, srcstepy, dststepx, dststepy;
    int loopx, loopy;

    if(numturns<0)
        numturns = 4+numturns;
    if(!(numturns % 2))
    {
        dstwidth = src->w;
        dstheight = src->h;
    }
    else
    {
        dstwidth = src->h;
        dstheight = src->w;
    }
    
    dst = newsurf_fromsurf(src, dstwidth, dstheight);
    if(!dst)
        return NULL;
	SDL_LockSurface(dst);
    srcrow = (char*)src->pixels;
    dstrow = (char*)dst->pixels;
    srcstepx = dststepx = src->format->BytesPerPixel;
    srcstepy = src->pitch;
    dststepy = dst->pitch;
    
    switch(numturns)
    {
    /*case 0: we don't need to change anything*/
    case 1:
        srcrow += ((src->w-1)*srcstepx);
        srcstepy = -srcstepx;
        srcstepx = src->pitch;
        break;
    case 2:
        srcrow += ((src->h-1)*srcstepy) + ((src->w-1)*srcstepx);
        srcstepx = -srcstepx;
        srcstepy = -srcstepy;
        break;
    case 3:
        srcrow += ((src->h-1)*srcstepy);
        srcstepx = -srcstepy;
        srcstepy = src->format->BytesPerPixel;
        break;
    }

    switch(src->format->BytesPerPixel)
    {
    case 1:
        for(loopy=0; loopy<dstheight; ++loopy)
        {
            dstpix = dstrow; srcpix = srcrow;
            for(loopx=0; loopx<dstwidth; ++loopx)
            {
                *dstpix = *srcpix;
                srcpix += srcstepx; dstpix += dststepx;
            }
            dstrow += dststepy; srcrow += srcstepy;
        }break;
    case 2:
        for(loopy=0; loopy<dstheight; ++loopy)
        {
            dstpix = dstrow; srcpix = srcrow;
            for(loopx=0; loopx<dstwidth; ++loopx)
            {
                *(Uint16*)dstpix = *(Uint16*)srcpix;
                srcpix += srcstepx; dstpix += dststepx;
            }
            dstrow += dststepy; srcrow += srcstepy;
        }break;
    case 3:
        for(loopy=0; loopy<dstheight; ++loopy)
        {
            dstpix = dstrow; srcpix = srcrow;
            for(loopx=0; loopx<dstwidth; ++loopx)
            {
                dstpix[0] = srcpix[0]; dstpix[1] = srcpix[1]; dstpix[2] = srcpix[2];
                srcpix += srcstepx; dstpix += dststepx;
            }
            dstrow += dststepy; srcrow += srcstepy;
        }break;
    case 4:
        for(loopy=0; loopy<dstheight; ++loopy)
        {
            dstpix = dstrow; srcpix = srcrow;
            for(loopx=0; loopx<dstwidth; ++loopx)
            {
                *(Uint32*)dstpix = *(Uint32*)srcpix;
                srcpix += srcstepx; dstpix += dststepx;
            }
            dstrow += dststepy; srcrow += srcstepy;
        }
    }
	SDL_UnlockSurface(dst);
    return dst;
}


static void rotate(SDL_Surface *src, SDL_Surface *dst, Uint32 bgcolor, double sangle, double cangle)
{
	int x, y, dx, dy;
    
	Uint8 *srcpix = (Uint8*)src->pixels;
	Uint8 *dstrow = (Uint8*)dst->pixels;
	int srcpitch = src->pitch;
	int dstpitch = dst->pitch;

	int cy = dst->h / 2;
	int xd = ((src->w - dst->w) << 15);
	int yd = ((src->h - dst->h) << 15);
    
	int isin = (int)(sangle*65536);
	int icos = (int)(cangle*65536);
   
	int ax = ((dst->w) << 15) - (int)(cangle * ((dst->w-1) << 15));
	int ay = ((dst->h) << 15) - (int)(sangle * ((dst->w-1) << 15));

	int xmaxval = ((src->w) << 16) - 1;
	int ymaxval = ((src->h) << 16) - 1;
    
	switch(src->format->BytesPerPixel)
	{
	case 1:
		for(y = 0; y < dst->h; y++) {
			Uint8 *dstpos = (Uint8*)dstrow;
			dx = (ax + (isin * (cy - y))) + xd;
			dy = (ay - (icos * (cy - y))) + yd;
			for(x = 0; x < dst->w; x++) {
				if(dx<0 || dy<0 || dx>xmaxval || dy>ymaxval) *dstpos++ = bgcolor;
				else *dstpos++ = *(Uint8*)(srcpix + ((dy>>16) * srcpitch) + (dx>>16));
				dx += icos; dy += isin;
			}
			dstrow += dstpitch;
		}break;
        case 2:
		for(y = 0; y < dst->h; y++) {
			Uint16 *dstpos = (Uint16*)dstrow;
			dx = (ax + (isin * (cy - y))) + xd;
			dy = (ay - (icos * (cy - y))) + yd;
			for(x = 0; x < dst->w; x++) {
				if(dx<0 || dy<0 || dx>xmaxval || dy>ymaxval) *dstpos++ = bgcolor;
				else *dstpos++ = *(Uint16*)(srcpix + ((dy>>16) * srcpitch) + (dx>>16<<1));
				dx += icos; dy += isin;
			}
			dstrow += dstpitch;
		}break;
	case 4:
		for(y = 0; y < dst->h; y++) {
			Uint32 *dstpos = (Uint32*)dstrow;
			dx = (ax + (isin * (cy - y))) + xd;
			dy = (ay - (icos * (cy - y))) + yd;
			for(x = 0; x < dst->w; x++) {
				if(dx<0 || dy<0 || dx>xmaxval || dy>ymaxval) *dstpos++ = bgcolor;
				else *dstpos++ = *(Uint32*)(srcpix + ((dy>>16) * srcpitch) + (dx>>16<<2));
				dx += icos; dy += isin;
			}
			dstrow += dstpitch;
		}break;
	default: /*case 3:*/
		for(y = 0; y < dst->h; y++) {
			Uint8 *dstpos = (Uint8*)dstrow;
			dx = (ax + (isin * (cy - y))) + xd;
			dy = (ay - (icos * (cy - y))) + yd;
			for(x = 0; x < dst->w; x++) {
				if(dx<0 || dy<0 || dx>xmaxval || dy>ymaxval)
				{
					dstpos[0] = ((Uint8*)&bgcolor)[0]; dstpos[1] = ((Uint8*)&bgcolor)[1]; dstpos[2] = ((Uint8*)&bgcolor)[2];
					dstpos += 3;
				}
				else {
					Uint8* srcpos = (Uint8*)(srcpix + ((dy>>16) * srcpitch) + ((dx>>16) * 3));
					dstpos[0] = srcpos[0]; dstpos[1] = srcpos[1]; dstpos[2] = srcpos[2];
					dstpos += 3;
				}
				dx += icos; dy += isin;
			}
			dstrow += dstpitch;
		}break;
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
    /*DOC*/    "This will resize a surface to the given resolution.\n"
    /*DOC*/    "The size is simply any 2 number sequence representing\n"
    /*DOC*/    "the width and height.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This transformation is not filtered.\n"
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




    /*DOC*/ static char doc_scale2x[] =
    /*DOC*/    "pygame.transform.scale2x(Surface) -> Surface\n"
    /*DOC*/    "doubles the size of the image with advanced scaling\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will return a new image that is double the size of\n"
    /*DOC*/    "the original. It uses the AdvanceMAME Scale2X algorithm\n"
    /*DOC*/    "which does a 'jaggie-less' scale of bitmap graphics.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This really only has an effect on simple images with solid\n"
    /*DOC*/    "colors. On photographic and antialiased images it will look\n"
    /*DOC*/    "like a regular unfiltered scale.\n"
    /*DOC*/ ;

static PyObject* surf_scale2x(PyObject* self, PyObject* arg)
{
	PyObject *surfobj;
	SDL_Surface* surf, *newsurf;
	int width, height;

	/*get all the arguments*/
	if(!PyArg_ParseTuple(arg, "O!", &PySurface_Type, &surfobj))
		return NULL;
	surf = PySurface_AsSurface(surfobj);

	width = surf->w * 2;
	height = surf->h * 2;
	
	newsurf = newsurf_fromsurf(surf, width, height);
	if(!newsurf) return NULL;

	SDL_LockSurface(newsurf);
	PySurface_Lock(surfobj);

	scale2x(surf, newsurf);

	PySurface_Unlock(surfobj);
	SDL_UnlockSurface(newsurf);

	return PySurface_New(newsurf);
}




    /*DOC*/ static char doc_rotate[] =
    /*DOC*/    "pygame.transform.rotate(Surface, angle) -> Surface\n"
    /*DOC*/    "rotate a Surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Rotates the image counterclockwise with the given angle\n"
    /*DOC*/    "(in degrees). The angle can be any floating point value\n"
    /*DOC*/    "(negative rotation amounts will do clockwise rotations)\n"
    /*DOC*/    "\n"
    /*DOC*/    "Unless rotating by 90 degree increments, the resulting image\n"
    /*DOC*/    "size will be larger than the original. There will be newly\n"
    /*DOC*/    "uncovered areas in the image. These will filled with either\n"
    /*DOC*/    "the current colorkey for the Surface, or the topleft pixel value.\n"
    /*DOC*/    "(with the alpha channel zeroed out if available)\n"
    /*DOC*/    "\n"
    /*DOC*/    "This transformation is not filtered.\n"
    /*DOC*/ ;

static PyObject* surf_rotate(PyObject* self, PyObject* arg)
{
	PyObject *surfobj;
	SDL_Surface* surf, *newsurf;
	float angle;

	double radangle, sangle, cangle;
	double x, y, cx, cy, sx, sy;
	int nxmax,nymax;
	Uint32 bgcolor;

	/*get all the arguments*/
	if(!PyArg_ParseTuple(arg, "O!f", &PySurface_Type, &surfobj, &angle))
		return NULL;
	surf = PySurface_AsSurface(surfobj);


	if(surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
		return RAISE(PyExc_ValueError, "unsupport Surface bit depth for transform");

        if(!(((int)angle)%90))
        {
			PySurface_Lock(surfobj);
            newsurf = rotate90(surf, (int)angle);
			PySurface_Unlock(surfobj);
            if(!newsurf) return NULL;
            return PySurface_New(newsurf);
        }
        
        
	radangle = angle*.01745329251994329;
	sangle = sin(radangle);
	cangle = cos(radangle);

	x = surf->w;
	y = surf->h;
	cx = cangle*x;
	cy = cangle*y;
	sx = sangle*x;
	sy = sangle*y;
        nxmax = (int)(max(max(max(fabs(cx+sy), fabs(cx-sy)), fabs(-cx+sy)), fabs(-cx-sy)));
	nymax = (int)(max(max(max(fabs(sx+cy), fabs(sx-cy)), fabs(-sx+cy)), fabs(-sx-cy)));

	newsurf = newsurf_fromsurf(surf, nxmax, nymax);
	if(!newsurf) return NULL;

	/* get the background color */
	if(surf->flags & SDL_SRCCOLORKEY)
	{
		bgcolor = surf->format->colorkey;
	}
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
		bgcolor &= ~surf->format->Amask;
	}

	SDL_LockSurface(newsurf);
	PySurface_Lock(surfobj);

	rotate(surf, newsurf, bgcolor, sangle, cangle);

	PySurface_Unlock(surfobj);
	SDL_UnlockSurface(newsurf);

	return PySurface_New(newsurf);
}




    /*DOC*/ static char doc_flip[] =
    /*DOC*/    "pygame.transform.flip(Surface, xaxis, yaxis) -> Surface\n"
    /*DOC*/    "flips a surface on either axis\n"
    /*DOC*/    "\n"
    /*DOC*/    "Flips the image on the x-axis or the y-axis if the argument\n"
    /*DOC*/    "for that axis is true.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The flip operation is nondestructive, you may flip the image\n"
    /*DOC*/    "as many times as you like, and always be able to recreate the\n"
    /*DOC*/    "exact original image.\n"
    /*DOC*/ ;

static PyObject* surf_flip(PyObject* self, PyObject* arg)
{
	PyObject *surfobj;
	SDL_Surface* surf, *newsurf;
	int xaxis, yaxis;
	int loopx, loopy;
	int pixsize, srcpitch, dstpitch;
	Uint8 *srcpix, *dstpix;

	/*get all the arguments*/
	if(!PyArg_ParseTuple(arg, "O!ii", &PySurface_Type, &surfobj, &xaxis, &yaxis))
		return NULL;
	surf = PySurface_AsSurface(surfobj);

	newsurf = newsurf_fromsurf(surf, surf->w, surf->h);
	if(!newsurf) return NULL;

	pixsize = surf->format->BytesPerPixel;
	srcpitch = surf->pitch;
	dstpitch = newsurf->pitch;

	SDL_LockSurface(newsurf);
	PySurface_Lock(surfobj);

	srcpix = (Uint8*)surf->pixels;
	dstpix = (Uint8*)newsurf->pixels;

	if(!xaxis)
	{
		if(!yaxis)
		{
			for(loopy = 0; loopy < surf->h; ++loopy)
				memcpy(dstpix+loopy*dstpitch, srcpix+loopy*srcpitch, surf->w*surf->format->BytesPerPixel);
		}
		else
		{
			for(loopy = 0; loopy < surf->h; ++loopy)
				memcpy(dstpix+loopy*dstpitch, srcpix+(surf->h-1-loopy)*srcpitch, surf->w*surf->format->BytesPerPixel);
		}
	}
	else /*if (xaxis)*/
	{
		if(yaxis)
		{
			switch(surf->format->BytesPerPixel)
			{
			case 1:
				for(loopy = 0; loopy < surf->h; ++loopy) {
					Uint8* dst = (Uint8*)(dstpix+loopy*dstpitch);
					Uint8* src = ((Uint8*)(srcpix+(surf->h-1-loopy)*srcpitch)) + surf->w - 1;
					for(loopx = 0; loopx < surf->w; ++loopx)
						*dst++ = *src--;
				}break;
			case 2:
				for(loopy = 0; loopy < surf->h; ++loopy) {
					Uint16* dst = (Uint16*)(dstpix+loopy*dstpitch);
					Uint16* src = ((Uint16*)(srcpix+(surf->h-1-loopy)*srcpitch)) + surf->w - 1;
					for(loopx = 0; loopx < surf->w; ++loopx)
						*dst++ = *src--;
				}break;
			case 4:
				for(loopy = 0; loopy < surf->h; ++loopy) {
					Uint32* dst = (Uint32*)(dstpix+loopy*dstpitch);
					Uint32* src = ((Uint32*)(srcpix+(surf->h-1-loopy)*srcpitch)) + surf->w - 1;
					for(loopx = 0; loopx < surf->w; ++loopx)
						*dst++ = *src--;
				}break;
			case 3:
				for(loopy = 0; loopy < surf->h; ++loopy) {
					Uint8* dst = (Uint8*)(dstpix+loopy*dstpitch);
					Uint8* src = ((Uint8*)(srcpix+(surf->h-1-loopy)*srcpitch)) + surf->w*3 - 3;
					for(loopx = 0; loopx < surf->w; ++loopx)
					{
						dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
						dst += 3;
						src -= 3;
					}
				}break;
			}
		}
		else
		{
			switch(surf->format->BytesPerPixel)
			{
			case 1:
				for(loopy = 0; loopy < surf->h; ++loopy) {
					Uint8* dst = (Uint8*)(dstpix+loopy*dstpitch);
					Uint8* src = ((Uint8*)(srcpix+loopy*srcpitch)) + surf->w - 1;
					for(loopx = 0; loopx < surf->w; ++loopx)
						*dst++ = *src--;
				}break;
			case 2:
				for(loopy = 0; loopy < surf->h; ++loopy) {
					Uint16* dst = (Uint16*)(dstpix+loopy*dstpitch);
					Uint16* src = ((Uint16*)(srcpix+loopy*srcpitch)) + surf->w - 1;
					for(loopx = 0; loopx < surf->w; ++loopx)
						*dst++ = *src--;
				}break;
			case 4:
				for(loopy = 0; loopy < surf->h; ++loopy) {
					Uint32* dst = (Uint32*)(dstpix+loopy*dstpitch);
					Uint32* src = ((Uint32*)(srcpix+loopy*srcpitch)) + surf->w - 1;
					for(loopx = 0; loopx < surf->w; ++loopx)
						*dst++ = *src--;
				}break;
			case 3:
				for(loopy = 0; loopy < surf->h; ++loopy) {
					Uint8* dst = (Uint8*)(dstpix+loopy*dstpitch);
					Uint8* src = ((Uint8*)(srcpix+loopy*srcpitch)) + surf->w*3 - 3;
					for(loopx = 0; loopx < surf->w; ++loopx)
					{
						dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
						dst += 3;
						src -= 3;
					}
				}break;
			}
		}
	}

	PySurface_Unlock(surfobj);
	SDL_UnlockSurface(newsurf);

	return PySurface_New(newsurf);
}




extern SDL_Surface *rotozoomSurface(SDL_Surface *src, double angle, double zoom, int smooth);

    /*DOC*/ static char doc_rotozoom[] =
    /*DOC*/    "pygame.transform.rotozoom(Surface, angle, zoom) -> Surface\n"
    /*DOC*/    "smoothly scale and/or rotate an image\n"
    /*DOC*/    "\n"
    /*DOC*/    "The angle argument is the number of degrees to rotate\n"
    /*DOC*/    "counter-clockwise. The angle can be any floating point value.\n"
    /*DOC*/    "(negative rotation amounts will do clockwise rotations)\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will smoothly rotate and scale an image in one pass.\n"
    /*DOC*/    "The resulting image will always be a 32bit version of the\n"
    /*DOC*/    "original surface. The scale is a multiplier for the image\n"
    /*DOC*/    "size, and angle is the degrees to rotate counter clockwise.\n"
    /*DOC*/    "\n"
    /*DOC*/    "It calls the SDL_rotozoom library which is compiled in.\n"
    /*DOC*/    "Note that the code in SDL_rotozoom is fairly messy and your\n"
    /*DOC*/    "resulting image could be shifted and contain artifacts.\n"
    /*DOC*/ ;

static PyObject* surf_rotozoom(PyObject* self, PyObject* arg)
{
	PyObject *surfobj;
	SDL_Surface *surf, *newsurf, *surf32;
	float scale, angle;

	/*get all the arguments*/
	if(!PyArg_ParseTuple(arg, "O!ff", &PySurface_Type, &surfobj, &angle, &scale))
		return NULL;
	surf = PySurface_AsSurface(surfobj);

	if(surf->format->BitsPerPixel == 32)
	{
		surf32 = surf;
		PySurface_Lock(surfobj);
	}
	else
	{
	    surf32 = SDL_CreateRGBSurface(SDL_SWSURFACE, surf->w, surf->h, 32,
					0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000);
		SDL_BlitSurface(surf, NULL, surf32, NULL);
	}

/* don't special case the 90 degrees, makes the rotating image pop
	if(scale == 1.0 && !(((int)angle)%90))
		newsurf = rotate90(surf32, (int)angle);
	else
*/		newsurf = rotozoomSurface(surf32, angle, scale, 1);

	if(surf32 == surf)
		PySurface_Unlock(surfobj);
	else
		SDL_FreeSurface(surf32);
	return PySurface_New(newsurf);
}



static PyMethodDef transform_builtins[] =
{
	{ "scale", surf_scale, 1, doc_scale },
	{ "rotate", surf_rotate, 1, doc_rotate },
	{ "flip", surf_flip, 1, doc_flip },
	{ "rotozoom", surf_rotozoom, 1, doc_rotozoom},
	{ "scale2x", surf_scale2x, 1, doc_scale2x},
		
	{ NULL, NULL }
};



    /*DOC*/ static char doc_pygame_transform_MODULE[] =
    /*DOC*/    "Contains routines to transform a Surface image.\n"
    /*DOC*/    "\n"
    /*DOC*/    "All transformation functions take a source Surface and\n"
    /*DOC*/    "return a new copy of that surface in the same format as\n"
    /*DOC*/    "the original.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Some of the\n"
    /*DOC*/    "filters are 'destructive', which means if you transform\n"
    /*DOC*/    "the image one way, you can't transform the image back to\n"
    /*DOC*/    "the exact same way as it was before. If you plan on doing\n"
    /*DOC*/    "many transforms, it is good practice to keep the original\n"
    /*DOC*/    "untransformed image, and only translate that image.\n"
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


