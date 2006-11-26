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
#include "pygamedocs.h"
#include <math.h>


void scale2x(SDL_Surface *src, SDL_Surface *dst);



static SDL_Surface* newsurf_fromsurf(SDL_Surface* surf, int width, int height)
{
	SDL_Surface* newsurf;
        int result;

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

	if(surf->flags&SDL_SRCALPHA) {
            result = SDL_SetAlpha(newsurf, surf->flags, surf->format->alpha);

            if(result == -1)
                return (SDL_Surface*)(RAISE(PyExc_SDLError, SDL_GetError()));
        }



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


static PyObject* surf_scale(PyObject* self, PyObject* arg)
{
	PyObject *surfobj, *surfobj2;
	SDL_Surface* surf, *newsurf;
	int width, height;
        surfobj2 = NULL;


	/*get all the arguments*/
	if(!PyArg_ParseTuple(arg, "O!(ii)|O!", &PySurface_Type, &surfobj, 
                                               &width, &height, 
                                               &PySurface_Type, &surfobj2)) {
		return NULL;
        }

	if(width < 0 || height < 0)
		return RAISE(PyExc_ValueError, "Cannot scale to negative size");

	surf = PySurface_AsSurface(surfobj);
	
        if(!surfobj2) {

            newsurf = newsurf_fromsurf(surf, width, height);
            if(!newsurf) return NULL;
        } else {
            newsurf = PySurface_AsSurface(surfobj2);
        }


        /* check to see if the size is twice as big. */
        if(newsurf->w != (width) || newsurf->h != (height)) {
            return RAISE(PyExc_ValueError, 
                         "Destination surface not the given width or height.");
        }

        /* check to see if the format of the surface is the same. */
        if(surf->format->BytesPerPixel != newsurf->format->BytesPerPixel) {
            return RAISE(PyExc_ValueError, 
                         "Source and destination surfaces need the same format.");
        }

	if(width && height)
	{
		SDL_LockSurface(newsurf);
		PySurface_Lock(surfobj);
	
                Py_BEGIN_ALLOW_THREADS
		stretch(surf, newsurf);
                Py_END_ALLOW_THREADS

		PySurface_Unlock(surfobj);
		SDL_UnlockSurface(newsurf);
	}

	if(surfobj2) {
            Py_INCREF(surfobj2);
            return surfobj2;
        } else {
            return PySurface_New(newsurf);
        }
}



static PyObject* surf_scale2x(PyObject* self, PyObject* arg)
{
	PyObject *surfobj, *surfobj2;
	SDL_Surface *surf;
        SDL_Surface *newsurf;
	int width, height;
        surfobj2 = NULL;
        

	/*get all the arguments*/
        if(!PyArg_ParseTuple(arg, "O!|O!", &PySurface_Type, &surfobj, 
                                           &PySurface_Type, &surfobj2)) {
            return NULL;
        }

	surf = PySurface_AsSurface(surfobj);

        /* if the second surface is not there, then make a new one. */

        if(!surfobj2) {
            width = surf->w * 2;
            height = surf->h * 2;

            newsurf = newsurf_fromsurf(surf, width, height);

            if(!newsurf) return NULL;
        } else {
            newsurf = PySurface_AsSurface(surfobj2);
        }

	

        /* check to see if the size is twice as big. */
        if(newsurf->w != (surf->w * 2) || newsurf->h != (surf->h * 2)) {
            return RAISE(PyExc_ValueError, 
                         "Destination surface not 2x bigger.");
        }

        /* check to see if the format of the surface is the same. */
        if(surf->format->BytesPerPixel != newsurf->format->BytesPerPixel) {
            return RAISE(PyExc_ValueError, 
                         "Source and destination surfaces need the same format.");
        }

        SDL_LockSurface(newsurf);
        SDL_LockSurface(surf);

        Py_BEGIN_ALLOW_THREADS
	scale2x(surf, newsurf);
        Py_END_ALLOW_THREADS

        SDL_UnlockSurface(surf);
        SDL_UnlockSurface(newsurf);

	if(surfobj2) {
            Py_INCREF(surfobj2);
            return surfobj2;
        } else {
            return PySurface_New(newsurf);
        }
}





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

            Py_BEGIN_ALLOW_THREADS
            newsurf = rotate90(surf, (int)angle);
            Py_END_ALLOW_THREADS

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

        Py_BEGIN_ALLOW_THREADS
	rotate(surf, newsurf, bgcolor, sangle, cangle);
        Py_END_ALLOW_THREADS

	PySurface_Unlock(surfobj);
	SDL_UnlockSurface(newsurf);

	return PySurface_New(newsurf);
}


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


        Py_BEGIN_ALLOW_THREADS

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
        Py_END_ALLOW_THREADS



	PySurface_Unlock(surfobj);
	SDL_UnlockSurface(newsurf);

	return PySurface_New(newsurf);
}




extern SDL_Surface *rotozoomSurface(SDL_Surface *src, double angle, double zoom, int smooth);


static PyObject* surf_rotozoom(PyObject* self, PyObject* arg)
{
	PyObject *surfobj;
	SDL_Surface *surf, *newsurf, *surf32;
	float scale, angle;

	/*get all the arguments*/
	if(!PyArg_ParseTuple(arg, "O!ff", &PySurface_Type, &surfobj, &angle, &scale))
		return NULL;
	surf = PySurface_AsSurface(surfobj);
	if(scale == 0.0)
	{
		newsurf = newsurf_fromsurf(surf, surf->w, surf->h);
		return PySurface_New(newsurf);
	}

	if(surf->format->BitsPerPixel == 32)
	{
		surf32 = surf;
		PySurface_Lock(surfobj);
	}
	else
	{
            Py_BEGIN_ALLOW_THREADS
	    surf32 = SDL_CreateRGBSurface(SDL_SWSURFACE, surf->w, surf->h, 32,
					0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000);
		SDL_BlitSurface(surf, NULL, surf32, NULL);
            Py_END_ALLOW_THREADS
	}

        Py_BEGIN_ALLOW_THREADS
	newsurf = rotozoomSurface(surf32, angle, scale, 1);
        Py_END_ALLOW_THREADS

	if(surf32 == surf)
		PySurface_Unlock(surfobj);
	else
		SDL_FreeSurface(surf32);
	return PySurface_New(newsurf);
}


static SDL_Surface* chop(SDL_Surface *src, int x, int y, int width, int height)
{
  SDL_Surface* dst;
  int dstwidth,dstheight;
  char *srcpix, *dstpix, *srcrow, *dstrow;
  int srcstepx, srcstepy, dststepx, dststepy;
  int loopx,loopy;

  if((x+width) > src->w)
    width=src->w-x;
  if((y+height) > src->h)
    height=src->h-y;
  if(x < 0)
    {
      width-=(-x);
      x=0;
    }
  if(y < 0)
    {
      height-=(-y);
      y=0;
    }

  dstwidth=src->w-width;
  dstheight=src->h-height;

  dst=newsurf_fromsurf(src,dstwidth,dstheight);
  if(!dst)
    return NULL;
  SDL_LockSurface(dst);
  srcrow=(char*)src->pixels;
  dstrow=(char*)dst->pixels;
  srcstepx=dststepx=src->format->BytesPerPixel;
  srcstepy=src->pitch;
  dststepy=dst->pitch;

  for(loopy=0; loopy < src->h; loopy++)
    {
      if((loopy < y) || (loopy >= (y+height)))
	{
	  dstpix=dstrow;
	  srcpix=srcrow;
	  for(loopx=0; loopx < src->w; loopx++)
	    {
	      if((loopx < x) || (loopx>= (x+width)))
		{
		  switch(src->format->BytesPerPixel)
		    {
		    case 1:
		      *dstpix=*srcpix;
		      break;
		    case 2:
		      *(Uint16*) dstpix=*(Uint16*) srcpix;
		      break;
		    case 3:
		      dstpix[0] = srcpix[0];
		      dstpix[1] = srcpix[1];
		      dstpix[2] = srcpix[2];    
		      break;
		    case 4:
		      *(Uint32*) dstpix=*(Uint32*) srcpix;
		      break;
		    }
		  dstpix+=dststepx;
		}
	      srcpix+=srcstepx;
	    }
	  dstrow+=dststepy;
	}
      srcrow+=srcstepy;
    }
  SDL_UnlockSurface(dst);
  return dst;
}


static PyObject* surf_chop(PyObject* self, PyObject* arg)
{
  PyObject *surfobj, *rectobj;
  SDL_Surface* surf, *newsurf;
  GAME_Rect* rect, temp;
	
  if(!PyArg_ParseTuple(arg, "O!O", &PySurface_Type, &surfobj, &rectobj))
    return NULL;
  if(!(rect = GameRect_FromObject(rectobj, &temp)))
    return RAISE(PyExc_TypeError, "Rect argument is invalid");

  surf=PySurface_AsSurface(surfobj);
  Py_BEGIN_ALLOW_THREADS
  newsurf=chop(surf, rect->x, rect->y, rect->w, rect->h);
  Py_END_ALLOW_THREADS

  return PySurface_New(newsurf);
}


static PyMethodDef transform_builtins[] =
{
	{ "scale", surf_scale, 1, DOC_PYGAMETRANSFORMSCALE },
	{ "rotate", surf_rotate, 1, DOC_PYGAMETRANSFORMROTATE },
	{ "flip", surf_flip, 1, DOC_PYGAMETRANSFORMFLIP },
	{ "rotozoom", surf_rotozoom, 1, DOC_PYGAMETRANSFORMROTOZOOM},
	{ "chop", surf_chop, 1, DOC_PYGAMETRANSFORMCHOP },
	{ "scale2x", surf_scale2x, 1, DOC_PYGAMETRANSFORMSCALE2X },
		
	{ NULL, NULL }
};



PYGAME_EXPORT
void inittransform(void)
{
	PyObject *module;
	module = Py_InitModule3("transform", transform_builtins, DOC_PYGAMETRANSFORM);

	/*imported needed apis*/
	import_pygame_base();
	import_pygame_rect();
	import_pygame_surface();
}


