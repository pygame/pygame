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
 *  drawing module for pygame
 */
#include "pygame.h"

static int clip_and_draw_line(SDL_Surface* surf, SDL_Rect* rect, Uint32 color, int* pts);
static int clipline(int* pts, int left, int top, int right, int bottom);
static void drawline(SDL_Surface* surf, Uint32 color, int startx, int starty, int endx, int endy);
static void drawhorzline(SDL_Surface* surf, Uint32 color, int startx, int starty, int endx);


    /*DOC*/ static char doc_line[] =
    /*DOC*/    "pygame.draw.line(Surface, color, startpos, endpos) -> Rect\n"
    /*DOC*/    "draw a line on a surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Draws a line on a surface. This will respect the clipping\n"
    /*DOC*/    "rectangle. A bounding box of the effected area is returned\n"
    /*DOC*/    "as a rectangle.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The color argument can be either a RGB sequence or mapped color integer.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function will temporarily lock the surface.\n"
    /*DOC*/ ;

static PyObject* line(PyObject* self, PyObject* arg)
{
	PyObject *surfobj, *colorobj, *start, *end;
	SDL_Surface* surf;
	short startx, starty, endx, endy;
	int top, left, bottom, right;
	int pts[4];
	Uint8 rgba[4];
	Uint32 color;
	int anydraw;

	/*get all the arguments*/
	if(!PyArg_ParseTuple(arg, "O!OOO", &PySurface_Type, &surfobj, &colorobj, &start, &end))
		return NULL;
	surf = PySurface_AsSurface(surfobj);

	if(surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
		return RAISE(PyExc_ValueError, "unsupport bit depth for line draw");

	if(PyInt_Check(colorobj))
		color = (Uint32)PyInt_AsLong(colorobj);
	else if(RGBAFromObj(colorobj, rgba))
		color = SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
	else
		return RAISE(PyExc_TypeError, "invalid color argument");

	if(!TwoShortsFromObj(start, &startx, &starty))
		return RAISE(PyExc_TypeError, "Invalid start position argument");
	if(!TwoShortsFromObj(end, &endx, &endy))
		return RAISE(PyExc_TypeError, "Invalid end position argument");
	

	if(!PySurface_Lock(surfobj)) return NULL;

	pts[0] = startx; pts[1] = starty;
	pts[2] = endx; pts[3] = endy;
	anydraw = clip_and_draw_line(surf, &surf->clip_rect, color, pts);

	if(!PySurface_Unlock(surfobj)) return NULL;


	/*compute return rect*/
	if(!anydraw)
		return PyRect_New4(startx, starty, 0, 0);
	if(pts[0] < pts[2])
	{
		left = pts[0];
		right = pts[2];
	}
	else
	{
		left = pts[2];
		right = pts[0];
	}
	if(pts[1] < pts[3])
	{
		top = pts[1];
		bottom = pts[3];
	}
	else
	{
		top = pts[3];
		bottom = pts[1];
	}
	return PyRect_New4((short)left, (short)top, (short)(right-left), (short)(bottom-top));
}


    /*DOC*/ static char doc_lines[] =
    /*DOC*/    "pygame.draw.lines(Surface, color, closed, point_array) -> Rect\n"
    /*DOC*/    "draw multiple connected lines on a surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Draws a sequence on a surface. You must pass at least two points\n"
    /*DOC*/    "in the sequence of points. The closed argument is a simple boolean\n"
    /*DOC*/    "and if true, a line will be draw between the first and last points.\n"
    /*DOC*/    "\n"
	/*DOC*/    "This will respect the clipping rectangle. A bounding box of the\n"
	/*DOC*/    "effected area is returned as a rectangle.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The color argument can be either a RGB sequence or mapped color integer.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function will temporarily lock the surface.\n"
    /*DOC*/ ;

static PyObject* lines(PyObject* self, PyObject* arg)
{
	PyObject *surfobj, *colorobj, *closedobj, *points, *item;
	SDL_Surface* surf;
	short x, y;
	int top, left, bottom, right;
	int pts[4];
	Uint8 rgba[4];
	Uint32 color;
	int closed;
	int result, loop, length, drawn;

	/*get all the arguments*/
	if(!PyArg_ParseTuple(arg, "O!OOO", &PySurface_Type, &surfobj, &colorobj, &closedobj, &points))
		return NULL;
	surf = PySurface_AsSurface(surfobj);

	if(surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
		return RAISE(PyExc_ValueError, "unsupport bit depth for line draw");

	if(PyInt_Check(colorobj))
		color = (Uint32)PyInt_AsLong(colorobj);
	else if(RGBAFromObj(colorobj, rgba))
		color = SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
	else
		return RAISE(PyExc_TypeError, "invalid color argument");

	closed = PyObject_IsTrue(closedobj);

	if(!PySequence_Check(points))
		return RAISE(PyExc_TypeError, "points argument must be a sequence of number pairs");
	length = PySequence_Length(points);
	if(length < 2)
		return RAISE(PyExc_ValueError, "points argument must contain more than 1 points");

	item = PySequence_GetItem(points, 0);
	result = TwoShortsFromObj(item, &x, &y);
	Py_DECREF(item);
	if(!result) return RAISE(PyExc_TypeError, "points must be number pairs");

	pts[0] = left = right = x;
	pts[1] = top = bottom = y;

	if(!PySurface_Lock(surfobj)) return NULL;

	drawn = 1;
	for(loop = 1; loop < length; ++loop)
	{
		item = PySequence_GetItem(points, loop);
		result = TwoShortsFromObj(item, &x, &y);
		Py_DECREF(item);
		if(!result) continue; /*note, we silently skip over bad points :[ */
		++drawn;
		pts[2] = x;
		pts[3] = y;

		if(clip_and_draw_line(surf, &surf->clip_rect, color, pts))
		{
			left = min(min(pts[0], pts[2]), left);
			top = min(min(pts[1], pts[3]), top);
			right = max(max(pts[0], pts[2]), right);
			bottom = max(max(pts[1], pts[3]), bottom);
		}

		pts[0] = pts[2];
		pts[1] = pts[3];
	}
	if(closed && drawn > 2)
	{
		item = PySequence_GetItem(points, 0);
		result = TwoShortsFromObj(item, &x, &y);
		Py_DECREF(item);
		if(result)
		{
			pts[2] = x;
			pts[3] = y;
			clip_and_draw_line(surf, &surf->clip_rect, color, pts);
		}
	}


	if(!PySurface_Unlock(surfobj)) return NULL;

	/*compute return rect*/
	return PyRect_New4((short)left, (short)top, (short)(right-left), (short)(bottom-top));
}







/*internal drawing tools*/

static int clip_and_draw_line(SDL_Surface* surf, SDL_Rect* rect, Uint32 color, int* pts)
{
	if(!clipline(pts, rect->x, rect->y, rect->x+rect->w-1, rect->y+rect->h-1))
		return 0;
	if(pts[1] == pts[3])
		drawhorzline(surf, color, pts[0], pts[1], pts[2]);
	else
		drawline(surf, color, pts[0], pts[1], pts[2], pts[3]);
	return 1;
}


/*this line clipping based heavily off of code from
http://www.ncsa.uiuc.edu/Vis/Graphics/src/clipCohSuth.c */
#define LEFT_EDGE   0x1
#define RIGHT_EDGE  0x2
#define BOTTOM_EDGE 0x4
#define TOP_EDGE    0x8
#define INSIDE(a)   (!a)
#define REJECT(a,b) (a&b)
#define ACCEPT(a,b) (!(a|b))

static int encode(int x, int y, int left, int top, int right, int bottom)
{
	int code = 0;
	if(x < left)  code |= LEFT_EDGE;
	if(x > right) code |= RIGHT_EDGE;
	if(y < top)   code |= TOP_EDGE;
	if(y > bottom)code |= BOTTOM_EDGE;
	return code;
}

static int clipline(int* pts, int left, int top, int right, int bottom)
{
	int x1 = pts[0];
	int y1 = pts[1];
	int x2 = pts[2];
	int y2 = pts[3];
	int code1, code2;
	int draw = 0;
	int swaptmp;
	float m; /*slope*/

	while(1)
	{
		code1 = encode(x1, y1, left, top, right, bottom);
		code2 = encode(x2, y2, left, top, right, bottom);
		if(ACCEPT(code1, code2)) {
			draw = 1;
			break;
		}
		else if(REJECT(code1, code2))
			break;
		else {
			if(INSIDE(code1)) {
				swaptmp = x2; x2 = x1; x1 = swaptmp;
				swaptmp = y2; y2 = y1; y1 = swaptmp;
				swaptmp = code2; code2 = code1; code1 = swaptmp;
			}
			if(x2 != x1)      
				m = (y2 - y1) / (float)(x2 - x1);
			else
				m = 1.0f;
			if(code1 & LEFT_EDGE) {
				y1 += (int)((left - x1) * m);
				x1 = left; 
			} 
			else if(code1 & RIGHT_EDGE) {
				y1 += (int)((right - x1) * m);
				x1 = right; 
			} 
			else if(code1 & BOTTOM_EDGE) {
				if(x2 != x1)
					x1 += (int)((bottom - y1) / m);
				y1 = bottom;
			} 
			else if(code1 & TOP_EDGE) {
				if(x2 != x1)
					x1 += (int)((top - y1) / m);
				y1 = top;
			} 
		}
	}
	if(draw) {
		pts[0] = x1; pts[1] = y1;
		pts[2] = x2; pts[3] = y2;
	}
	return draw;
}




/*here's my sdl'ized version of bresenham*/
static void drawline(SDL_Surface* surf, Uint32 color, int x1, int y1, int x2, int y2)
{
	int deltax, deltay, signx, signy;
	int pixx, pixy;
	int x = 0, y = 0;
	int swaptmp;
	Uint8 *pixel;
	Uint8 *colorptr;

 	deltax = x2 - x1;
 	deltay = y2 - y1;
 	signx = (deltax < 0) ? -1 : 1;
 	signy = (deltay < 0) ? -1 : 1;
 	deltax = signx * deltax + 1;
 	deltay = signy * deltay + 1;

	pixx = surf->format->BytesPerPixel;
	pixy = surf->pitch;
	pixel = ((Uint8*)surf->pixels) + pixx * x1 + pixy * y1;

	pixx *= signx;
	pixy *= signy;
	if(deltax < deltay) /*swap axis if rise > run*/
	{
		swaptmp = deltax; deltax = deltay; deltay = swaptmp;
		swaptmp = pixx; pixx = pixy; pixy = swaptmp;
	}

	switch(surf->format->BytesPerPixel)
	{
	case 1:
		for(; x < deltax; x++, pixel += pixx) {
			*pixel = (Uint8)color;
			y += deltay; if(y >= deltax) {y -= deltax; pixel += pixy;}
		}break;
	case 2:
		for(; x < deltax; x++, pixel += pixx) {
			*(Uint16*)pixel = (Uint16)color;
			y += deltay; if(y >= deltax) {y -= deltax; pixel += pixy;}
		}break;
	case 3:
		if(SDL_BYTEORDER == SDL_BIG_ENDIAN) color <<= 8;
		colorptr = (Uint8*)&color;
		for(; x < deltax; x++, pixel += pixx) {
			pixel[0] = colorptr[0];
			pixel[1] = colorptr[1];
			pixel[2] = colorptr[2];
			y += deltay; if(y >= deltax) {y -= deltax; pixel += pixy;}
		}break;
	default: /*case 4*/
		for(; x < deltax; x++, pixel += pixx) {
			*(Uint32*)pixel = (Uint32)color;
			y += deltay; if(y >= deltax) {y -= deltax; pixel += pixy;}
		}break;
	}
}



static void drawhorzline(SDL_Surface* surf, Uint32 color, int x1, int y1, int x2)
{
	Uint8 *pixel, *end;
	Uint8 *colorptr;

	if(x1 == x2) return;

	pixel = ((Uint8*)surf->pixels) + surf->pitch * y1;
	if(x1 < x2)
	{
		end = pixel + x2 * surf->format->BytesPerPixel;
		pixel += x1 * surf->format->BytesPerPixel;
	}
	else
	{
		end = pixel + x1 * surf->format->BytesPerPixel;
		pixel += x2 * surf->format->BytesPerPixel;
	}

	switch(surf->format->BytesPerPixel)
	{
	case 1:
		for(; pixel <= end; ++pixel) {
			*pixel = (Uint8)color;
		}break;
	case 2:
		for(; pixel <= end; ++pixel) {
			*(Uint16*)pixel = (Uint16)color;
		}break;
	case 3:
		if(SDL_BYTEORDER == SDL_BIG_ENDIAN) color <<= 8;
		colorptr = (Uint8*)&color;
		for(; pixel <= end; ++pixel) {
			pixel[0] = colorptr[0];
			pixel[1] = colorptr[1];
			pixel[2] = colorptr[2];
		}break;
	default: /*case 4*/
		for(; pixel <= end; ++pixel) {
			*(Uint32*)pixel = (Uint32)color;
		}break;
	}
}



static PyMethodDef draw_builtins[] =
{
	{ "line", line, 1, doc_line },
	{ "lines", lines, 1, doc_lines },

	{ NULL, NULL }
};



    /*DOC*/ static char doc_pygame_draw_MODULE[] =
    /*DOC*/    "Contains routines to draw onto a surface.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Note that all\n"
    /*DOC*/    "drawing routines use direct pixel access, so the surfaces\n"
    /*DOC*/    "must be locked for use. The draw functions will temporarily\n"
    /*DOC*/    "lock the surface if needed, but if performing many drawing\n"
    /*DOC*/    "routines together, it would be best to surround the drawing\n"
    /*DOC*/    "code with a lock/unlock pair.\n"
    /*DOC*/ ;

void initdraw(void)
{
	PyObject *module, *dict;

    /* create the module */
	module = Py_InitModule3("draw", draw_builtins, doc_pygame_draw_MODULE);
	dict = PyModule_GetDict(module);

	/*imported needed apis*/
	import_pygame_base();
	import_pygame_rect();
	import_pygame_surface();
}



#if 0
/*CODE I PLAN TO IMPLEMENT IN THE FUTURE*/
/*along with this ellipse stuff, a triangle would be keen,
from triangle a simple polygons will be easy too, but needed?*/

/*anyone feel free to contribute :] */
Ok... The circle algorithm:

        x = 0
        y = radius
        switch = 3 - 2*radius

    loop:

        plot (x,y): plot (x,-y): plot (-x,y): plot (-x,-y)
        plot (y,x): plot (y,-x): plot (-y,x): plot (-y,-x)

        if switch < 0 then
            switch = switch + 4 * x + 6
        else
            switch = switch + 4 * (x - y) + 10
            y = y - 1

        x = x + 1

        if x <= y then goto loop

        end



void symmetry(x,y)
int x,y;
{
 PUT_PIXEL(+x,+y);  // This would obviously be inlined!
 PUT_PIXEL(-x,+y);  // and offset by a constant amount
 PUT_PIXEL(-x,-y);
 PUT_PIXEL(+x,-y);
}

void bresenham_ellipse(a,b) /*a = xradius, b= yradius*/
int a,b;
{
 int x,y,a2,b2, S, T;

 a2 = a*a;
 b2 = b*b;
 x = 0;
 y = b;
 S = a2*(1-2*b) + 2*b2;
 T = b2 - 2*a2*(2*b-1);
 symmetry(x,y);
 do
   {
    if (S<0)
       {
        S += 2*b2*(2*x+3);
        T += 4*b2*(x+1);
        x++;
       }
      else if (T<0)
          {
           S += 2*b2*(2*x+3) - 4*a2*(y-1);
           T += 4*b2*(x+1) - 2*a2*(2*y-3);
           x++;
           y--;
          }
         else
          {
           S -= 4*a2*(y-1);
           T -= 2*a2*(2*y-3);
           y--;
          }
    symmetry(x,y);
   }
 while (y>0);
}
#endif
