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
#include <math.h>

#ifdef _MSC_VER
#pragma warning (disable:4244)

float trunc(float d)
{
    if (d >= 0)
        return floor(d);
    return ceil(d);
}

#endif

#define FRAC(z) (z-trunc(z))
#define INVFRAC(z) (1-(z-trunc(z)))

static int clip_and_draw_line(SDL_Surface* surf, SDL_Rect* rect, Uint32 color, int* pts);
static int clip_and_draw_aaline(SDL_Surface* surf, SDL_Rect* rect, Uint32 color, float* pts, int blend);
static int clip_and_draw_line_width(SDL_Surface* surf, SDL_Rect* rect, Uint32 color, int width, int* pts);
static int clipline(int* pts, int left, int top, int right, int bottom);
static int clipaaline(float* pts, int left, int top, int right, int bottom);
static void drawline(SDL_Surface* surf, Uint32 color, int startx, int starty, int endx, int endy);
static void drawaaline(SDL_Surface* surf, Uint32 color, float startx, float starty, float endx, float endy,
		int blend);
static void drawhorzline(SDL_Surface* surf, Uint32 color, int startx, int starty, int endx);
static void drawvertline(SDL_Surface* surf, Uint32 color, int x1, int y1, int y2);
static void draw_arc(SDL_Surface *dst, int x, int y, int radius1, int radius2, double angle_start, double angle_stop, Uint32 color);
static void draw_ellipse(SDL_Surface *dst, int x, int y, int rx, int ry, Uint32 color);
static void draw_fillellipse(SDL_Surface *dst, int x, int y, int rx, int ry, Uint32 color);
static void draw_fillpoly(SDL_Surface *dst, int *vx, int *vy, int n, Uint32 color);


    /*DOC*/ static char doc_aaline[] =
    /*DOC*/    "pygame.draw.aaline(Surface, color, startpos, endpos, blend=1) -> Rect\n"
    /*DOC*/    "draw a line on a surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Draws an anti-aliased line on a surface. This will respect the\n"
    /*DOC*/    "clipping rectangle. A bounding box of the effected area is returned\n"
    /*DOC*/    "returned as a rectangle.  The if blend is true, the shades will be\n"
    /*DOC*/    "be blended with existing pixel shades instead of overwriting them.\n"
    /*DOC*/    "This function accepts floating point values for the end points.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The color argument must be a RGB sequence.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function will temporarily lock the surface.\n"
    /*DOC*/ ;

static PyObject* aaline(PyObject* self, PyObject* arg)
{
	PyObject *surfobj, *colorobj, *start, *end;
	SDL_Surface* surf;
	float startx, starty, endx, endy;
	int top, left, bottom, right;
	int blend=1;
	float pts[4];
	Uint8 rgba[4];
	Uint32 color;
	int anydraw;

	/*get all the arguments*/
	if(!PyArg_ParseTuple(arg, "O!OOO|i", &PySurface_Type, &surfobj, &colorobj, &start, &end, &blend))
		return NULL;
	surf = PySurface_AsSurface(surfobj);

	if(surf->format->BytesPerPixel !=3 && surf->format->BytesPerPixel != 4)
		return RAISE(PyExc_ValueError, "unsupported bit depth for aaline draw (supports 32 & 24 bit)");

	if(RGBAFromObj(colorobj, rgba))
		color = SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
	else
		return RAISE(PyExc_TypeError, "invalid color argument");

	if(!TwoFloatsFromObj(start, &startx, &starty))
		return RAISE(PyExc_TypeError, "Invalid start position argument");
	if(!TwoFloatsFromObj(end, &endx, &endy))
		return RAISE(PyExc_TypeError, "Invalid end position argument");

	if(!PySurface_Lock(surfobj)) return NULL;

	pts[0] = startx; pts[1] = starty;
	pts[2] = endx; pts[3] = endy;
	anydraw = clip_and_draw_aaline(surf, &surf->clip_rect, color, pts, blend);

	if(!PySurface_Unlock(surfobj)) return NULL;

	/*compute return rect*/
	if(!anydraw)
		return PyRect_New4(startx, starty, 0, 0);
	if(pts[0] < pts[2])
	{
		left = (int)(pts[0]);
		right = (int)(pts[2]);
	}
	else
	{
		left = (int)(pts[2]);
		right = (int)(pts[0]);
	}
	if(pts[1] < pts[3])
	{
		top = (int)(pts[1]);
		bottom = (int)(pts[3]);
	}
	else
	{
		top = (int)(pts[3]);
		bottom = (int)(pts[1]);
	}
	return PyRect_New4(left, top, right-left+2, bottom-top+2);
}

    /*DOC*/ static char doc_line[] =
    /*DOC*/    "pygame.draw.line(Surface, color, startpos, endpos, width=1) -> Rect\n"
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
	int startx, starty, endx, endy;
	int top, left, bottom, right;
	int width = 1;
	int pts[4];
	Uint8 rgba[4];
	Uint32 color;
	int anydraw;

	/*get all the arguments*/
	if(!PyArg_ParseTuple(arg, "O!OOO|i", &PySurface_Type, &surfobj, &colorobj, &start, &end, &width))
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

	if(!TwoIntsFromObj(start, &startx, &starty))
		return RAISE(PyExc_TypeError, "Invalid start position argument");
	if(!TwoIntsFromObj(end, &endx, &endy))
		return RAISE(PyExc_TypeError, "Invalid end position argument");

	if(width < 1)
		return PyRect_New4(startx, starty, 0, 0);


	if(!PySurface_Lock(surfobj)) return NULL;

	pts[0] = startx; pts[1] = starty;
	pts[2] = endx; pts[3] = endy;
	anydraw = clip_and_draw_line_width(surf, &surf->clip_rect, color, width, pts);

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
	return PyRect_New4(left, top, right-left+1, bottom-top+1);
}


    /*DOC*/ static char doc_aalines[] =
    /*DOC*/    "pygame.draw.aalines(Surface, color, closed, point_array, blend=1) -> Rect\n"
    /*DOC*/    "draw multiple connected anti-aliased lines on a surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Draws a sequence on a surface. You must pass at least two points\n"
    /*DOC*/    "in the sequence of points. The closed argument is a simple boolean\n"
    /*DOC*/    "and if true, a line will be draw between the first and last points.\n"
    /*DOC*/    "The boolean blend argument set to true will blend the shades with\n"
    /*DOC*/    "existing shades instead of overwriting them.\n"
    /*DOC*/    "This function accepts floating point values for the end points.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will respect the clipping rectangle. A bounding box of the\n"
    /*DOC*/    "effected area is returned as a rectangle.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The color argument must be an RGB sequence.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function will temporarily lock the surface.\n"
    /*DOC*/ ;

static PyObject* aalines(PyObject* self, PyObject* arg)
{
	PyObject *surfobj, *colorobj, *closedobj, *points, *item;
	SDL_Surface* surf;
	float x, y;
	int top, left, bottom, right;
	float pts[4];
	Uint8 rgba[4];
	Uint32 color;
	int closed, blend;
	int result, loop, length, drawn;
	float startx, starty;

	/*get all the arguments*/
	if(!PyArg_ParseTuple(arg, "O!OOO|i", &PySurface_Type, &surfobj, &colorobj, &closedobj,
		&points, &blend))
		return NULL;
	surf = PySurface_AsSurface(surfobj);

	if(surf->format->BytesPerPixel !=3 && surf->format->BytesPerPixel != 4)
		return RAISE(PyExc_ValueError, "unsupported bit depth for aaline draw (supports 32 & 24 bit)");

	if(RGBAFromObj(colorobj, rgba))
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
	result = TwoFloatsFromObj(item, &x, &y);
	Py_DECREF(item);
	if(!result) return RAISE(PyExc_TypeError, "points must be number pairs");

	startx = pts[0] = x;
	starty = pts[1] = y;
	left = right = (int)x;
	top = bottom = (int)y;

	if(!PySurface_Lock(surfobj)) return NULL;

	drawn = 1;
	for(loop = 1; loop < length; ++loop)
	{
		item = PySequence_GetItem(points, loop);
		result = TwoFloatsFromObj(item, &x, &y);
		Py_DECREF(item);
		if(!result) continue; /*note, we silently skip over bad points :[ */
		++drawn;
		pts[0] = startx;
		pts[1] = starty;
		startx = pts[2] = x;
		starty = pts[3] = y;
		if(clip_and_draw_aaline(surf, &surf->clip_rect, color, pts, blend))
		{
			left = min((int)min(pts[0], pts[2]), left);
			top = min((int)min(pts[1], pts[3]), top);
			right = max((int)max(pts[0], pts[2]), right);
			bottom = max((int)max(pts[1], pts[3]), bottom);
		}
	}
	if(closed && drawn > 2)
	{
		item = PySequence_GetItem(points, 0);
		result = TwoFloatsFromObj(item, &x, &y);
		Py_DECREF(item);
		if(result)
		{
			pts[0] = startx;
			pts[1] = starty;
			pts[2] = x;
			pts[3] = y;
			clip_and_draw_aaline(surf, &surf->clip_rect, color, pts, blend);
		}
	}

	if(!PySurface_Unlock(surfobj)) return NULL;

	/*compute return rect*/
	return PyRect_New4(left, top, right-left+2, bottom-top+2);
}

    /*DOC*/ static char doc_lines[] =
    /*DOC*/    "pygame.draw.lines(Surface, color, closed, point_array, width=1) -> Rect\n"
    /*DOC*/    "draw multiple connected lines on a surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Draws a sequence on a surface. You must pass at least two points\n"
    /*DOC*/    "in the sequence of points. The closed argument is a simple boolean\n"
    /*DOC*/    "and if true, a line will be draw between the first and last points.\n"
    /*DOC*/    "Note that specifying a linewidth wider than 1 does not fill in the\n"
    /*DOC*/    "gaps between the lines. Therefore wide lines and sharp corners won't\n"
    /*DOC*/    "be joined seamlessly.\n"
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
	int x, y;
	int top, left, bottom, right;
	int pts[4], width=1;
	Uint8 rgba[4];
	Uint32 color;
	int closed;
	int result, loop, length, drawn;
	int startx, starty;

	/*get all the arguments*/
	if(!PyArg_ParseTuple(arg, "O!OOO|i", &PySurface_Type, &surfobj, &colorobj, &closedobj, &points, &width))
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
	result = TwoIntsFromObj(item, &x, &y);
	Py_DECREF(item);
	if(!result) return RAISE(PyExc_TypeError, "points must be number pairs");

	startx = pts[0] = left = right = x;
	starty = pts[1] = top = bottom = y;

	if(width < 1)
		return PyRect_New4(left, top, 0, 0);

	if(!PySurface_Lock(surfobj)) return NULL;

	drawn = 1;
	for(loop = 1; loop < length; ++loop)
	{
		item = PySequence_GetItem(points, loop);
		result = TwoIntsFromObj(item, &x, &y);
		Py_DECREF(item);
		if(!result) continue; /*note, we silently skip over bad points :[ */
		++drawn;
		pts[0] = startx;
		pts[1] = starty;
		startx = pts[2] = x;
		starty = pts[3] = y;
		if(clip_and_draw_line_width(surf, &surf->clip_rect, color, width, pts))
		{
			left = min(min(pts[0], pts[2]), left);
			top = min(min(pts[1], pts[3]), top);
			right = max(max(pts[0], pts[2]), right);
			bottom = max(max(pts[1], pts[3]), bottom);
		}
	}
	if(closed && drawn > 2)
	{
		item = PySequence_GetItem(points, 0);
		result = TwoIntsFromObj(item, &x, &y);
		Py_DECREF(item);
		if(result)
		{
			pts[0] = startx;
			pts[1] = starty;
			pts[2] = x;
			pts[3] = y;
			clip_and_draw_line_width(surf, &surf->clip_rect, color, width, pts);
		}
	}


	if(!PySurface_Unlock(surfobj)) return NULL;

	/*compute return rect*/
	return PyRect_New4(left, top, right-left+1, bottom-top+1);
}


    /*DOC*/ static char doc_arc[] =
    /*DOC*/    "pygame.draw.arc(Surface, color, Rect, angle_start, angle_stop, width=0) -> Rect\n"
    /*DOC*/    "draw an elliptic arc on a surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Draws an elliptical arc on the Surface. The given rectangle\n"
    /*DOC*/    "is the area that the circle will fill. The two angle arguments\n"
    /*DOC*/    "are the initial and final angle (radians, with the zero on the right).\n"
    /*DOC*/    "The width argument is the thickness to draw the outer edge.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The color argument can be either a RGB sequence or mapped color integer.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function will temporarily lock the surface.\n"
    /*DOC*/ ;

static PyObject* arc(PyObject* self, PyObject* arg)
{
	PyObject *surfobj, *colorobj, *rectobj;
	GAME_Rect *rect, temp;
	SDL_Surface* surf;
	Uint8 rgba[4];
	Uint32 color;
	int width=1, loop, t, l, b, r;
	double angle_start, angle_stop;

	/*get all the arguments*/
	if(!PyArg_ParseTuple(arg, "O!OOdd|i", &PySurface_Type, &surfobj, &colorobj, &rectobj,
			                      &angle_start, &angle_stop, &width))
		return NULL;
	rect = GameRect_FromObject(rectobj, &temp);
	if(!rect)
		return RAISE(PyExc_TypeError, "Invalid recstyle argument");

	surf = PySurface_AsSurface(surfobj);
	if(surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
		return RAISE(PyExc_ValueError, "unsupport bit depth for drawing");

	if(PyInt_Check(colorobj))
		color = (Uint32)PyInt_AsLong(colorobj);
	else if(RGBAFromObj(colorobj, rgba))
		color = SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
	else
		return RAISE(PyExc_TypeError, "invalid color argument");

	if ( width < 0 )
		return RAISE(PyExc_ValueError, "negative width");
	if ( width > rect->w / 2 || width > rect->h / 2 )
		return RAISE(PyExc_ValueError, "width greater than ellipse radius");
	if ( angle_stop < angle_start )
		angle_stop += 360;

	if(!PySurface_Lock(surfobj)) return NULL;

	width = min(width, min(rect->w, rect->h) / 2);
	for(loop=0; loop<width; ++loop)
	{
		draw_arc(surf, rect->x+rect->w/2, rect->y+rect->h/2,
			 rect->w/2-loop, rect->h/2-loop,
			 angle_start, angle_stop, color);
	}

	if(!PySurface_Unlock(surfobj)) return NULL;

	l = max(rect->x, surf->clip_rect.x);
	t = max(rect->y, surf->clip_rect.y);
	r = min(rect->x + rect->w, surf->clip_rect.x + surf->clip_rect.w);
	b = min(rect->y + rect->h, surf->clip_rect.y + surf->clip_rect.h);
	return PyRect_New4(l, t, max(r-l, 0), max(b-t, 0));
}



    /*DOC*/ static char doc_ellipse[] =
    /*DOC*/    "pygame.draw.ellipse(Surface, color, Rect, width=0) -> Rect\n"
    /*DOC*/    "draw an ellipse on a surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Draws an elliptical shape on the Surface. The given rectangle\n"
    /*DOC*/    "is the area that the circle will fill. The width argument is\n"
    /*DOC*/    "the thickness to draw the outer edge. If width is zero then\n"
    /*DOC*/    "the ellipse will be filled.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The color argument can be either a RGB sequence or mapped color integer.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function will temporarily lock the surface.\n"
    /*DOC*/ ;

static PyObject* ellipse(PyObject* self, PyObject* arg)
{
	PyObject *surfobj, *colorobj, *rectobj;
	GAME_Rect *rect, temp;
	SDL_Surface* surf;
	Uint8 rgba[4];
	Uint32 color;
	int width=0, loop, t, l, b, r;

	/*get all the arguments*/
	if(!PyArg_ParseTuple(arg, "O!OO|i", &PySurface_Type, &surfobj, &colorobj, &rectobj, &width))
		return NULL;
	rect = GameRect_FromObject(rectobj, &temp);
	if(!rect)
		return RAISE(PyExc_TypeError, "Invalid recstyle argument");

	surf = PySurface_AsSurface(surfobj);
	if(surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
		return RAISE(PyExc_ValueError, "unsupport bit depth for drawing");

	if(PyInt_Check(colorobj))
		color = (Uint32)PyInt_AsLong(colorobj);
	else if(RGBAFromObj(colorobj, rgba))
		color = SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
	else
		return RAISE(PyExc_TypeError, "invalid color argument");

	if ( width < 0 )
		return RAISE(PyExc_ValueError, "negative width");
	if ( width > rect->w / 2 || width > rect->h / 2 )
		return RAISE(PyExc_ValueError, "width greater than ellipse radius");

	if(!PySurface_Lock(surfobj)) return NULL;

	if(!width)
		draw_fillellipse(surf, (Sint16)(rect->x+rect->w/2), (Sint16)(rect->y+rect->h/2),
					(Sint16)(rect->w/2), (Sint16)(rect->h/2), color);
	else
	{
		width = min(width, min(rect->w, rect->h) / 2);
		for(loop=0; loop<width; ++loop)
		{
			draw_ellipse(surf, rect->x+rect->w/2, rect->y+rect->h/2,
						rect->w/2-loop, rect->h/2-loop, color);
		}
	}

	if(!PySurface_Unlock(surfobj)) return NULL;

	l = max(rect->x, surf->clip_rect.x);
	t = max(rect->y, surf->clip_rect.y);
	r = min(rect->x + rect->w, surf->clip_rect.x + surf->clip_rect.w);
	b = min(rect->y + rect->h, surf->clip_rect.y + surf->clip_rect.h);
	return PyRect_New4(l, t, max(r-l, 0), max(b-t, 0));
}



    /*DOC*/ static char doc_circle[] =
    /*DOC*/    "pygame.draw.circle(Surface, color, pos, radius, width=0) -> Rect\n"
    /*DOC*/    "draw a circle on a surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Draws a circular shape on the Surface. The given position\n"
    /*DOC*/    "is the center of the circle, and radius is the size. The width\n"
    /*DOC*/    "argument is the thickness to draw the outer edge. If width is\n"
    /*DOC*/    "zero then the circle will be filled.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The color argument can be either a RGB sequence or mapped color integer.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function will temporarily lock the surface.\n"
    /*DOC*/ ;

static PyObject* circle(PyObject* self, PyObject* arg)
{
	PyObject *surfobj, *colorobj;
	SDL_Surface* surf;
	Uint8 rgba[4];
	Uint32 color;
	int posx, posy, radius, t, l, b, r;
	int width=0, loop;

	/*get all the arguments*/
	if(!PyArg_ParseTuple(arg, "O!O(ii)i|i", &PySurface_Type, &surfobj, &colorobj, &posx, &posy, &radius, &width))
		return NULL;

	surf = PySurface_AsSurface(surfobj);
	if(surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
		return RAISE(PyExc_ValueError, "unsupport bit depth for drawing");

	if(PyInt_Check(colorobj))
		color = (Uint32)PyInt_AsLong(colorobj);
	else if(RGBAFromObj(colorobj, rgba))
		color = SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
	else
		return RAISE(PyExc_TypeError, "invalid color argument");

	if ( radius < 0 )
		return RAISE(PyExc_ValueError, "negative radius");
	if ( width < 0 )
		return RAISE(PyExc_ValueError, "negative width");
	if ( width > radius )
		return RAISE(PyExc_ValueError, "width greater than radius");


	if(!PySurface_Lock(surfobj)) return NULL;

	if(!width)
		draw_fillellipse(surf, (Sint16)posx, (Sint16)posy, (Sint16)radius, (Sint16)radius, color);
	else
		for(loop=0; loop<width; ++loop)
			draw_ellipse(surf, posx, posy, radius-loop, radius-loop, color);

	if(!PySurface_Unlock(surfobj)) return NULL;

	l = max(posx - radius, surf->clip_rect.x);
	t = max(posy - radius, surf->clip_rect.y);
	r = min(posx + radius, surf->clip_rect.x + surf->clip_rect.w);
	b = min(posy + radius, surf->clip_rect.y + surf->clip_rect.h);
	return PyRect_New4(l, t, max(r-l, 0), max(b-t, 0));
}




    /*DOC*/ static char doc_polygon[] =
    /*DOC*/    "pygame.draw.polygon(Surface, color, pointslist, width=0) -> Rect\n"
    /*DOC*/    "draws a polygon on a surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Draws a polygonal shape on the Surface. The given pointlist\n"
    /*DOC*/    "is the vertices of the polygon. The width argument is\n"
    /*DOC*/    "the thickness to draw the outer edge. If width is zero then\n"
    /*DOC*/    "the polygon will be filled.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The color argument can be either a RGB sequence or mapped color integer.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function will temporarily lock the surface.\n"
    /*DOC*/ ;

static PyObject* polygon(PyObject* self, PyObject* arg)
{
	PyObject *surfobj, *colorobj, *points, *item;
	SDL_Surface* surf;
	Uint8 rgba[4];
	Uint32 color;
	int width=0, length, loop, numpoints;
	int *xlist, *ylist;
	int x, y, top, left, bottom, right, result;

	/*get all the arguments*/
	if(!PyArg_ParseTuple(arg, "O!OO|i", &PySurface_Type, &surfobj, &colorobj, &points, &width))
		return NULL;


	if(width)
	{
		PyObject *args, *ret;
		args = Py_BuildValue("(OOiOi)", surfobj, colorobj, 1, points, width);
		if(!args) return NULL;
		ret = lines(NULL, args);
		Py_DECREF(args);
		return ret;
	}


	surf = PySurface_AsSurface(surfobj);

	if(surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
		return RAISE(PyExc_ValueError, "unsupport bit depth for line draw");

	if(PyInt_Check(colorobj))
		color = (Uint32)PyInt_AsLong(colorobj);
	else if(RGBAFromObj(colorobj, rgba))
		color = SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
	else
		return RAISE(PyExc_TypeError, "invalid color argument");

	if(!PySequence_Check(points))
		return RAISE(PyExc_TypeError, "points argument must be a sequence of number pairs");
	length = PySequence_Length(points);
	if(length < 3)
		return RAISE(PyExc_ValueError, "points argument must contain more than 2 points");


	item = PySequence_GetItem(points, 0);
	result = TwoIntsFromObj(item, &x, &y);
	Py_DECREF(item);
	if(!result) return RAISE(PyExc_TypeError, "points must be number pairs");
	left = right = x;
	top = bottom = y;

	xlist = PyMem_New(int, length);
	ylist = PyMem_New(int, length);

	numpoints = 0;
	for(loop = 0; loop < length; ++loop)
	{
		item = PySequence_GetItem(points, loop);
		result = TwoIntsFromObj(item, &x, &y);
		Py_DECREF(item);
		if(!result) continue; /*note, we silently skip over bad points :[ */
		xlist[numpoints] = x;
		ylist[numpoints] = y;
		++numpoints;
		left = min(x, left);
		top = min(y, top);
		right = max(x, right);
		bottom = max(y, bottom);
	}

	if(!PySurface_Lock(surfobj))
	{
		PyMem_Del(xlist); PyMem_Del(ylist);
		return NULL;
	}

	draw_fillpoly(surf, xlist, ylist, numpoints, color);

	PyMem_Del(xlist); PyMem_Del(ylist);
	if(!PySurface_Unlock(surfobj))
		return NULL;

	left = max(left, surf->clip_rect.x);
	top = max(top, surf->clip_rect.y);
	right = min(right, surf->clip_rect.x + surf->clip_rect.w);
	bottom = min(bottom, surf->clip_rect.y + surf->clip_rect.h);
	return PyRect_New4(left, top, right-left+1, bottom-top+1);
}


    /*DOC*/ static char doc_rect[] =
    /*DOC*/    "pygame.draw.rect(Surface, color, Rect, width=0) -> Rect\n"
    /*DOC*/    "draws a rectangle on a surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Draws a rectangular shape on the Surface. The given Rect\n"
    /*DOC*/    "is the area of the rectangle. The width argument is\n"
    /*DOC*/    "the thickness to draw the outer edge. If width is zero then\n"
    /*DOC*/    "the rectangle will be filled.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The color argument can be either a RGB sequence or mapped color integer.\n"
    /*DOC*/    "\n"
    /*DOC*/    "This function will temporarily lock the surface.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Keep in mind the Surface.fill() method works just as well\n"
    /*DOC*/    "for drawing filled rectangles. In fact the Surface.fill()\n"
    /*DOC*/    "can be hardware accelerated when the moons are in alignement.\n"
    /*DOC*/ ;

static PyObject* rect(PyObject* self, PyObject* arg)
{
	PyObject *surfobj, *colorobj, *rectobj, *points, *args, *ret=NULL;
	GAME_Rect* rect, temp;
	int t, l, b, r, width=0;

	/*get all the arguments*/
	if(!PyArg_ParseTuple(arg, "O!OO|i", &PySurface_Type, &surfobj, &colorobj, &rectobj, &width))
		return NULL;

	if(!(rect = GameRect_FromObject(rectobj, &temp)))
		return RAISE(PyExc_TypeError, "Rect argument is invalid");

	l = rect->x; r = rect->x + rect->w - 1;
	t = rect->y; b = rect->y + rect->h - 1;

	/*build the pointlist*/
	points = Py_BuildValue("((ii)(ii)(ii)(ii))", l, t, r, t, r, b, l, b);

	args = Py_BuildValue("(OOOi)", surfobj, colorobj, points, width);
	if(args) ret = polygon(NULL, args);

	Py_XDECREF(args);
	Py_XDECREF(points);
	return ret;
}




/*internal drawing tools*/

static int clip_and_draw_aaline(SDL_Surface* surf, SDL_Rect* rect, Uint32 color, float* pts, int blend)
{
	if(!clipaaline(pts, rect->x+1, rect->y+1, rect->x+rect->w-2, rect->y+rect->h-2))
		return 0;
	drawaaline(surf, color, pts[0], pts[1], pts[2], pts[3], blend);
	return 1;
}

static int clip_and_draw_line(SDL_Surface* surf, SDL_Rect* rect, Uint32 color, int* pts)
{
	if(!clipline(pts, rect->x, rect->y, rect->x+rect->w-1, rect->y+rect->h-1))
		return 0;
	if(pts[1] == pts[3])
		drawhorzline(surf, color, pts[0], pts[1], pts[2]);
	else if(pts[0] == pts[2])
		drawvertline(surf, color, pts[0], pts[1], pts[3]);
	else
		drawline(surf, color, pts[0], pts[1], pts[2], pts[3]);
	return 1;
}

static int clip_and_draw_line_width(SDL_Surface* surf, SDL_Rect* rect, Uint32 color, int width, int* pts)
{
	int loop;
	int xinc=0, yinc=0;
	int newpts[4];
	int range[4];
	int anydrawn = 0;

	if(abs(pts[0]-pts[2]) > abs(pts[1]-pts[3]))
		yinc = 1;
	else
		xinc = 1;

	memcpy(newpts, pts, sizeof(int)*4);
	if(clip_and_draw_line(surf, rect, color, newpts))
	{
		anydrawn = 1;
		memcpy(range, newpts, sizeof(int)*4);
	}
	else
	{
		range[0] = range[1] = 10000;
		range[2] = range[3] = -10000;
	}

	for(loop = 1; loop < width; loop += 2)
	{
		newpts[0] = pts[0] + xinc*(loop/2+1);
		newpts[1] = pts[1] + yinc*(loop/2+1);
		newpts[2] = pts[2] + xinc*(loop/2+1);
		newpts[3] = pts[3] + yinc*(loop/2+1);
		if(clip_and_draw_line(surf, rect, color, newpts))
		{
			anydrawn = 1;
			range[0] = min(newpts[0], range[0]);
			range[1] = min(newpts[1], range[1]);
			range[2] = max(newpts[2], range[2]);
			range[3] = max(newpts[3], range[3]);
		}
		if(loop+1<width)
		{
			newpts[0] = pts[0] - xinc*(loop/2+1);
			newpts[1] = pts[1] - yinc*(loop/2+1);
			newpts[2] = pts[2] - xinc*(loop/2+1);
			newpts[3] = pts[3] - yinc*(loop/2+1);
			if(clip_and_draw_line(surf, rect, color, newpts))
			{
				anydrawn = 1;
				range[0] = min(newpts[0], range[0]);
				range[1] = min(newpts[1], range[1]);
				range[2] = max(newpts[2], range[2]);
				range[3] = max(newpts[3], range[3]);
			}
		}
	}
	if(anydrawn)
		memcpy(pts, range, sizeof(int)*4);
	return anydrawn;
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

static int encodeFloat(float x, float y, int left, int top, int right, int bottom)
{
	int code = 0;
	if(x < left)  code |= LEFT_EDGE;
	if(x > right) code |= RIGHT_EDGE;
	if(y < top)   code |= TOP_EDGE;
	if(y > bottom)code |= BOTTOM_EDGE;
	return code;
}

static int clipaaline(float* pts, int left, int top, int right, int bottom)
{
	float x1 = pts[0];
	float y1 = pts[1];
	float x2 = pts[2];
	float y2 = pts[3];
	int code1, code2;
	int draw = 0;
	float swaptmp;
	int intswaptmp;
	float m; /*slope*/

	while(1) {
		code1 = encodeFloat(x1, y1, left, top, right, bottom);
		code2 = encodeFloat(x2, y2, left, top, right, bottom);
		if(ACCEPT(code1, code2)) {
			draw = 1;
			break;
		}
		else if(REJECT(code1, code2)) {
			break;
		}
		else {
			if(INSIDE(code1)) {
				swaptmp = x2; x2 = x1; x1 = swaptmp;
				swaptmp = y2; y2 = y1; y1 = swaptmp;
				intswaptmp = code2; code2 = code1; code1 = intswaptmp;
			}
			if(x2 != x1)
				m = (y2 - y1) / (x2 - x1);
			else
				m = 1.0f;
			if(code1 & LEFT_EDGE) {
				y1 += ((float)left - x1) * m;
				x1 = (float)left;
			}
			else if(code1 & RIGHT_EDGE) {
				y1 += ((float)right - x1) * m;
				x1 = (float)right;
			}
			else if(code1 & BOTTOM_EDGE) {
				if(x2 != x1)
					x1 += ((float)bottom - y1) / m;
				y1 = (float)bottom;
			}
			else if(code1 & TOP_EDGE) {
				if(x2 != x1)
					x1 += ((float)top - y1) / m;
				y1 = (float)top;
			}
		}
	}
	if(draw) {
		pts[0] = x1; pts[1] = y1;
		pts[2] = x2; pts[3] = y2;
	}
	return draw;
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



static int set_at(SDL_Surface* surf, int x, int y, Uint32 color)
{
	SDL_PixelFormat* format = surf->format;
	Uint8* pixels = (Uint8*)surf->pixels;
	Uint8* byte_buf, rgb[4];

	if(x < surf->clip_rect.x || x >= surf->clip_rect.x + surf->clip_rect.w ||
				y < surf->clip_rect.y || y >= surf->clip_rect.y + surf->clip_rect.h)
	return 0;

	switch(format->BytesPerPixel)
	{
		case 1:
			*((Uint8*)pixels + y * surf->pitch + x) = (Uint8)color;
			break;
		case 2:
			*((Uint16*)(pixels + y * surf->pitch) + x) = (Uint16)color;
			break;
		case 4:
			*((Uint32*)(pixels + y * surf->pitch) + x) = color;
/*			  *((Uint32*)(pixels + y * surf->pitch) + x) =
				 ~(*((Uint32*)(pixels + y * surf->pitch) + x)) * 31;
*/			  break;
		default:/*case 3:*/
			SDL_GetRGB(color, format, rgb, rgb+1, rgb+2);
			byte_buf = (Uint8*)(pixels + y * surf->pitch) + x * 3;
			*(byte_buf + (format->Rshift >> 3)) = rgb[0];
			*(byte_buf + (format->Gshift >> 3)) = rgb[1];
			*(byte_buf + (format->Bshift >> 3)) = rgb[2];
			break;
	}
	return 1;
}


#define DRAWPIX32(pixel,colorptr,br,blend) \
	if(SDL_BYTEORDER == SDL_BIG_ENDIAN) color <<= 8; \
	if(blend) { \
		int x; \
		x = colorptr[0]*br+pixel[0]; \
		pixel[0]= (x>254) ? 255: x; \
		x = colorptr[1]*br+pixel[1]; \
		pixel[1]= (x>254) ? 255: x; \
		x = colorptr[2]*br+pixel[2]; \
		pixel[2]= (x>254) ? 255: x; \
	} else { \
		pixel[0]=(Uint8)(colorptr[0]*br); \
		pixel[1]=(Uint8)(colorptr[1]*br); \
		pixel[2]=(Uint8)(colorptr[2]*br); \
	}

/* Adapted from http://freespace.virgin.net/hugo.elias/graphics/x_wuline.htm */
static void drawaaline(SDL_Surface* surf, Uint32 color, float x1, float y1, float x2, float y2, int blend) {
	float grad, xd, yd;
   	float xgap, ygap, xend, yend, xf, yf;
	float brightness1, brightness2;
	float swaptmp;
	int x, y, ix1, ix2, iy1, iy2;
	int pixx, pixy;
	Uint8* pixel;
	Uint8* pm = (Uint8*)surf->pixels;
	Uint8* colorptr = (Uint8*)&color;

	pixx = surf->format->BytesPerPixel;
	pixy = surf->pitch;

	xd = x2-x1;
	yd = y2-y1;
	if(fabs(xd)>fabs(yd)) {
		if(x1>x2) {
			swaptmp=x1; x1=x2; x2=swaptmp;
			swaptmp=y1; y1=y2; y2=swaptmp;
			xd = (x2-x1);
			yd = (y2-y1);
		}
		grad = yd/xd;
		xend = trunc(x1)+0.5; /* This makes more sense than trunc(x1+0.5) */
		yend = y1+grad*(xend-x1);
		xgap = INVFRAC(x1);
		ix1 = (int)xend;
		iy1 = (int)yend;
		yf = yend+grad;
		brightness1 = INVFRAC(yend) * xgap;
		brightness2 =    FRAC(yend) * xgap;
		pixel = pm + pixx * ix1 + pixy * iy1;
		DRAWPIX32(pixel, colorptr, brightness1, blend)
		pixel += pixy;
		DRAWPIX32(pixel, colorptr, brightness2, blend)
		xend = trunc(x2)+0.5;
		yend = y2+grad*(xend-x2);
		xgap =    FRAC(x2); /* this also differs from Hugo's description. */
		ix2 = (int)xend;
		iy2 = (int)yend;
		brightness1 = INVFRAC(yend) * xgap;
		brightness2 =    FRAC(yend) * xgap;
		pixel = pm + pixx * ix2 + pixy * iy2;
		DRAWPIX32(pixel, colorptr, brightness1, blend)
		pixel += pixy;
		DRAWPIX32(pixel, colorptr, brightness2, blend)
		for(x=ix1+1; x<ix2; ++x) {
			brightness1=INVFRAC(yf);
			brightness2=   FRAC(yf);
			pixel = pm + pixx * x + pixy * (int)yf;
			DRAWPIX32(pixel, colorptr, brightness1, blend)
			pixel += pixy;
			DRAWPIX32(pixel, colorptr, brightness2, blend)
			yf += grad;
		}
	}
	else {
		if(y1>y2) {
			swaptmp=y1; y1=y2; y2=swaptmp;
			swaptmp=x1; x1=x2; x2=swaptmp;
			yd = (y2-y1);
			xd = (x2-x1);
		}
		grad = xd/yd;
		yend = trunc(y1)+0.5;  /* This makes more sense than trunc(x1+0.5) */
		xend = x1+grad*(yend-y1);
		ygap = INVFRAC(y1);
		iy1 = (int)yend;
		ix1 = (int)xend;
		xf = xend+grad;
		brightness1 = INVFRAC(xend) * ygap;
		brightness2 =    FRAC(xend) * ygap;
		pixel = pm + pixx * ix1 + pixy * iy1;
		DRAWPIX32(pixel, colorptr, brightness1, blend)
		pixel += pixx;
		DRAWPIX32(pixel, colorptr, brightness2, blend)
		yend = trunc(y2)+0.5;
		xend = x2+grad*(yend-y2);
		ygap = FRAC(y2);
		iy2 = (int)yend;
		ix2 = (int)xend;
		brightness1 = INVFRAC(xend) * ygap;
		brightness2 =    FRAC(xend) * ygap;
		pixel = pm + pixx * ix2 + pixy * iy2;
		DRAWPIX32(pixel, colorptr, brightness1, blend)
		pixel += pixx;
		DRAWPIX32(pixel, colorptr, brightness2, blend)
		for(y=iy1+1; y<iy2; ++y) {
			brightness1=INVFRAC(xf);
			brightness2=   FRAC(xf);
			pixel = pm + pixx * (int)xf + pixy * y;
			DRAWPIX32(pixel, colorptr, brightness1, blend)
			pixel += pixx;
			DRAWPIX32(pixel, colorptr, brightness2, blend)
			xf += grad;
		}
	}
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

	if(x1 == x2)
        {
            set_at(surf, x1, y1, color);
            return;
        }

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
		for(; pixel <= end; pixel+=2) {
			*(Uint16*)pixel = (Uint16)color;
		}break;
	case 3:
		if(SDL_BYTEORDER == SDL_BIG_ENDIAN) color <<= 8;
		colorptr = (Uint8*)&color;
		for(; pixel <= end; pixel+=3) {
			pixel[0] = colorptr[0];
			pixel[1] = colorptr[1];
			pixel[2] = colorptr[2];
		}break;
	default: /*case 4*/
		for(; pixel <= end; pixel+=4) {
			*(Uint32*)pixel = color;
		  }break;
	}
}

static void drawhorzlineclip(SDL_Surface* surf, Uint32 color, int x1, int y1, int x2)
{
	if(y1 < surf->clip_rect.y || y1 >= surf->clip_rect.y + surf->clip_rect.h)
		return;

	if( x2 < x1 )
	{
                int temp = x1;
                x1 = x2; x2 = temp;
	}

	x1 = max(x1, surf->clip_rect.x);
	x2 = min(x2, surf->clip_rect.x + surf->clip_rect.w-1);

        if(x2 < surf->clip_rect.x || x1 >= surf->clip_rect.x + surf->clip_rect.w)
                return;

	if(x1 == x2)
		set_at(surf, x1, y1, color);
	else
		drawhorzline(surf, color, x1, y1, x2);
}

static void drawvertline(SDL_Surface* surf, Uint32 color, int x1, int y1, int y2)
{
       Uint8   *pixel, *end;
       Uint8   *colorptr;
       Uint32  pitch = surf->pitch;

       if(y1 == y2)
       {
           set_at(surf, x1, y1, color);
           return;
       }

       pixel = ((Uint8*)surf->pixels) + x1 * surf->format->BytesPerPixel;
       if(y1 < y2)
       {
	   end	  = pixel + surf->pitch * y2;
	   pixel += surf->pitch * y1;
       }
       else
       {
	   end	  = pixel + surf->pitch * y1;
	   pixel += surf->pitch * y2;
       }

       switch(surf->format->BytesPerPixel)
       {
       case 1:
	       for(; pixel <= end; pixel+=pitch) {
		       *pixel = (Uint8)color;
	       }break;
       case 2:
	       for(; pixel <= end; pixel+=pitch) {
		       *(Uint16*)pixel = (Uint16)color;
	       }break;
       case 3:
	       if(SDL_BYTEORDER == SDL_BIG_ENDIAN) color <<= 8;
	       colorptr = (Uint8*)&color;
	       for(; pixel <= end; pixel+=pitch) {
		       pixel[0] = colorptr[0];
		       pixel[1] = colorptr[1];
		       pixel[2] = colorptr[2];
	       }break;
       default: /*case 4*/
	       for(; pixel <= end; pixel+=pitch) {
		       *(Uint32*)pixel = color;
	       }break;
       }
}


static void drawvertlineclip(SDL_Surface* surf, Uint32 color, int x1, int y1, int y2)
{
	if(x1 < surf->clip_rect.x || x1 >= surf->clip_rect.x + surf->clip_rect.w)
	return;
	if ( y2 < y1 )
	{
		int temp = y1;
		y1 = y2; y2 = temp;
	}
	y1 = max(y1, surf->clip_rect.y);
	y2 = min(y2, surf->clip_rect.y + surf->clip_rect.h-1);
	if(y2 - y1 < 1)
		set_at( surf, x1, y1, color);
	else
		drawvertline(surf, color, x1, y1, y2);
}

static void draw_arc(SDL_Surface *dst, int x, int y, int radius1, int radius2,
                     double angle_start, double angle_stop, Uint32 color)
{
    double aStep;            // Angle Step (rad)
    double a;                // Current Angle (rad)
    int x_last, x_next, y_last, y_next;

    // Angle step in rad
    if (radius1<radius2) {
        if (radius1<1.0e-4) {
            aStep=1.0;
        } else {
            aStep=asin(2.0/radius1);
        }
    } else {
        if (radius2<1.0e-4) {
            aStep=1.0;
        } else {
            aStep=asin(2.0/radius2);
        }
    }

    if(aStep<0.05) {
        aStep = 0.05;
    }

    x_last = x+cos(angle_start)*radius1;
    y_last = y-sin(angle_start)*radius2;
    for(a=angle_start+aStep; a<=angle_stop; a+=aStep) {
      x_next = x+cos(a)*radius1;
      y_next = y-sin(a)*radius2;
      drawline(dst, color, x_last, y_last, x_next, y_next);
      x_last = x_next;
      y_last = y_next;
    }
}

static void draw_ellipse(SDL_Surface *dst, int x, int y, int rx, int ry, Uint32 color)
{
	int ix, iy;
	int h, i, j, k;
	int oh, oi, oj, ok;
	int xmh, xph, ypk, ymk;
	int xmi, xpi, ymj, ypj;
	int xmj, xpj, ymi, ypi;
	int xmk, xpk, ymh, yph;

	if (rx==0 && ry==0) {  /* Special case - draw a single pixel */
		set_at( dst, x, y, color);
		return;
	}
	if (rx==0) { /* Special case for rx=0 - draw a vline */
		drawvertlineclip( dst, color, x, (Sint16)(y-ry), (Sint16)(y+ry) );
		return;
	}
	if (ry==0) { /* Special case for ry=0 - draw a hline */
		drawhorzlineclip( dst, color, (Sint16)(x-rx), y, (Sint16)(x+rx) );
		return;
	}


	/* Init vars */
	oh = oi = oj = ok = 0xFFFF;
	if (rx > ry) {
		ix = 0;
		iy = rx * 64;
		do {
			h = (ix + 16) >> 6;
			i = (iy + 16) >> 6;
			j = (h * ry) / rx;
			k = (i * ry) / rx;

			if (((ok!=k) && (oj!=k)) || ((oj!=j) && (ok!=j)) || (k!=j)) {
				xph=x+h-1;
				xmh=x-h;
				if (k>0) {
					ypk=y+k-1;
					ymk=y-k;
					if(h > 0) {
						set_at(dst, xmh, ypk, color);
						set_at(dst, xmh, ymk, color);
					}
					set_at(dst, xph, ypk, color);
					set_at(dst, xph, ymk, color);
				}
				ok=k;
				xpi=x+i-1;
				xmi=x-i;
				if (j>0) {
					ypj=y+j-1;
					ymj=y-j;
					set_at(dst, xmi, ypj, color);
					set_at(dst, xpi, ypj, color);
					set_at(dst, xmi, ymj, color);
					set_at(dst, xpi, ymj, color);
				}
				oj=j;
			}
			ix = ix + iy / rx;
			iy = iy - ix / rx;

		} while (i > h);
	} else {
		ix = 0;
		iy = ry * 64;
		do {
			h = (ix + 32) >> 6;
			i = (iy + 32) >> 6;
			j = (h * rx) / ry;
			k = (i * rx) / ry;

			if (((oi!=i) && (oh!=i)) || ((oh!=h) && (oi!=h) && (i!=h))) {
				xmj=x-j;
				xpj=x+j-1;
				if (i>0) {
					ypi=y+i-1;
					ymi=y-i;
					if(j > 0) {
						set_at(dst, xmj, ypi, color);
						set_at(dst, xmj, ymi, color);
					}
					set_at(dst, xpj, ypi, color);
					set_at(dst, xpj, ymi, color);
				}
				oi=i;
				xmk=x-k;
				xpk=x+k-1;
				if (h>0) {
					yph=y+h-1;
					ymh=y-h;
					set_at(dst, xmk, yph, color);
					set_at(dst, xpk, yph, color);
					set_at(dst, xmk, ymh, color);
					set_at(dst, xpk, ymh, color);
				}
				oh=h;
			}
			ix = ix + iy / ry;
			iy = iy - ix / ry;
		} while(i > h);
	}
}







static void draw_fillellipse(SDL_Surface *dst, int x, int y, int rx, int ry, Uint32 color)
{
	int ix, iy;
	int h, i, j, k;
	int oh, oi, oj, ok;

	if (rx==0 && ry==0) {  /* Special case - draw a single pixel */
		set_at( dst, x, y, color);
		return;
	}
	if (rx==0) { /* Special case for rx=0 - draw a vline */
		drawvertlineclip( dst, color, x, (Sint16)(y-ry), (Sint16)(y+ry) );
		return;
	}
	if (ry==0) { /* Special case for ry=0 - draw a hline */
		drawhorzlineclip( dst, color, (Sint16)(x-rx), y, (Sint16)(x+rx) );
		return;
	}

	/* Init vars */
	oh = oi = oj = ok = 0xFFFF;

	/* Draw */
	if (rx >= ry) {
		ix = 0;
		iy = rx * 64;

		do {
			h = (ix + 8) >> 6;
			i = (iy + 8) >> 6;
			j = (h * ry) / rx;
			k = (i * ry) / rx;
			if ((ok!=k) && (oj!=k) && (k<ry)) {
				drawhorzlineclip(dst, color, x-h, y-k-1, x+h-1);
				drawhorzlineclip(dst, color, x-h, y+k, x+h-1);
				ok=k;
			}
			if ((oj!=j) && (ok!=j) && (k!=j))  {
				drawhorzlineclip(dst, color, x-i, y+j, x+i-1);
				drawhorzlineclip(dst, color, x-i, y-j-1, x+i-1);
				oj=j;
			}
			ix = ix + iy / rx;
			iy = iy - ix / rx;

		} while (i > h);
	} else {
		ix = 0;
		iy = ry * 64;

		do {
			h = (ix + 8) >> 6;
			i = (iy + 8) >> 6;
			j = (h * rx) / ry;
			k = (i * rx) / ry;

			if ((oi!=i) && (oh!=i) && (i<ry)) {
				drawhorzlineclip(dst, color, x-j, y+i, x+j-1);
				drawhorzlineclip(dst, color, x-j, y-i-1, x+j-1);
				oi=i;
			}
			if ((oh!=h) && (oi!=h) && (i!=h)) {
				drawhorzlineclip(dst, color, x-k, y+h, x+k-1);
				drawhorzlineclip(dst, color, x-k, y-h-1, x+k-1);
				oh=h;
			}

			ix = ix + iy / ry;
			iy = iy - ix / ry;

		} while(i > h);
	}
}


static int compare_int(const void *a, const void *b)
{
	return (*(const int *)a) - (*(const int *)b);
}

static void draw_fillpoly(SDL_Surface *dst, int *vx, int *vy, int n, Uint32 color)
{
	int i;
	int y;
	int miny, maxy;
	int x1, y1;
	int x2, y2;
	int ind1, ind2;
	int ints;
	int *polyints = PyMem_New(int, n);


	/* Determine Y maxima */
	miny = vy[0];
	maxy = vy[0];
	for (i=1; (i < n); i++)
	{
		miny = min(miny, vy[i]);
		maxy = max(maxy, vy[i]);
	}

	/* Draw, scanning y */
	for(y=miny; (y <= maxy); y++) {
		ints = 0;
		for (i=0; (i < n); i++) {
			if (!i) {
				ind1 = n-1;
				ind2 = 0;
			} else {
				ind1 = i-1;
				ind2 = i;
			}
			y1 = vy[ind1];
			y2 = vy[ind2];
			if (y1 < y2) {
				x1 = vx[ind1];
				x2 = vx[ind2];
			} else if (y1 > y2) {
				y2 = vy[ind1];
				y1 = vy[ind2];
				x2 = vx[ind1];
				x1 = vx[ind2];
			} else {
				continue;
			}
			if ((y >= y1) && (y < y2)) {
				polyints[ints++] = (y-y1) * (x2-x1) / (y2-y1) + x1;
			} else if ((y == maxy) && (y > y1) && (y <= y2)) {
				polyints[ints++] = (y-y1) * (x2-x1) / (y2-y1) + x1;
			}
		}
		qsort(polyints, ints, sizeof(int), compare_int);

		for (i=0; (i<ints); i+=2) {
			drawhorzlineclip(dst, color, polyints[i], y, polyints[i+1]);
		}
	}
}







static PyMethodDef draw_builtins[] =
{
	{ "aaline", aaline, 1, doc_aaline },
	{ "line", line, 1, doc_line },
	{ "aalines", aalines, 1, doc_aalines },
	{ "lines", lines, 1, doc_lines },
	{ "ellipse", ellipse, 1, doc_ellipse },
	{ "arc", arc, 1, doc_arc },
	{ "circle", circle, 1, doc_circle },
	{ "polygon", polygon, 1, doc_polygon },
	{ "rect", rect, 1, doc_rect },

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

PYGAME_EXPORT
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


