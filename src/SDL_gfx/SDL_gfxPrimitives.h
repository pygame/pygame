/* 

 SDL_gfxPrimitives: graphics primitives for SDL

 LGPL (c) A. Schiffler
 
*/

#ifndef _SDL_gfxPrimitives_h
#define _SDL_gfxPrimitives_h

#include <math.h>
#ifndef M_PI
#define M_PI	3.141592654
#endif

#include "SDL.h"

/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/* ----- Versioning */

#define SDL_GFXPRIMITIVES_MAJOR	2
#define SDL_GFXPRIMITIVES_MINOR	0
#define SDL_GFXPRIMITIVES_MICRO	19

/* ----- Statically Linking so don't want a DLL interface */

#define DLLINTERFACE

/* ----- Prototypes */

/* Note: all ___Color routines expect the color to be in format 0xRRGGBBAA */

/* Pixel */

    DLLINTERFACE int pixelColor(SDL_Surface * dst, Sint16 x, Sint16 y, Uint32 color);
    DLLINTERFACE int pixelRGBA(SDL_Surface * dst, Sint16 x, Sint16 y, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

/* Horizontal line */

    DLLINTERFACE int hlineColor(SDL_Surface * dst, Sint16 x1, Sint16 x2, Sint16 y, Uint32 color);
    DLLINTERFACE int hlineRGBA(SDL_Surface * dst, Sint16 x1, Sint16 x2, Sint16 y, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

/* Vertical line */

    DLLINTERFACE int vlineColor(SDL_Surface * dst, Sint16 x, Sint16 y1, Sint16 y2, Uint32 color);
    DLLINTERFACE int vlineRGBA(SDL_Surface * dst, Sint16 x, Sint16 y1, Sint16 y2, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

/* Rectangle */

    DLLINTERFACE int rectangleColor(SDL_Surface * dst, Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Uint32 color);
    DLLINTERFACE int rectangleRGBA(SDL_Surface * dst, Sint16 x1, Sint16 y1,
				   Sint16 x2, Sint16 y2, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

/* Filled rectangle (Box) */

    DLLINTERFACE int boxColor(SDL_Surface * dst, Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Uint32 color);
    DLLINTERFACE int boxRGBA(SDL_Surface * dst, Sint16 x1, Sint16 y1, Sint16 x2,
			     Sint16 y2, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

/* Line */

    DLLINTERFACE int lineColor(SDL_Surface * dst, Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Uint32 color);
    DLLINTERFACE int lineRGBA(SDL_Surface * dst, Sint16 x1, Sint16 y1,
			      Sint16 x2, Sint16 y2, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

/* AA Line */
    DLLINTERFACE int aalineColor(SDL_Surface * dst, Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Uint32 color);
    DLLINTERFACE int aalineRGBA(SDL_Surface * dst, Sint16 x1, Sint16 y1,
				Sint16 x2, Sint16 y2, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

/* Circle */

    DLLINTERFACE int circleColor(SDL_Surface * dst, Sint16 x, Sint16 y, Sint16 r, Uint32 color);
    DLLINTERFACE int circleRGBA(SDL_Surface * dst, Sint16 x, Sint16 y, Sint16 rad, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

/* Circle */

    DLLINTERFACE int arcColor(SDL_Surface * dst, Sint16 x, Sint16 y, Sint16 r, Sint16 start, Sint16 end, Uint32 color);
    DLLINTERFACE int arcRGBA(SDL_Surface * dst, Sint16 x, Sint16 y, Sint16 rad, Sint16 start, Sint16 end, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

/* AA Circle */

    DLLINTERFACE int aacircleColor(SDL_Surface * dst, Sint16 x, Sint16 y, Sint16 r, Uint32 color);
    DLLINTERFACE int aacircleRGBA(SDL_Surface * dst, Sint16 x, Sint16 y,
				  Sint16 rad, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

/* Filled Circle */

    DLLINTERFACE int filledCircleColor(SDL_Surface * dst, Sint16 x, Sint16 y, Sint16 r, Uint32 color);
    DLLINTERFACE int filledCircleRGBA(SDL_Surface * dst, Sint16 x, Sint16 y,
				      Sint16 rad, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

/* Ellipse */

    DLLINTERFACE int ellipseColor(SDL_Surface * dst, Sint16 x, Sint16 y, Sint16 rx, Sint16 ry, Uint32 color);
    DLLINTERFACE int ellipseRGBA(SDL_Surface * dst, Sint16 x, Sint16 y,
				 Sint16 rx, Sint16 ry, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

/* AA Ellipse */

    DLLINTERFACE int aaellipseColor(SDL_Surface * dst, Sint16 xc, Sint16 yc, Sint16 rx, Sint16 ry, Uint32 color);
    DLLINTERFACE int aaellipseRGBA(SDL_Surface * dst, Sint16 x, Sint16 y,
				   Sint16 rx, Sint16 ry, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

/* Filled Ellipse */

    DLLINTERFACE int filledEllipseColor(SDL_Surface * dst, Sint16 x, Sint16 y, Sint16 rx, Sint16 ry, Uint32 color);
    DLLINTERFACE int filledEllipseRGBA(SDL_Surface * dst, Sint16 x, Sint16 y,
				       Sint16 rx, Sint16 ry, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

#define CLOCKWISE

/* Pie */

    DLLINTERFACE int pieColor(SDL_Surface * dst, Sint16 x, Sint16 y, Sint16 rad,
			      Sint16 start, Sint16 end, Uint32 color);
    DLLINTERFACE int pieRGBA(SDL_Surface * dst, Sint16 x, Sint16 y, Sint16 rad,
			     Sint16 start, Sint16 end, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

/* Filled Pie */

    DLLINTERFACE int filledPieColor(SDL_Surface * dst, Sint16 x, Sint16 y, Sint16 rad,
				    Sint16 start, Sint16 end, Uint32 color);
    DLLINTERFACE int filledPieRGBA(SDL_Surface * dst, Sint16 x, Sint16 y, Sint16 rad,
				   Sint16 start, Sint16 end, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

/* Trigon */

    DLLINTERFACE int trigonColor(SDL_Surface * dst, Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Sint16 x3, Sint16 y3, Uint32 color);
    DLLINTERFACE int trigonRGBA(SDL_Surface * dst, Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Sint16 x3, Sint16 y3,
				 Uint8 r, Uint8 g, Uint8 b, Uint8 a);

/* AA-Trigon */

    DLLINTERFACE int aatrigonColor(SDL_Surface * dst, Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Sint16 x3, Sint16 y3, Uint32 color);
    DLLINTERFACE int aatrigonRGBA(SDL_Surface * dst,  Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Sint16 x3, Sint16 y3,
				   Uint8 r, Uint8 g, Uint8 b, Uint8 a);

/* Filled Trigon */

    DLLINTERFACE int filledTrigonColor(SDL_Surface * dst, Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Sint16 x3, Sint16 y3, Uint32 color);
    DLLINTERFACE int filledTrigonRGBA(SDL_Surface * dst, Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Sint16 x3, Sint16 y3,
				       Uint8 r, Uint8 g, Uint8 b, Uint8 a);

/* Polygon */

    DLLINTERFACE int polygonColor(SDL_Surface * dst, const Sint16 * vx, const Sint16 * vy, int n, Uint32 color);
    DLLINTERFACE int polygonRGBA(SDL_Surface * dst, const Sint16 * vx, const Sint16 * vy,
				 int n, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

/* AA-Polygon */

    DLLINTERFACE int aapolygonColor(SDL_Surface * dst, const Sint16 * vx, const Sint16 * vy, int n, Uint32 color);
    DLLINTERFACE int aapolygonRGBA(SDL_Surface * dst, const Sint16 * vx, const Sint16 * vy,
				   int n, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

/* Filled Polygon */

    DLLINTERFACE int filledPolygonColor(SDL_Surface * dst, const Sint16 * vx, const Sint16 * vy, int n, Uint32 color);
    DLLINTERFACE int filledPolygonRGBA(SDL_Surface * dst, const Sint16 * vx,
				       const Sint16 * vy, int n, Uint8 r, Uint8 g, Uint8 b, Uint8 a);
    DLLINTERFACE int texturedPolygon(SDL_Surface * dst, const Sint16 * vx, const Sint16 * vy, int n, SDL_Surface * texture,int texture_dx,int texture_dy);

/* (Note: These MT versions are required for multi-threaded operation.) */

    DLLINTERFACE int filledPolygonColorMT(SDL_Surface * dst, const Sint16 * vx, const Sint16 * vy, int n, Uint32 color, int **polyInts, int *polyAllocated);
    DLLINTERFACE int filledPolygonRGBAMT(SDL_Surface * dst, const Sint16 * vx,
				       const Sint16 * vy, int n, Uint8 r, Uint8 g, Uint8 b, Uint8 a,
				       int **polyInts, int *polyAllocated);
    DLLINTERFACE int texturedPolygonMT(SDL_Surface * dst, const Sint16 * vx, const Sint16 * vy, int n, SDL_Surface * texture,int texture_dx,int texture_dy, int **polyInts, int *polyAllocated);

/* Bezier */
/* (s = number of steps) */

    DLLINTERFACE int bezierColor(SDL_Surface * dst, const Sint16 * vx, const Sint16 * vy, int n, int s, Uint32 color);
    DLLINTERFACE int bezierRGBA(SDL_Surface * dst, const Sint16 * vx, const Sint16 * vy,
				 int n, int s, Uint8 r, Uint8 g, Uint8 b, Uint8 a);


/* Characters/Strings */

    DLLINTERFACE int characterColor(SDL_Surface * dst, Sint16 x, Sint16 y, char c, Uint32 color);
    DLLINTERFACE int characterRGBA(SDL_Surface * dst, Sint16 x, Sint16 y, char c, Uint8 r, Uint8 g, Uint8 b, Uint8 a);
    DLLINTERFACE int stringColor(SDL_Surface * dst, Sint16 x, Sint16 y, const char *c, Uint32 color);
    DLLINTERFACE int stringRGBA(SDL_Surface * dst, Sint16 x, Sint16 y, const char *c, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

    DLLINTERFACE void gfxPrimitivesSetFont(const void *fontdata, int cw, int ch);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif

#endif				/* _SDL_gfxPrimitives_h */
