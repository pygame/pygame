#ifndef _PYGAME_MATH_AABBBOX_
#define _PYGAME_MATH_AABBBOX_

#include "pgVector2.h"

//typedef struct _pgAABBBox pgAABBBox;

//TODO: complete the AABBBox
typedef struct _pgAABBBox{
	union
	{
		struct
		{
			double left, bottom, right, top;
		};

		struct
		{
			double from[2], to[2];
		};
	};
} pgAABBBox;

typedef enum _pgBoxDirect
{
	BD_NONE = -1,
	BD_LEFT,
	BD_BOTTOM,
	BD_RIGHT,
	BD_TOP
}pgBoxDirect;

pgAABBBox PG_GenAABB(double left, double right, double bottom, double top);
void PG_AABBExpandTo(pgAABBBox* box, pgVector2* p);
void PG_AABBClear(pgAABBBox* box);
int PG_IsOverlap(pgAABBBox* boxA, pgAABBBox* boxB);

#endif //_PYGAME_MATH_AABBBOX_

