#ifndef _PYGAME_MATH_AABBBOX_
#define _PYGAME_MATH_AABBBOX_

//TODO: complete the AABBBox and ODBBox
typedef struct _pgAABBBox{
	double left, right, bottom, top;
} pgAABBBox;

pgAABBBox PG_GenAABB(double left, double right, double bottom, double top);

#endif //_PYGAME_MATH_AABBBOX_