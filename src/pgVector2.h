#ifndef _PYGAME_MATH_VECTOR2_
#define _PYGAME_MATH_VECTOR2_

#include "pgMathTypeDef.h"

typedef struct _pgVector2
{
	pgReal x,y;
} pgVector2;

static pgVector2 pgInitVector2(pgReal x,pgReal y);
static pgVector2 pgAddVector2(const pgVector2 vec1,const pgVector2 vec2);
static pgVector2 pgSubVector2(const pgVector2 vec1,const pgVector2 vec2);
static pgVector2 pgMulVector2(const pgVector2 vec1,const pgVector2 vec2);
static pgVector2 pgMulVector2WithReal(const pgVector2 vec,const pgReal real);
static pgReal	pgDotVector2(const pgVector2 vec1,const pgVector2 vec2);
static pgReal	pgGetLengthVector2(const pgVector2 vec);
static pgReal	pgGetLengthSquareVector2(const pgVector2 vec);
static void		pgNormalizeVector2(pgVector2*	pVec);
static pgVector2 pgCrossVector2(const pgVector2 vec1,const pgVector2 vec2);


static pgVector2 pgMoveVector2(const pgVector2 vec,const pgVector2 moveVec);
static pgVector2 pgScaleVector2(const pgVector2 vec,const pgVector2 scaleVec);
static pgVector2 pgRotate(const pgVector2 vec,const pgVector2 centerVec,const pgReal angle);

#endif //_PYGAME_MATH_VECTOR2_