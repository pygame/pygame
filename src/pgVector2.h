#ifndef _PYGAME_MATH_VECTOR2_
#define _PYGAME_MATH_VECTOR2_


typedef struct _pgVector2
{
	pgReal x,y;
} pgVector2;

static inline pgVector2 pgInitVector2(pgReal x,pgReal y);
static inline pgVector2 pgAddVector2(const pgVector2 vec1,const pgVector2 vec2);
static inline pgVector2 pgSubVector2(const pgVector2 vec1,const pgVector2 vec2);
static inline pgVector2 pgMulVector2(const pgVector2 vec1,const pgVector2 vec2);
static inline pgVector2 pgMulVector2WithReal(const pgVector2 vec,const pgReal real);
static inline pgReal	pgDotVector2(const pgVector2 vec1,const pgVector2 vec2);
static inline pgReal	pgGetLengthVector2(const pgVector2 vec);
static inline pgReal	pgGetLengthSquareVector2(const pgVector2 vec);
static inline void		pgNormalizeVector2(pgVector*	pVec);
static inline pgVector2 pgCrossVector2(const pgVector2 vec1,const pgVector2 vec2);


static inline pgVector2 pgMoveVector2(const pgVector2 vec,const pgVector2 moveVec);
static inline pgVector2 pgScaleVector2(const pgVector2 vec,const pgVector2 scaleVec);
static inline pgVector2 pgRotate(const pgVector2 vec,const pgVector centerVec,const pgReal angle);

#endif //_PYGAME_MATH_VECTOR2_