#include "pgCollision.h"
#include "pgAABBBox.h"
#include <assert.h>

// We borrow this graph from Box2DLite
// Box vertex and edge numbering:
//
//        ^ y
//        |
//        e3
//   v3 ----- v2
//    |        |
// e0 |        | e2  --> x
//    |        |
//   v0 ----- v1
//        e1


//TODO: add rest contact


// Apply Liang-Barskey clip on a AABB box
// (p1, p2) is the input line segment to be clipped (note: it's a 2d vector)
// (ans_p1, ans_p2) is the output line segment
// TODO: tone Liang-Barskey clip


typedef struct _LB_Rec
{
	double val;
	pgBoxDirect d;
}LB_Rec;

#define LB_Rec_MAX(x, y) ( ( ( (x).val ) > ( (y).val ) ) ? (x) : (y) )
#define LB_Rec_MIN(x, y) ( ( ( (x).val ) < ( (y).val ) ) ? (x) : (y) )

int _LiangBarskey_Internal(double p, double q, double* u1, double* u2)
{
	double val;

	if(is_zero(p))
	{
		if(q < 0)
			return 0;
		
		return 1;
	}
	
	val = q/p;
	
	if(p < 0)
		*u1 = MAX(*u1, val);
	else
		*u2 = MIN(*u2, val);

	return 1; 
}

int PG_LiangBarskey(pgAABBBox* box, pgVector2* p1, pgVector2* p2, 
					 pgVector2* ans_p1, pgVector2* ans_p2)
{
	pgVector2 dp;
	double u1, u2;
	

	u1 = 0.f;
	u2 = 1.f;
	dp = c_diff(*p2, *p1);	//dp = p2 - p1

	if(!_LiangBarskey_Internal(-dp.real, p1->real - box->left, &u1, &u2)) return 0;
	if(!_LiangBarskey_Internal(dp.real, box->right - p1->real, &u1, &u2)) return 0;
	if(!_LiangBarskey_Internal(-dp.imag, p1->imag - box->bottom, &u1, &u2)) return 0;
	if(!_LiangBarskey_Internal(dp.imag, box->top - p1->imag, &u1, &u2)) return 0;

	if(u1 > u2) return 0;

	if(u1 == 0.f)
		*ans_p1 = *p1;
	else
		*ans_p1 = c_sum(*p1, c_mul_complex_with_real(dp, u1)); //ans_p1 = p1 + u1*dp
	if(u2 == 1.f)
		*ans_p2 = *p2;
	else
		*ans_p2 = c_sum(*p1, c_mul_complex_with_real(dp, u2)); //ans_p2 = p2 + u2*dp;

	return 1;
}

pgContact* pg_BuildContact(pgBodyObject* bodyA, pgBodyObject* bodyB)
{
	//clipping
	//find contact points
	//find the (proper) normal

	return NULL;
}



