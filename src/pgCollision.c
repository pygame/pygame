#include "pgCollision.h"
#include "pgAABBBox.h"

// We borrow this graph from Box2DLite
// Box vertex and edge numbering:
//
//        ^ y
//        |
//        e1
//   v2 ----- v1
//    |        |
// e2 |        | e4  --> x
//    |        |
//   v3 ----- v4
//        e3


//TODO: add rest contact


// Apply Liang-Barskey clip on a AABB box
// (p1, p2) is the input line segment to be clipped (note: it's a 2d vector)
// (ans_p1, ans_p2) is the output line segment
// TODO: tone Liang-Barskey clip
int _LiangBarskey_Internal(double p, double q, double* u1, double* u2)
{
	if(is_zero(p) && q < 0) return 0;
	
	if(p < 0)
		*u1 = MAX(*u1, q/p);
	else
		*u2 = MIN(*u2, q/p);

	return 1; 
}

int _PG_LiangBarskey(pgAABBBox* box, pgVector2* p1, pgVector2* p2, pgVector2* ans_p1, pgVector2* ans_p2)
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

	*ans_p1 = c_sum(*p1, c_mul_complex_with_real(dp, u1)); //ans_p1 = p1 + u1*dp
	*ans_p2 = c_sum(*p2, c_mul_complex_with_real(dp, u2)); //ans_p2 = p2 + u2*dp;

	return 1;
}

pgContact* pg_BuildContact(pgBodyObject* bodyA, pgBodyObject* bodyB)
{
	//clipping
	//find contact points
	//find the (proper) normal

	return NULL;
}



