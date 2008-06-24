#include "pgCollision.h"
#include "pgAABBBox.h"
#include "pgShapeObject.h"
#include "pgBodyObject.h"
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
//TEST: Liang-Barskey clip has been tested.
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

void PG_AppendContact(pgBodyObject* refBody, pgBodyObject* incidBody, PyObject* contactList)
{
	refBody->shape->Collision(refBody, incidBody, contactList);
}

void PG_UpdateV(pgJointObject* joint, double step)
{
	pgVector2 neg_dV, refV, incidV;
	pgVector2 refR, incidR;
	pgContact *contact;
	pgBodyObject *refBody, *incidBody;
	double k, tmp1, tmp2;
	double moment_len;
	pgVector2 moment;

	contact = (pgContact*)joint;
	refBody = joint->body1;
	incidBody = joint->body2;

	//calculate the normal impulse
	//k
	refR = c_diff(contact->pos, refBody->vecPosition);
	incidR = c_diff(contact->pos, incidBody->vecPosition);
	
	tmp1 = refR.real*contact->normal.imag - refR.imag*contact->normal.real;
	tmp2 = incidR.real*contact->normal.imag - incidR.imag*contact->normal.real;

	k = 1/refBody->fMass + 1/incidBody->fMass + tmp1*tmp1/refBody->shape->rInertia
		+ tmp2*tmp2/incidBody->shape->rInertia;
	
	//dV = v2 + w2xr2 - (v1 + w1xr1)
	incidV = c_sum(incidBody->vecLinearVelocity, PG_AngleToLinear1(incidBody, &(contact->pos)));
	refV = c_sum(refBody->vecLinearVelocity, PG_AngleToLinear1(refBody, &(contact->pos)));
	neg_dV = c_diff(refV, incidV);
	
	moment_len = c_dot(neg_dV, contact->normal)/k;
	moment_len *= refBody->fRestitution;
	if(moment_len < 0)
		moment_len = 0;
	//finally we get the momentum(oh...)
	moment = c_mul_complex_with_real(contact->normal, moment_len);

	//update the v and w
	refBody->vecLinearVelocity = c_diff(refBody->vecLinearVelocity, 
		c_div_complex_with_real(moment, refBody->fMass));
	refBody->fAngleVelocity -= c_cross(refR, moment)/refBody->shape->rInertia;

	incidBody->vecLinearVelocity = c_sum(incidBody->vecLinearVelocity, 
		c_div_complex_with_real(moment, incidBody->fMass));
	incidBody->fAngleVelocity -= c_cross(refR, moment)/incidBody->shape->rInertia;
}

void PG_UpdateP(pgJointObject* joint, double step)
{
	//TODO: concern dt
	joint->body1->vecPosition = c_sum(joint->body1->vecPosition, joint->body1->vecLinearVelocity);
	joint->body1->fRotation += joint->body1->fAngleVelocity;

	joint->body2->vecPosition = c_sum(joint->body2->vecPosition, joint->body2->vecLinearVelocity);
	joint->body2->fRotation += joint->body2->fAngleVelocity;
}



pgJointObject* PG_ContactNew(pgBodyObject* refBody, pgBodyObject* incidBody)
{
	pgContact* contact;
	//TODO: this function would be changed
	contact = (pgContact*)PyObject_MALLOC(sizeof(pgContact));
	contact->joint.body1 = refBody;
	contact->joint.body2 = incidBody;
	contact->joint.SolveConstraintPosition = PG_UpdateP;
	contact->joint.SolveConstraintVelocity = PG_UpdateV;

	return (pgJointObject*)contact;
}