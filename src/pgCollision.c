#include "pgCollision.h"
#include "pgAABBBox.h"
#include "pgShapeObject.h"
#include "pgBodyObject.h"
#include <assert.h>

extern PyTypeObject pgContactType;

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

int PG_PartlyLB(pgAABBBox* box, pgVector2* p1, pgVector2* p2, 
				pgCollisionAxis axis, pgVector2* ans_p1, pgVector2* ans_p2, 
				int* valid_p1, int* valid_p2)
{
	pgVector2 dp;
	double u1, u2;
	
	u1 = 0.f;
	u2 = 1.f;
	dp = c_diff(*p2, *p1);

	switch(axis)
	{
	case CA_X:
		if(!_LiangBarskey_Internal(-dp.imag, p1->imag - box->bottom, &u1, &u2)) return 0;
		if(!_LiangBarskey_Internal(dp.imag, box->top - p1->imag, &u1, &u2)) return 0;
		break;
	case CA_Y:
		if(!_LiangBarskey_Internal(-dp.real, p1->real - box->left, &u1, &u2)) return 0;
		if(!_LiangBarskey_Internal(dp.real, box->right - p1->real, &u1, &u2)) return 0;
		break;
	default:
		assert(0);
		break;
	}

	if(u1 > u2) return 0;

	if(u1 == 0.f)
		*ans_p1 = *p1;
	else
		*ans_p1 = c_sum(*p1, c_mul_complex_with_real(dp, u1)); //ans_p1 = p1 + u1*dp
	if(u2 == 1.f)
		*ans_p2 = *p2;
	else
		*ans_p2 = c_sum(*p1, c_mul_complex_with_real(dp, u2)); //ans_p2 = p2 + u2*dp;

	switch(axis)
	{
	case CA_X:
		*valid_p1 = less_equal(box->left, ans_p1->real) && 
					less_equal(ans_p1->real, box->right);
		*valid_p2 = less_equal(box->left, ans_p2->real) && 
					less_equal(ans_p2->real, box->right);
		break;
	case CA_Y:
		*valid_p1 = less_equal(box->bottom, ans_p1->imag) && 
					less_equal(ans_p1->imag, box->top);
		*valid_p2 = less_equal(box->bottom, ans_p2->imag) && 
					less_equal(ans_p2->imag, box->top);
		break;
	default:
		assert(0);
		break;
	}

	return *valid_p1 || *valid_p2;
}


void PG_AppendContact(pgBodyObject* refBody, pgBodyObject* incidBody, PyObject* contactList)
{
	refBody->shape->Collision(refBody, incidBody, contactList);
}

void PG_ApplyContact(PyObject* contactObject, double step)
{
#define MAX_C_DEP 0.01
#define BIAS_FACTOR 0.25

	pgVector2 neg_dV, refV, incidV;
	pgVector2 refR, incidR;
	pgContact *contact;
	pgBodyObject *refBody, *incidBody;
	double moment_len;
	pgVector2 moment;
	pgVector2* p;

	double vbias;
	pgVector2 brefV, bincidV, bneg_dV;
	double bm_len;
	pgVector2 bm;

	contact = (pgContact*)contactObject;
	refBody = contact->joint.body1;
	incidBody = contact->joint.body2;

	contact->resist = sqrtf(refBody->fRestitution*incidBody->fRestitution);

	refR = c_diff(contact->pos, refBody->vecPosition);
	incidR = c_diff(contact->pos, incidBody->vecPosition);
	//dV = v2 + w2xr2 - (v1 + w1xr1)
	incidV = c_sum(incidBody->vecLinearVelocity, c_fcross(incidBody->fAngleVelocity,
		           incidR));
	refV = c_sum(refBody->vecLinearVelocity, c_fcross(refBody->fAngleVelocity,
		         refR));

	contact->dv = c_diff(incidV, refV);
	neg_dV = c_diff(refV, incidV);
	
	moment_len = c_dot(c_mul_complex_with_real(neg_dV, (1 + contact->resist)), 
		contact->normal)/contact->kFactor;
	moment_len = MAX(0, moment_len);
	
	//finally we get the momentum(oh...)
	moment = c_mul_complex_with_real(contact->normal, moment_len);
	p = *(contact->ppAccMoment);
	//TODO: test weight
	p->real += moment.real/contact->weight;
	p->imag += moment.imag/contact->weight; 

	//split impulse
	vbias = BIAS_FACTOR*MAX(0, contact->depth - MAX_C_DEP)/step;
	//biasdv
	bincidV = c_sum(incidBody->cBiasLV, c_fcross(incidBody->cBiasW, incidR));
	brefV = c_sum(refBody->cBiasLV, c_fcross(refBody->cBiasW, refR));
	bneg_dV = c_diff(brefV, bincidV); 
	//bias_moment
	bm_len = c_dot(c_mul_complex_with_real(bneg_dV, (1)),
		contact->normal)/contact->kFactor;
	bm_len = MAX(0, bm_len + vbias/contact->kFactor);
	bm = c_mul_complex_with_real(contact->normal, bm_len);
	p = *(contact->ppSplitAccMoment);
	p->real += bm.real/contact->weight;
	p->imag += bm.imag/contact->weight;

}

void PG_UpdateV(pgJointObject* joint, double step)
{
	pgContact *contact;
	pgBodyObject *refBody, *incidBody;
	pgVector2 moment, bm;
	pgVector2 refR, incidR;

	contact = (pgContact*)joint;
	refBody = joint->body1;
	incidBody = joint->body2;
	moment = **(contact->ppAccMoment);
	bm = **(contact->ppSplitAccMoment);

	refR = c_diff(contact->pos, refBody->vecPosition);
	incidR = c_diff(contact->pos, incidBody->vecPosition);

	if(c_dot(contact->dv, contact->normal) > 0) return;

	if(!refBody->bStatic)
	{
		refBody->vecLinearVelocity = c_diff(refBody->vecLinearVelocity, 
			c_div_complex_with_real(moment, refBody->fMass));
		refBody->fAngleVelocity -= c_cross(refR, moment)/refBody->shape->rInertia;

		refBody->cBiasLV = c_diff(refBody->cBiasLV,
			c_div_complex_with_real(bm, refBody->fMass));
		refBody->cBiasW -= c_cross(refR, bm)/refBody->shape->rInertia;
	}

	if(!incidBody->bStatic)
	{
		incidBody->vecLinearVelocity = c_sum(incidBody->vecLinearVelocity, 
			c_div_complex_with_real(moment, incidBody->fMass));
		incidBody->fAngleVelocity += c_cross(incidR, moment)/incidBody->shape->rInertia;

		incidBody->cBiasLV = c_sum(incidBody->cBiasLV,
			c_div_complex_with_real(bm, incidBody->fMass));
		incidBody->cBiasW += c_cross(incidR, bm)/incidBody->shape->rInertia;
	}
}

void PG_UpdateP(pgJointObject* joint, double step)
{
	//isolated function
}

void PG_ContactDestroy(pgJointObject* contact)
{
	pgVector2 **p = ((pgContact*)contact)->ppAccMoment;
	if(p)
	{
		if(*p)
		{
			PyObject_Free(*p);
			*p = NULL;
		}
		PyObject_Free(p);
		p = NULL;
	}

	p = ((pgContact*)contact)->ppSplitAccMoment;
	if(p)
	{
		if(*p)
		{
			PyObject_Free(*p);
			*p = NULL;
		}
		PyObject_Free(p);
		p = NULL;
	}
}

PyObject* _PG_ContactNew(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	pgContact* op;
	type->tp_base = &pgJointType;
	if(PyType_Ready(type) < 0) return NULL;
	op = (pgContact*)type->tp_alloc(type, 0);
	return (PyObject*)op;
}

pgJointObject* PG_ContactNew(pgBodyObject* refBody, pgBodyObject* incidBody)
{
	pgContact* contact;
	//TODO: this function would be replaced.
	contact = (pgContact*)_PG_ContactNew(&pgContactType, NULL, NULL);
	contact->joint.body1 = refBody;
	contact->joint.body2 = incidBody;
	contact->joint.SolveConstraintPosition = PG_UpdateP;
	contact->joint.SolveConstraintVelocity = PG_UpdateV;
	contact->joint.Destroy = PG_ContactDestroy;

	contact->ppAccMoment = NULL;
	contact->ppSplitAccMoment = NULL;

	return (pgJointObject*)contact;
}

PyTypeObject pgContactType =
{
	PyObject_HEAD_INIT(NULL)
	0,
	"physics.Contact",			/* tp_name */
	sizeof(pgContact),			/* tp_basicsize */
	0,                          /* tp_itemsize */
	0,							/* tp_dealloc */
	0,                          /* tp_print */
	0,                          /* tp_getattr */
	0,                          /* tp_setattr */
	0,                          /* tp_compare */
	0,                          /* tp_repr */
	0,                          /* tp_as_number */
	0,                          /* tp_as_sequence */
	0,                          /* tp_as_mapping */
	0,                          /* tp_hash */
	0,                          /* tp_call */
	0,                          /* tp_str */
	0,                          /* tp_getattro */
	0,                          /* tp_setattro */
	0,                          /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
	"",                         /* tp_doc */
	0,                          /* tp_traverse */
	0,                          /* tp_clear */
	0,                          /* tp_richcompare */
	0,                          /* tp_weaklistoffset */
	0,                          /* tp_iter */
	0,                          /* tp_iternext */
	0,							/* tp_methods */
	0,							/* tp_members */
	0,							/* tp_getset */
	0,							/* tp_base */
	0,                          /* tp_dict */
	0,                          /* tp_descr_get */
	0,                          /* tp_descr_set */
	0,                          /* tp_dictoffset */
	0,							/* tp_init */
	0,							/* tp_alloc */
	_PG_ContactNew,				/* tp_new */
	0,                          /* tp_free */
	0,                          /* tp_is_gc */
	0,                          /* tp_bases */
	0,                          /* tp_mro */
	0,                          /* tp_cache */
	0,                          /* tp_subclasses */
	0,                          /* tp_weaklist */
	0                           /* tp_del */
};
