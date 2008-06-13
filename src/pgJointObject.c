#include "pgJointObject.h"

void PG_InitJointBase(pgJointObject* joint,pgBodyObject* b1,pgBodyObject* b2,int bCollideConnect)
{
	joint->body1 = b1;
	joint->body2 = b2;
	joint->isCollideConnect = bCollideConnect;
	joint->SolveConstraint = NULL;
}

void PG_SolveDistanceJoint(pgJointObject* joint,double stepTime)
{
	pgVector2 vecL;
	double lamda,cosTheta1V,cosTheta2V,mk;
	Py_complex impuseAdd,v1Add,v2Add;
	pgDistanceJoint* pJoint = (pgDistanceJoint*)joint;
	if (joint->body1 && (!joint->body2))
	{
		vecL = c_diff(joint->body1->vecPosition,pJoint->anchor2);
		c_normalize(&vecL);
		lamda = -c_dot(vecL,joint->body1->vecLinearVelocity);
		vecL = c_mul_complex_with_real(vecL,lamda);
		joint->body1->vecLinearVelocity = c_sum(joint->body1->vecLinearVelocity,vecL);
		return;
	}

	if(joint->body1 && joint->body2)
	{
		vecL = c_diff(joint->body1->vecPosition,joint->body2->vecPosition);
		c_normalize(&vecL);
		cosTheta1V = c_dot(vecL,joint->body1->vecLinearVelocity);
		cosTheta2V = c_dot(vecL,joint->body2->vecLinearVelocity);
		lamda = cosTheta1V - cosTheta2V;
		mk = joint->body1->fMass * joint->body2->fMass / (joint->body1->fMass + joint->body2->fMass);
		lamda *= mk;
		impuseAdd = c_mul_complex_with_real(vecL,lamda);
		v1Add = c_div_complex_with_real(impuseAdd,joint->body1->fMass);
		v2Add = c_div_complex_with_real(impuseAdd,joint->body2->fMass);
		/*joint->body1->vecLinearVelocity = c_sum(joint->body1->vecLinearVelocity,v1Add);
		joint->body2->vecLinearVelocity = c_diff(joint->body2->vecLinearVelocity,v2Add);*/
		joint->body1->vecLinearVelocity = c_diff(joint->body1->vecLinearVelocity,v1Add);
		joint->body2->vecLinearVelocity = c_sum(joint->body2->vecLinearVelocity,v2Add);
		return;
	}
}

pgJointObject* PG_DistanceJointNew(pgBodyObject* b1,pgBodyObject* b2,int bCollideConnect,double dist,pgVector2 a1,pgVector2 a2)
{
	pgDistanceJoint* pjoint = (pgDistanceJoint*)PyObject_MALLOC(sizeof(pgDistanceJoint));
	PG_InitJointBase(&(pjoint->joint), b1, b2, bCollideConnect);
	pjoint->distance = dist;
	pjoint->anchor1 = a1;
	pjoint->anchor2 = a2;
	pjoint->joint.SolveConstraint = PG_SolveDistanceJoint;
	return (pgJointObject*)pjoint;
}
