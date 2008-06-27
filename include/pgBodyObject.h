#ifndef _PYGAME_PHYSICS_BODY_
#define _PYGAME_PHYSICS_BODY_


#include "pgVector2.h"
#include "pgDeclare.h"

//! typedef struct _pgWorldObject pgWorldObject;
//! typedef struct _pgShapeObject pgShapeObject;

struct _pgBodyObject{
	PyObject_HEAD

	double		fMass;
	pgVector2	vecLinearVelocity;
	double		fAngleVelocity;
	int			bStatic;

	pgVector2	vecPosition;
	double		fRotation;
	pgVector2	vecImpulse;
	pgVector2	vecForce;
	double		fTorque;

	double		fRestitution;
	double		fFriction;

	pgShapeObject* shape;

};

pgBodyObject* PG_BodyNew();
void	PG_BodyDestroy(pgBodyObject* body);

void PG_FreeUpdateBodyVel(pgWorldObject* world, pgBodyObject* body, double dt);
void PG_FreeUpdateBodyPos(pgWorldObject* world, pgBodyObject* body, double dt);

//transform point local_p's position from body's local coordinate to the world's global one.
//TODO: is the local coordinate necessary? anyway let it alone right now.
pgVector2 PG_GetGlobalPos(pgBodyObject* body, pgVector2* local_p);

//return the global velocity of a point p (on the rigid body)
//(notice: here p is defined in the global coordinate)
pgVector2 PG_AngleToLinear1(pgBodyObject* body, pgVector2* global_p);
pgVector2 PG_AngleToLinear(pgVector2* r, double w);

//translate vector from coordinate B to coordinate A
pgVector2 PG_GetRelativePos(pgBodyObject* bodyA, pgBodyObject* bodyB, pgVector2* p_in_B);


//bind rect shape with body
void PG_Bind_RectShape(pgBodyObject* body, int width, int height, double seta);

#endif //_PYGAME_PHYSICS_BODY_

