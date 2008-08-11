#ifndef _PYGAME_PHYSICS_RENDERER_
#define _PYGAME_PHYSICS_RENDERER_

//#include <physics/pgphysics.h>
#include "pgphysics.h"
void PGT_RenderWorld(PyWorldObject* world);
void PGT_RenderBody(PyBodyObject* body);
void PGT_RenderJoint(PyJointObject* joint);

extern int RENDER_AABB;

#endif //_PYGAME_PHYSICS_RENDERER_


