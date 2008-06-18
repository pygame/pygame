#ifndef _PYGAME_PHYSICS_RENDERER_
#define _PYGAME_PHYSICS_RENDERER_


#include "pgBodyObject.h"
#include "pgWorldObject.h"
#include "pgJointObject.h"

void PGT_RenderWorld(pgWorldObject* world);
void PGT_RenderBody(pgBodyObject* body);
void PGT_RenderJoint(pgJointObject* joint);

extern int RENDER_AABB;

#endif //_PYGAME_PHYSICS_RENDERER_