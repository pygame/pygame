#ifndef _PYGAME_PHYSICS_RENDERER_
#define _PYGAME_PHYSICS_RENDERER_


#include "pgBodyObject.h"
#include "pgWorldObject.h"

void PGT_RenderWorld(pgWorldObject* world);
void PGT_RenderBody(pgBodyObject* body);

#endif //_PYGAME_PHYSICS_RENDERER_