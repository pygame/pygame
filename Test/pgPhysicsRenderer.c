#include "pgPhysicsRenderer.h"
#include "pgShapeObject.h"
#include <gl/glut.h>

void PGT_RenderWorld(pgWorldObject* world)
{
	Py_ssize_t size = PyList_Size((PyObject*)(world->bodyList));
	Py_ssize_t i;
	for (i = 0;i < size;i++)
	{
		pgBodyObject* body = (pgBodyObject*)(PyList_GetItem((PyObject*)(world->bodyList),i));
		PGT_RenderBody(body);
	}

	size = PyList_Size((PyObject*)(world->jointList));
	for (i = 0;i < size;i++)
	{
		pgJointObject* joint = (pgJointObject*)(PyList_GetItem((PyObject*)(world->jointList),i));
		PGT_RenderJoint(joint);
	}
}

void PGT_RenderBody(pgBodyObject* body)
{
	pgVector2 gp[4];
	int i;
	pgRectShape* rect = (pgRectShape*)body->shape;

	for(i = 0; i < 4; ++i)
		gp[i] = PG_GetGlobalCor(body, &(rect->point[i]));

	glColor3f(1.f, 1.f, 0.f);
	glLineWidth(2.f);
	glBegin(GL_LINE_LOOP);
	glVertex2d(gp[0].real, gp[0].imag);
	glVertex2d(gp[1].real, gp[1].imag);
	glVertex2d(gp[2].real, gp[2].imag);
	glVertex2d(gp[3].real, gp[3].imag);
	glVertex2d(gp[0].real, gp[0].imag);
	glEnd();
	glLineWidth(1.f);
}

void PGT_RenderJoint(pgJointObject* joint)
{
	pgDistanceJoint* pj = (pgDistanceJoint*)joint;
	glColor3f(1.f, 0.f, 0.f);
	glLineWidth(2.f);
	glBegin(GL_LINES);
	if (joint->body1 && (!joint->body2))
	{
		glVertex2d(joint->body1->vecPosition.real,joint->body1->vecPosition.imag);
		glVertex2d(pj->anchor2.real,pj->anchor2.imag);
	}
	else if(joint->body1 && joint->body2)
	{
		glVertex2d(joint->body1->vecPosition.real,joint->body1->vecPosition.imag);
		glVertex2d(joint->body2->vecPosition.real,joint->body2->vecPosition.imag);
	}
	
	glEnd();
	glLineWidth(1.f);
}
