#include "pgPhysicsRenderer.h"
#include "pgShapeObject.h"
#include "pgAABBBox.h"
#include <gl/glut.h>

int RENDER_AABB = 1;

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

void PGT_RenderAABB(pgBodyObject* body)
{
	pgVector2 p[4], gp[4];
	int i;
	pgAABBBox* box;

	box = &(body->shape->box);
	
	PG_Set_Vector2(p[0], box->left, box->bottom);
	PG_Set_Vector2(p[1], box->right, box->bottom);
	PG_Set_Vector2(p[2], box->right, box->top);
	PG_Set_Vector2(p[3], box->left, box->top);
	for(i = 0; i < 4; ++i)
		gp[i] = PG_GetGlobalPos(body, &p[i]);

	glColor3f(0.f, 1.f, 1.f);
	glEnable(GL_LINE_STIPPLE);
	glLineWidth(1.f);
	glLineStipple(1, 0xF0F0);
	glBegin(GL_LINE_LOOP);
	glVertex2d(gp[0].real, gp[0].imag);
	glVertex2d(gp[1].real, gp[1].imag);
	glVertex2d(gp[2].real, gp[2].imag);
	glVertex2d(gp[3].real, gp[3].imag);
	glVertex2d(gp[0].real, gp[0].imag);
	glEnd();
	glDisable(GL_LINE_STIPPLE);
}

void PGT_RenderBody(pgBodyObject* body)
{
	pgVector2 gp[4];
	int i;
	pgRectShape* rect = (pgRectShape*)body->shape;

	for(i = 0; i < 4; ++i)
		gp[i] = PG_GetGlobalPos(body, &(rect->point[i]));

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

	if(RENDER_AABB)
		PGT_RenderAABB(body);
}

void PGT_RenderJoint(pgJointObject* joint)
{
	pgDistanceJointObject* pj = (pgDistanceJointObject*)joint;
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
