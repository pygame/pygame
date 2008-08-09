#ifdef WIN32
#include <windows.h>
#endif
#include <GL/glut.h>
#include "pgPhysicsRenderer.h"

int RENDER_AABB = 0;

void PGT_RenderWorld(PyWorldObject* world)
{
	Py_ssize_t size = PyList_Size(world->bodyList);
	Py_ssize_t i;
	for (i = 0;i < size;i++)
	{
		PyBodyObject* body = (PyBodyObject*)(PyList_GetItem(world->bodyList,i));
		PGT_RenderBody(body);
	}

	size = PyList_Size(world->jointList);
	for (i = 0;i < size;i++)
	{
		PyJointObject* joint = (PyJointObject*)(PyList_GetItem(world->jointList,i));
		PGT_RenderJoint(joint);
	}
}

void PGT_RenderAABB(PyBodyObject* body)
{
	PyVector2 p[4];
	AABBBox* box;

	box = &(((PyShapeObject*)body->shape)->box);
	
	PyVector2_Set(p[0], box->left, box->bottom);
	PyVector2_Set(p[1], box->right, box->bottom);
	PyVector2_Set(p[2], box->right, box->top);
	PyVector2_Set(p[3], box->left, box->top);

	glColor3f(0.f, 1.f, 1.f);
	glEnable(GL_LINE_STIPPLE);
	glLineWidth(1.f);
	glLineStipple(1, 0xF0F0);
	glBegin(GL_LINE_LOOP);
	glVertex2d(p[0].real, p[0].imag);
	glVertex2d(p[1].real, p[1].imag);
	glVertex2d(p[2].real, p[2].imag);
	glVertex2d(p[3].real, p[3].imag);
	glVertex2d(p[0].real, p[0].imag);
	glEnd();
	glDisable(GL_LINE_STIPPLE);
}

void PGT_RenderBody(PyBodyObject* body)
{
	PyVector2 gp[4];
	int i;
	PyRectShapeObject* rect = (PyRectShapeObject*)body->shape;

	// Evil hack for the threads.
	import_physics ();
	
	gp[0] = PyBody_GetGlobalPos((PyObject*)body, rect->bottomleft);
	gp[1] = PyBody_GetGlobalPos((PyObject*)body, rect->bottomright);
	gp[2] = PyBody_GetGlobalPos((PyObject*)body, rect->topright);
	gp[3] = PyBody_GetGlobalPos((PyObject*)body, rect->topleft);

	glColor3f(1.f, 1.f, 0.f);
	glLineWidth(1.f);
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

void PGT_RenderJoint(PyJointObject* joint)
{
	PyDistanceJointObject* pj = (PyDistanceJointObject*)joint;
	glColor3f(1.f, 0.f, 0.f);
	glLineWidth(1.f);
	glBegin(GL_LINES);
	if (joint->body1 && (!joint->body2))
	{
		PyVector2 pos = PyBody_GetGlobalPos(joint->body1, pj->anchor1);
		glVertex2d(pos.real,pos.imag);
		glVertex2d(pj->anchor2.real,pj->anchor2.imag);
	}
	else if(joint->body1 && joint->body2)
	{
		PyVector2 pos1 = PyBody_GetGlobalPos(joint->body1, pj->anchor1);
		PyVector2 pos2 = PyBody_GetGlobalPos(joint->body2, pj->anchor2);
		glVertex2d(pos1.real,pos1.imag);
		glVertex2d(pos2.real,pos2.imag);
	}
	
	glEnd();
	glLineWidth(1.f);
}
