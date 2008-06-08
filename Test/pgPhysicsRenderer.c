#include "pgPhysicsRenderer.h"
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
}

void PGT_RenderBody(pgBodyObject* body)
{
	glColor3f(1,0,0);
	glPointSize(20);
	glBegin(GL_POINTS);
		glVertex2d(body->vecPosition.real,body->vecPosition.imag);
	glEnd();
}