#include "pgVector2.h"
#include "pgBodyObject.h"
#include "pgShapeObject.h"

//#define FLOAT_TO_INT_MUL 10
//
////these functions are for pygame rendering
//static PyObject * _pg_getPointListFromBody(PyObject *self, PyObject *args)
//{
//	pgBodyObject* body;
//	int i;
//	PyListObject* list;
//	
//	if (!PyArg_ParseTuple(args,"O",&body))
//	{
//		PyErr_SetString(PyExc_ValueError,"arg is not body type");
//		return NULL;
//	}
//	else
//	{
//		if (body->shape == NULL)
//		{
//			PyErr_SetString(PyExc_ValueError,"Shape is NULL");
//			return NULL;
//		}
//		list = (PyListObject*)PyList_New(4);
//		for (i = 0;i < 4;i++)
//		{
//			pgVector2* pVertex = &(((pgRectShape*)(body->shape))->point[i]);
//			pgVector2 golVertex = PG_GetGlobalPos(body,pVertex);
//			PyTupleObject* tuple = (PyTupleObject*)PyTuple_New(2);
//			
//			long ix = golVertex.real * FLOAT_TO_INT_MUL;
//			long iy = golVertex.imag * FLOAT_TO_INT_MUL;
//			PyIntObject* xnum = PyInt_FromLong(ix);
//			PyIntObject* ynum = PyInt_FromLong(iy);
//			PyTuple_SetItem((PyObject*)tuple,0,xnum);
//			PyTuple_SetItem((PyObject*)tuple,1,ynum);
//			PyList_SetItem((PyObject*)list,i,(PyObject*)tuple);
//		}
//		return (PyObject*)list;
//	}
//}
//
////static PyObject * _pg_getPointListFromBody(PyObject *self, PyObject *args)
////{
////	pgRectShape* shape;
////	int i;
////	PyListObject* list;
////
////	if (!PyArg_ParseTuple(args,"O",&shape))
////	{
////		PyErr_SetString(PyExc_ValueError,"arg is not shape type");
////		return NULL;
////	}
////	else
////	{
////		if (shape == NULL)
////		{
////			PyErr_SetString(PyExc_ValueError,"Shape is NULL");
////			return NULL;
////		}
////		list = (PyListObject*)PyList_New(4);
////		for (i = 0;i < 4;i++)
////		{|
////
////			pgVector2* pVertex = &(shape->point[i]);
////			PyTupleObject* tuple = (PyTupleObject*)PyTuple_New(2);
////			long ix = pVertex->real * FLOAT_TO_INT_MUL;
////			long iy = pVertex->imag * FLOAT_TO_INT_MUL;
////			PyIntObject* xnum = PyInt_FromLong(ix);
////			PyIntObject* ynum = PyInt_FromLong(iy);
////			PyTuple_SetItem((PyObject*)tuple,0,xnum);
////			PyTuple_SetItem((PyObject*)tuple,1,ynum);
////			PyList_SetItem((PyObject*)list,i,(PyObject*)tuple);
////		}
////		return (PyObject*)list;
////	}
////}
//
PyMethodDef pgHelpMethods[] = {
	//{"get_point_list",_pg_getPointListFromBody,METH_VARARGS,""	},
	{NULL, NULL, 0, NULL} 
};