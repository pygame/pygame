/*
  pygame physics - Pygame physics module

  Copyright (C) 2008 Zhang Fan

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Library General Public
  License as published by the Free Software Foundation; either
  version 2 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Library General Public License for more details.

  You should have received a copy of the GNU Library General Public
  License along with this library; if not, write to the Free
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include "pgVector2.h"
#include <assert.h>
#include "pgHelpFunctions.h"

int
DoubleFromObj (PyObject* obj, double* val)
{
    PyObject* floatobj;
    
    if (PyNumber_Check (obj))
    {
        if (!(floatobj = PyNumber_Float (obj)))
            return 0;
        *val = PyFloat_AsDouble (floatobj);
        Py_DECREF (floatobj);

        if (PyErr_Occurred ())
            return 0;
        return 1;
    }
    return 0;
}

int
IntFromObj (PyObject* obj, int* val)
{
    PyObject* intobj;
    long tmp;
    
    if (PyNumber_Check (obj))
    {
        if (!(intobj = PyNumber_Int (obj)))
            return 0;
        tmp = PyInt_AsLong (intobj);
        Py_DECREF (intobj);
        if (tmp == -1 && PyErr_Occurred ())
            return 0;
        *val = tmp;
        return 1;
    }
    return 0;
}

PyObject* FromPhysicsVector2ToPoint(PyVector2 v2)
{
	PyObject* tuple = PyTuple_New(2);

	PyObject* xnum = PyFloat_FromDouble (v2.real);
	PyObject* ynum = PyFloat_FromDouble (v2.imag);
	PyTuple_SetItem(tuple,0,xnum);
	PyTuple_SetItem(tuple,1,ynum);
	return tuple;
}


double PG_Clamp(double x,double low,double high)
{
	double t;
	assert(low<high);
	t = x<high?x:high;
	return t>low?t:low;
}
