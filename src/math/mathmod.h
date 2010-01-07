/*
  pygame - Python Game Library

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
#ifndef _PYGAME_MATHMOD_H_
#define _PYGAME_MATHMOD_H_

#include <Python.h>

#define PYGAME_MATH_INTERNAL
#define PYGAME_MATHVECTOR_INTERNAL
#define PYGAME_MATHVECTOR2_INTERNAL
#define PYGAME_MATHVECTOR3_INTERNAL

/* Math operation types. */
#define OP_ADD            1
#define OP_IADD           2
#define OP_SUB            3
#define OP_ISUB           4
#define OP_MUL            5
#define OP_IMUL           6
#define OP_DIV            7
#define OP_IDIV           8
#define OP_FLOOR_DIV      9
#define OP_IFLOOR_DIV    10
#define OP_MOD           11
#define OP_ARG_REVERSE   32
#define OP_ARG_UNKNOWN   64
#define OP_ARG_VECTOR   128
#define OP_ARG_NUMBER   256

extern PyTypeObject PyVector_Type;
#define PyVector_Check(x) (PyObject_TypeCheck (x, &PyVector_Type))
PyObject* PyVector_New (Py_ssize_t dim);
PyObject* PyVector_NewFromSeq (PyObject *seq);
PyObject* PyVector_NewSpecialized (Py_ssize_t dim);

extern PyTypeObject PyVector2_Type;
#define PyVector2_Check(x) (PyObject_TypeCheck (x, &PyVector2_Type))
PyObject* PyVector2_New (double x, double y);

extern PyTypeObject PyVector3_Type;
#define PyVector3_Check(x) (PyObject_TypeCheck (x, &PyVector3_Type))
PyObject* PyVector3_New (double x, double y, double z);

#define IsSimpleNumber(x) (PyNumber_Check(x) && !PyComplex_Check(x))
#define IsVectorCompatible(x) (PyVector_Check(x) || PySequence_Check(x))

double _ScalarProduct (const double *coords1, const double *coords2,
    Py_ssize_t size);

void vector_export_capi (void **capi);
void vector2_export_capi (void **capi);
void vector3_export_capi (void **capi);

#endif /* _PYGAME_MATHMOD_H_ */
