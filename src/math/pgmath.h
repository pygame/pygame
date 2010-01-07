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
#ifndef _PYGAME_MATH_H_
#define _PYGAME_MATH_H_

#include "pgcompat.h"
#include "pgdefines.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PYGAME_MATH_FIRSTSLOT 0
#define PYGAME_MATH_NUMSLOTS 0
#ifndef PYGAME_MATH_INTERNAL
#endif /* PYGAME_MATH_INTERNAL */

typedef struct
{
    PyObject_HEAD
    
    double     *coords;   /* Coordinates */
    Py_ssize_t  dim;      /* Dimension of the vector */
    double      epsilon;  /* Small value for comparisons */
} PyVector;

#define PYGAME_MATHVECTOR_FIRSTSLOT (PYGAME_MATH_FIRSTSLOT + PYGAME_MATH_NUMSLOTS)
#define PYGAME_MATHVECTOR_NUMSLOTS 4
#ifndef PYGAME_MATHVECTOR_INTERNAL
#define PyVector_Type \
    (*(PyTypeObject*)PyGameMath_C_API[PYGAME_MATHVECTOR_FIRSTSLOT+0])
#define PyVector_Check(x)                                                   \
    (PyObject_TypeCheck(x,                                                  \
        (PyTypeObject*)PyGameMath_C_API[PYGAME_MATHVECTOR_FIRSTSLOT+0]))
#define PyVector_New                                                        \
    (*(PyObject*(*)(Py_ssize_t))PyGameMath_C_API[PYGAME_MATHVECTOR_FIRSTSLOT+1])
#define PyVector_NewFromSeq                                                 \
    (*(PyObject*(*)(PyObject*))PyGameMath_C_API[PYGAME_MATHVECTOR_FIRSTSLOT+2])
#define PyVector_NewSpecialized                                             \
    (*(PyObject*(*)(Py_ssize_t))PyGameMath_C_API[PYGAME_MATHVECTOR_FIRSTSLOT+3])
#endif /* PYGAME_MATHVECTOR_INTERNAL */

typedef struct
{
    PyVector vector;
} PyVector2;
#define PYGAME_MATHVECTOR2_FIRSTSLOT \
    (PYGAME_MATHVECTOR_FIRSTSLOT + PYGAME_MATHVECTOR_NUMSLOTS)
#define PYGAME_MATHVECTOR2_NUMSLOTS 2
#ifndef PYGAME_MATHVECTOR2_INTERNAL
#define PyVector2_Type \
    (*(PyTypeObject*)PyGameMath_C_API[PYGAME_MATHVECTOR2_FIRSTSLOT+0])
#define PyVector2_Check(x)                                                   \
    (PyObject_TypeCheck(x,                                                   \
        (PyTypeObject*)PyGameMath_C_API[PYGAME_MATHVECTOR2_FIRSTSLOT+0]))
#define PyVector2_New                                                        \
    (*(PyObject*(*)(double,double))PyGameMath_C_API[PYGAME_MATHVECTOR2_FIRSTSLOT+1])
#endif /* PYGAME_MATHVECTOR2_INTERNAL */

typedef struct
{
    PyVector vector;
} PyVector3;
#define PYGAME_MATHVECTOR3_FIRSTSLOT \
    (PYGAME_MATHVECTOR2_FIRSTSLOT + PYGAME_MATHVECTOR2_NUMSLOTS)
#define PYGAME_MATHVECTOR3_NUMSLOTS 2
#ifndef PYGAME_MATHVECTOR3_INTERNAL
#define PyVector3_Type \
    (*(PyTypeObject*)PyGameMath_C_API[PYGAME_MATHVECTOR3_FIRSTSLOT+0])
#define PyVector3_Check(x)                                                   \
    (PyObject_TypeCheck(x,                                                   \
        (PyTypeObject*)PyGameMath_C_API[PYGAME_MATHVECTOR3_FIRSTSLOT+0]))
#define PyVector3_New                                                        \
    (*(PyObject*(*)(double,double,double))PyGameMath_C_API[PYGAME_MATHVECTOR3_FIRSTSLOT+1])
#endif /* PYGAME_MATHVECTOR2_INTERNAL */
/**
 * C API export.
 */
#ifdef PYGAME_INTERNAL
void **PyGameMath_C_API;
#else
static void **PyGameMath_C_API;
#endif

#define PYGAME_MATH_SLOTS \
    (PYGAME_MATHVECTOR3_FIRSTSLOT + PYGAME_MATHVECTOR3_NUMSLOTS)
#define PYGAME_MATH_ENTRY "_PYGAME_MATH_CAPI"

static int
import_pygame2_math (void)
{
    PyObject *_module = PyImport_ImportModule ("pygame2.math");
    if (_module != NULL)
    {
        PyObject *_capi = PyObject_GetAttrString(_module, PYGAME_MATH_ENTRY);
        if (!PyCObject_Check (_capi))
        {
            Py_DECREF (_module);
            return -1;
        }
        PyGameMath_C_API = (void**) PyCObject_AsVoidPtr (_capi);
        Py_DECREF (_capi);
        return 0;
    }
    return -1;
}

#ifdef __cplusplus
}
#endif

#endif /* _PYGAME_MATH_H_ */
