/*
  pygame - Python Game Library
  Copyright (C) 2008 Marcus von Appen

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
#ifndef _PYGAME_BASE_INTERNALS_H_
#define _PYGAME_BASE_INTERNALS_H_

#include "pgcompat.h"
#include "pgtypes.h"

#define PYGAME_BASE_INTERNAL
#define PYGAME_COLOR_INTERNAL
#define PYGAME_RECT_INTERNAL
#define PYGAME_FRECT_INTERNAL
#define PYGAME_BUFFERPROXY_INTERNAL
#define PYGAME_STREAMWRAPPER_INTERNAL
#define PYGAME_SURFACE_INTERNAL
#define PYGAME_FONT_INTERNAL

typedef struct {
    PyObject* error;
} _BaseState;

#ifdef IS_PYTHON_3
extern struct PyModuleDef _basemodule;
#define BASE_MOD_STATE(mod) ((_BaseState*)PyModule_GetState(mod))
#define BASE_STATE BASE_MOD_STATE(PyState_FindModule(&_basemodule))
#else
extern _BaseState _modstate;
#define BASE_MOD_STATE(mod) (&_modstate)
#define BASE_STATE BASE_MOD_STATE(NULL)
#endif
#define PyExc_PyGameError (BASE_STATE->error)

int DoubleFromObj (PyObject* obj, double* val);
int IntFromObj (PyObject* obj, int* val);
int UintFromObj (PyObject* obj, unsigned int* val);
int IntFromSeqIndex (PyObject* obj, Py_ssize_t _index, int* val);
int UintFromSeqIndex (PyObject* obj, Py_ssize_t _index, unsigned int* val);
int DoubleFromSeqIndex (PyObject* obj, Py_ssize_t _index, double* val);
int PointFromObject (PyObject* obj, int *x, int *y);
int SizeFromObject (PyObject* obj, pgint32 *w, pgint32 *h);
int FPointFromObject (PyObject* obj, double *x, double *y);
int FSizeFromObject (PyObject* obj, double *w, double *h);
int ASCIIFromObject (PyObject *obj, char **text, PyObject **freeme);
int UTF8FromObject (PyObject *obj, char **text, PyObject **freeme);
int LongFromObj (PyObject* obj, long* val);
unsigned long UlongFromObj (PyObject* obj, long* val);
int ColorFromObj (PyObject *obj, pguint32 *val);

extern PyTypeObject PyColor_Type;
#define PyColor_Check(x) (PyObject_TypeCheck(x, &PyColor_Type))
void color_export_capi (void **capi);

extern PyTypeObject PyRect_Type;
#define PyRect_Check(x) (PyObject_TypeCheck(x, &PyRect_Type))
PyObject* PyRect_New (pgint16 x, pgint16 y, pguint16 w, pguint16 h);
void rect_export_capi (void **capi);

extern PyTypeObject PyFRect_Type;
#define PyFRect_Check(x) (PyObject_TypeCheck(x, &PyFRect_Type))
PyObject* PyFRect_New (double x, double y, double w, double h);
void floatrect_export_capi (void **capi);

extern PyTypeObject PyBufferProxy_Type;
#define PyBufferProxy_Check(x) (PyObject_TypeCheck(x, &PyBufferProxy_Type))
void bufferproxy_export_capi (void **capi);

extern PyTypeObject PySurface_Type;
#define PySurface_Check(x) (PyObject_TypeCheck(x, &PySurface_Type))
PyObject* PySurface_New (void);
void surface_export_capi (void **capi);

extern PyTypeObject PyFont_Type;
#define PyFont_Check(x) (PyObject_TypeCheck(x, &PyFont_Type))
PyObject* PyFont_New (void);
void font_export_capi (void **capi);

void streamwrapper_export_capi (void **capi);

#endif /* _PYGAME_BASE_INTERNALS_H_ */
