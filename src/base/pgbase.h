/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners, 2008 Marcus von Appen

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
#ifndef _PYGAME_BASE_H_
#define _PYGAME_BASE_H_

#include "pgcompat.h"
#include "pgdefines.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Stream handling */

#define PYGAME_BASE_FIRSTSLOT 0
#define PYGAME_BASE_NUMSLOTS 15
#ifndef PYGAME_BASE_INTERNAL
#define PyExc_PyGameError ((PyObject*)PyGameBase_C_API[PYGAME_BASE_FIRSTSLOT])
#define DoubleFromObj                                                   \
    (*(int(*)(PyObject*, double*))PyGameBase_C_API[PYGAME_BASE_FIRSTSLOT+1])
#define IntFromObj                                                      \
    (*(int(*)(PyObject*, int*))PyGameBase_C_API[PYGAME_BASE_FIRSTSLOT+2])
#define UintFromObj                                                     \
    (*(int(*)(PyObject*, unsigned int*))PyGameBase_C_API[PYGAME_BASE_FIRSTSLOT+3])
#define DoubleFromSeqIndex                                              \
    (*(int(*)(PyObject*, Py_ssize_t, double*))PyGameBase_C_API[PYGAME_BASE_FIRSTSLOT+4])
#define IntFromSeqIndex                                                 \
    (*(int(*)(PyObject*, Py_ssize_t, int*))PyGameBase_C_API[PYGAME_BASE_FIRSTSLOT+5])
#define UintFromSeqIndex                                                \
    (*(int(*)(PyObject*, Py_ssize_t, unsigned int*))PyGameBase_C_API[PYGAME_BASE_FIRSTSLOT+6])
#define PointFromObject                                                 \
    (*(int(*)(PyObject*, int*, int*))PyGameBase_C_API[PYGAME_BASE_FIRSTSLOT+7])
#define SizeFromObject                                                  \
    (*(int(*)(PyObject*, pgint32*, pgint32*))PyGameBase_C_API[PYGAME_BASE_FIRSTSLOT+8])
#define FPointFromObject                                                \
    (*(int(*)(PyObject*, double*, double*))PyGameBase_C_API[PYGAME_BASE_FIRSTSLOT+9])
#define FSizeFromObject                                                 \
    (*(int(*)(PyObject*, double*, double*))PyGameBase_C_API[PYGAME_BASE_FIRSTSLOT+10])
#define ASCIIFromObject                                                 \
    (*(int(*)(PyObject*, char**, PyObject**))PyGameBase_C_API[PYGAME_BASE_FIRSTSLOT+11])
#define UTF8FromObject                                                  \
    (*(int(*)(PyObject*, char**, PyObject**))PyGameBase_C_API[PYGAME_BASE_FIRSTSLOT+12])
#define UlongFromObj                                                    \
    (*(int(*)(PyObject*, unsigned long*))PyGameBase_C_API[PYGAME_BASE_FIRSTSLOT+13])
#define LongFromObj                                                    \
    (*(int(*)(PyObject*, long*))PyGameBase_C_API[PYGAME_BASE_FIRSTSLOT+14])
#endif /* PYGAME_BASE_INTERNAL */

/*
 * C-only stream access wrapper with threading support. 
 */
typedef struct
{
    PyObject *read;
    PyObject *write;
    PyObject *seek;
    PyObject *tell;
    PyObject *close;
#ifdef WITH_THREAD
    PyThreadState *thread;
#endif
} CPyStreamWrapper;

#define PYGAME_STREAMWRAPPER_FIRSTSLOT \
    (PYGAME_BASE_FIRSTSLOT + PYGAME_BASE_NUMSLOTS)
#define PYGAME_STREAMWRAPPER_NUMSLOTS 15
#ifndef PYGAME_STREAMWRAPPER_INTERNAL
#define CPyStreamWrapper_New                                            \
    (*(CPyStreamWrapper*(*)(PyObject*)) PyGameBase_C_API[PYGAME_STREAMWRAPPER_FIRSTSLOT+0])
#define CPyStreamWrapper_Free                                           \
    (*(void(*)(CPyStreamWrapper*)) PyGameBase_C_API[PYGAME_STREAMWRAPPER_FIRSTSLOT+1])
#define CPyStreamWrapper_Read_Threaded                                  \
    (*(int(*)(CPyStreamWrapper*, void*, pguint32, pguint32, pguint32*)) \
        PyGameBase_C_API[PYGAME_STREAMWRAPPER_FIRSTSLOT+2])
#define CPyStreamWrapper_Read                                           \
    (*(int(*)(CPyStreamWrapper*, void*, pguint32, pguint32, pguint32*)) \
        PyGameBase_C_API[PYGAME_STREAMWRAPPER_FIRSTSLOT+3])
#define CPyStreamWrapper_Write_Threaded                                 \
    (*(int(*)(CPyStreamWrapper*, const void*, pguint32, pguint32, pguint32*)) \
        PyGameBase_C_API[PYGAME_STREAMWRAPPER_FIRSTSLOT+4])
#define CPyStreamWrapper_Write                                          \
    (*(int(*)(CPyStreamWrapper*, const void*, pguint32, pguint32, pguint32*)) \
        PyGameBase_C_API[PYGAME_STREAMWRAPPER_FIRSTSLOT+5])
#define CPyStreamWrapper_Seek_Threaded                                  \
    (*(int(*)(CPyStreamWrapper*, pgint32, int)) PyGameBase_C_API[PYGAME_STREAMWRAPPER_FIRSTSLOT+6])
#define CPyStreamWrapper_Seek                                         \
    (*(int(*)(CPyStreamWrapper*, pgint32, int)) PyGameBase_C_API[PYGAME_STREAMWRAPPER_FIRSTSLOT+7])
#define CPyStreamWrapper_Tell_Threaded                                  \
    (*(pgint32(*)(CPyStreamWrapper*)) PyGameBase_C_API[PYGAME_STREAMWRAPPER_FIRSTSLOT+8])
#define CPyStreamWrapper_Tell                                           \
    (*(pgint32(*)(CPyStreamWrapper*)) PyGameBase_C_API[PYGAME_STREAMWRAPPER_FIRSTSLOT+9])
#define CPyStreamWrapper_Close_Threaded                                 \
    (*(int(*)(CPyStreamWrapper*)) PyGameBase_C_API[PYGAME_STREAMWRAPPER_FIRSTSLOT+10])
#define CPyStreamWrapper_Close                                          \
    (*(int(*)(CPyStreamWrapper*)) PyGameBase_C_API[PYGAME_STREAMWRAPPER_FIRSTSLOT+11)
#define IsReadableStreamObj                                             \
    (*(int(*)(PyObject*))PyGameBase_C_API[PYGAME_STREAMWRAPPER_FIRSTSLOT+12])
#define IsWriteableStreamObj                                            \
    (*(int(*)(PyObject*))PyGameBase_C_API[PYGAME_STREAMWRAPPER_FIRSTSLOT+13])
#define IsReadWriteableStreamObj                                        \
    (*(int(*)(PyObject*))PyGameBase_C_API[PYGAME_STREAMWRAPPER_FIRSTSLOT+14])
#endif /* PYGAME_STREAMWRAPPER_INTERNAL */

typedef struct
{
    PyObject_HEAD
    /* RGBA */
    pgbyte r;
    pgbyte g;
    pgbyte b;
    pgbyte a;
} PyColor;
#define PYGAME_COLOR_FIRSTSLOT \
    (PYGAME_STREAMWRAPPER_FIRSTSLOT + PYGAME_STREAMWRAPPER_NUMSLOTS)
#define PYGAME_COLOR_NUMSLOTS 5
#ifndef PYGAME_COLOR_INTERNAL
#define PyColor_Type \
    (*(PyTypeObject*)PyGameBase_C_API[PYGAME_COLOR_FIRSTSLOT+0])
#define PyColor_Check(x)                                                \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyGameBase_C_API[PYGAME_COLOR_FIRSTSLOT+0]))
#define PyColor_New                                                     \
    (*(PyObject*(*)(pgbyte*)) PyGameBase_C_API[PYGAME_COLOR_FIRSTSLOT+1])
#define PyColor_NewFromNumber                                           \
    (*(PyObject*(*)(pguint32)) PyGameBase_C_API[PYGAME_COLOR_FIRSTSLOT+2])
#define PyColor_NewFromRGBA                                             \
    (*(PyObject*(*)(pgbyte,pgbyte,pgbyte,pgbyte)) PyGameBase_C_API[PYGAME_COLOR_FIRSTSLOT+3])
#define PyColor_AsNumber                                                \
    (*(pguint32(*)(PyObject*)) PyGameBase_C_API[PYGAME_COLOR_FIRSTSLOT+4])
#endif /* PYGAME_COLOR_INTERNAL */

typedef struct
{
    PyObject_HEAD
    pgint16 x;
    pgint16 y;
    pguint16 w;
    pguint16 h;
} PyRect;
#define PYGAME_RECT_FIRSTSLOT (PYGAME_COLOR_FIRSTSLOT + PYGAME_COLOR_NUMSLOTS)
#define PYGAME_RECT_NUMSLOTS 2
#ifndef PYGAME_RECT_INTERNAL
#define PyRect_Type \
    (*(PyTypeObject*)PyGameBase_C_API[PYGAME_RECT_FIRSTSLOT+0])
#define PyRect_Check(x)                                                 \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyGameBase_C_API[PYGAME_RECT_FIRSTSLOT+0]))
#define PyRect_New                                                      \
    (*(PyObject*(*)(pgint16,pgint16,pguint16,pguint16))PyGameBase_C_API[PYGAME_RECT_FIRSTSLOT+1])
#endif /* PYGAME_RECT_INTERNAL */

typedef struct
{
    PyObject_HEAD
    double x;
    double y;
    double w;
    double h;
} PyFRect;
#define PYGAME_FRECT_FIRSTSLOT (PYGAME_RECT_FIRSTSLOT + PYGAME_RECT_NUMSLOTS)
#define PYGAME_FRECT_NUMSLOTS 2
#ifndef PYGAME_FRECT_INTERNAL
#define PyFRect_Type \
    (*(PyTypeObject*)PyGameBase_C_API[PYGAME_FRECT_FIRSTSLOT+0])
#define PyFRect_Check(x)                                                 \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyGameBase_C_API[PYGAME_FRECT_FIRSTSLOT+0]))
#define PyFRect_New                                                      \
    (*(PyObject*(*)(double,double,double,double))PyGameBase_C_API[PYGAME_FRECT_FIRSTSLOT+1])
#endif /* PYGAME_FRECT_INTERNAL */

typedef int (*bufferunlock_func)(PyObject* object,PyObject* buffer);
typedef struct
{
    PyObject_HEAD
    PyObject         *dict;
    PyObject         *weakrefs;
    PyObject         *object;
    void             *buffer;
    Py_ssize_t        length;
    bufferunlock_func unlock_func;
} PyBufferProxy;
#define PyBufferProxy_AsBuffer(x) (((PyBufferProxy*)x)->buffer)
#define PYGAME_BUFFERPROXY_FIRSTSLOT                                    \
    (PYGAME_FRECT_FIRSTSLOT + PYGAME_FRECT_NUMSLOTS)
#define PYGAME_BUFFERPROXY_NUMSLOTS 2
#ifndef PYGAME_BUFFERPROXY_INTERNAL
#define PyBufferProxy_Type \
    (*(PyTypeObject*)PyGameBase_C_API[PYGAME_BUFFERPROXY_FIRSTSLOT+0])
#define PyBufferProxy_Check(x)                                          \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyGameBase_C_API[PYGAME_BUFFERPROXY_FIRSTSLOT+0]))
#define PyBufferProxy_New                                               \
    (*(PyObject*(*)(PyObject*,void*,Py_ssize_t,bufferunlock_func))PyGameBase_C_API[PYGAME_BUFFERPROXY_FIRSTSLOT+1])
#endif /* PYGAME_BUFFERPROXY_INTERNAL */

typedef struct
{
    PyObject_HEAD

    PyObject* (*get_width)(PyObject *self, void *closure);
    PyObject* (*get_height) (PyObject *self, void *closure);
    PyObject* (*get_size)(PyObject *self, void *closure);
    PyObject* (*get_pixels)(PyObject *self, void *closure);
    PyObject* (*blit)(PyObject *self, PyObject *args, PyObject *kwds); 
    PyObject* (*copy)(PyObject *self); 
} PySurface;
#define PYGAME_SURFACE_FIRSTSLOT                                        \
    (PYGAME_BUFFERPROXY_FIRSTSLOT + PYGAME_BUFFERPROXY_NUMSLOTS)
#define PYGAME_SURFACE_NUMSLOTS 2
#ifndef PYGAME_SURFACE_INTERNAL
#define PySurface_Type \
    (*(PyTypeObject*)PyGameBase_C_API[PYGAME_SURFACE_FIRSTSLOT+0])
#define PySurface_Check(x)                                              \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyGameBase_C_API[PYGAME_SURFACE_FIRSTSLOT+0]))
#define PySurface_New                                               \
    (*(PyObject*(*)(void))PyGameBase_C_API[PYGAME_SURFACE_FIRSTSLOT+1])
#endif /* PYGAME_SURFACE_INTERNAL */

typedef struct
{
    PyObject_HEAD

    PyObject* (*get_height)(PyObject *self, void *closure);
    PyObject* (*get_name)(PyObject *self, void *closure);
    PyObject* (*get_style)(PyObject *self, void *closure);
    int       (*set_style)(PyObject *self, PyObject *attr, void *closure);
    

    PyObject* (*get_size)(PyObject *self, PyObject *args, PyObject *kwds);
    PyObject* (*render)(PyObject *self, PyObject *args, PyObject *kwds);
    PyObject* (*copy)(PyObject *self);
} PyFont;

#define PYGAME_FONT_FIRSTSLOT   \
    (PYGAME_SURFACE_FIRSTSLOT + PYGAME_SURFACE_NUMSLOTS)
#define PYGAME_FONT_NUMSLOTS 2

#ifndef PYGAME_FONT_INTERNAL
#define PyFont_Type \
    (*(PyTypeObject*)PyGameBase_C_API[PYGAME_FONT_FIRSTSLOT+0])
#define PyFont_Check(x) \
    (PyObject_TypeCheck(x, \
        (PyTypeObject*)PyGameBase_C_API[PYGAME_FONT_FIRSTSLOT+0]))
#define PyFont_New \
    (*(PyObject*(*)(void))PyGameBase_C_API[PYGAME_FONT_FIRSTSLOT+1])
#endif /* PYGAME_FONT_INTERNAL */


/**
 * C API export.
 */
#ifdef PYGAME_INTERNAL
void **PyGameBase_C_API;
#else
static void **PyGameBase_C_API;
#endif

#define PYGAME_BASE_SLOTS \
    (PYGAME_FONT_FIRSTSLOT + PYGAME_FONT_NUMSLOTS)
#define PYGAME_BASE_ENTRY "_PYGAME_BASE_CAPI"

static int
import_pygame2_base (void)
{
    PyObject *_module = PyImport_ImportModule ("pygame2.base");
    if (_module != NULL)
    {
        PyObject *_capi = PyObject_GetAttrString(_module, PYGAME_BASE_ENTRY);
        if (!PyCObject_Check (_capi))
        {
            Py_DECREF (_module);
            return -1;
        }
        PyGameBase_C_API = (void**) PyCObject_AsVoidPtr (_capi);
        Py_DECREF (_capi);
        return 0;
    }
    return -1;
}

#ifdef __cplusplus
}
#endif

#endif /* _PYGAME_BASE_H_ */
