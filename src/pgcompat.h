#ifndef _PYGAME_PYTHONCOMPAT_H_
#define _PYGAME_PYTHONCOMPAT_H_

#include <Python.h>

/* Python 2.4 compatibility */
#if PY_VERSION_HEX < 0x02050000
typedef int Py_ssize_t;
#define PY_SSIZE_T_MAX INT_MAX
#define PY_SSIZE_T_MIN INT_MIN

typedef inquiry lenfunc;
typedef intargfunc ssizeargfunc;
typedef intobjargproc ssizeobjargproc;
typedef intintargfunc ssizessizeargfunc;
typedef intintobjargproc ssizessizeobjargproc;
typedef getreadbufferproc readbufferproc;
typedef getwritebufferproc writebufferproc;
typedef getsegcountproc segcountproc;
typedef getcharbufferproc charbufferproc;

#define PyIndex_Check(op) 0
#define PyInt_FromSsize_t(x) (PyInt_FromLong(x))
#endif /* PY_VERSION_HEX < 0x02050000 */

/* Python 3.x compatibility */
#if PY_VERSION_HEX >= 0x03000000
/* Define some aliases for the removed PyInt_* functions */
#define PyInt_Check(op) PyLong_Check(op)
#define PyInt_FromString PyLong_FromString
#define PyInt_FromUnicode PyLong_FromUnicode
#define PyInt_FromLong PyLong_FromLong
#define PyInt_FromSize_t PyLong_FromSize_t
#define PyInt_FromSsize_t PyLong_FromSsize_t
#define PyInt_AsLong PyLong_AsLong
#define PyInt_AsSsize_t PyLong_AsSsize_t
#define PyInt_AsUnsignedLongMask PyLong_AsUnsignedLongMask
#define PyInt_AsUnsignedLongLongMask PyLong_AsUnsignedLongLongMask
#define PyInt_AS_LONG PyLong_AS_LONG

/* Weakrefs flags changed in 3.x */
#define Py_TPFLAGS_HAVE_WEAKREFS 0

/* Module creation and type heads differ a lot. */
#define MODINIT_RETURN(x) return(x)
#define TYPE_HEAD(x,y) PyVarObject_HEAD_INIT(x,y)

/* TODO: REMOVE THOSE!
 * They are not safe - instead we will need the correct functions and
 * converters or whatever for them.
 */
#define PyString_AsString PyUnicode_AS_DATA
#define PyString_FromFormat PyUnicode_FromFormat
#define PyString_FromString PyUnicode_FromString
#define PyString_FromStringAndSize PyUnicode_FromStringAndSize
#define PyString_Check PyUnicode_Check
#define PyString_GET_SIZE PyUnicode_GET_SIZE
#define PyString_Size PyUnicode_GetSize
#define PyString_ConcatAndDel(x,y)

#define PyFile_Check(x) 1
#define PyFile_AsFile(x) PyObject_AsFileDescriptor(x)

/* TODO END */

#else /* PY_VERSION_HEX >= 0x03000000 */

/* Used for the changed PyMODINIT_FUNC defines in 3.0 and 
 * Python type head init code.
 */
#define MODINIT_RETURN(x) return
#define TYPE_HEAD(x,y)                          \
    PyObject_HEAD_INIT(x)                       \
    0,

#endif /* PY_VERSION_HEX >= 0x03000000 */ 

#endif /* _PYGAME_PYTHONCOMPAT_H_ */
