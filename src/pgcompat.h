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

#ifndef IS_PYTHON_3
#   define IS_PYTHON_3
#endif

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
#define PyNumber_Int PyNumber_Long

/* Weakrefs flags changed in 3.x */
#define Py_TPFLAGS_HAVE_WEAKREFS 0
/* No more type checks */
#define Py_TPFLAGS_CHECKTYPES 0

/* Module creation and type heads differ a lot. */
#define MODINIT_RETURN(x) return(x)
#define TYPE_HEAD(x,y) PyVarObject_HEAD_INIT(x,y)

#else /* PY_VERSION_HEX >= 0x03000000 */

/* Used for the changed PyMODINIT_FUNC defines in 3.0 and 
 * Python type head init code.
 */
#define MODINIT_RETURN(x) return
#define TYPE_HEAD(x,y)                          \
    PyObject_HEAD_INIT(x)                       \
    0,

#endif /* PY_VERSION_HEX >= 0x03000000 */ 

/**
 * Text interfaces. Those assume the text is pure ASCII or UTF-8
 */
#if PY_VERSION_HEX >= 0x03000000
#define Text_Check PyUnicode_Check
#define Text_FromUTF8 PyUnicode_FromString
#define Text_FromUTF8AndSize PyUnicode_FromStringAndSize
#define Text_FromFormat PyUnicode_FromFormat
#define Text_GetSize PyUnicode_GetSize
#define Text_GET_SIZE PyUnicode_GET_SIZE

#define Bytes_Check PyBytes_Check
#define Bytes_Size PyBytes_Size
#define Bytes_AsString PyBytes_AsString
#define Bytes_AsStringAndSize PyBytes_AsStringAndSize
#define Bytes_FromStringAndSize PyBytes_FromStringAndSize
#define Bytes_AS_STRING PyBytes_AS_STRING
#define Bytes_GET_SIZE PyBytes_GET_SIZE

#define IsTextObj(x) (PyUnicode_Check(x) || PyBytes_Check(x))
#define IsFileObj(x) (!PyNumber_Check(x) && PyObject_AsFileDescriptor(x) != -1)

#else  /* PY_VERSION_HEX >= 0x03000000 */

#define Text_Check PyString_Check
#define Text_FromUTF8 PyString_FromString
#define Text_FromUTF8AndSize PyString_FromStringAndSize
#define Text_FromFormat PyString_FromFormat
#define Text_GetSize PyString_GetSize
#define Text_GET_SIZE PyString_GET_SIZE

#define Bytes_Check PyString_Check
#define Bytes_Size PyString_Size
#define Bytes_AsString PyString_AsString
#define Bytes_AsStringAndSize PyString_AsStringAndSize
#define Bytes_FromStringAndSize PyString_FromStringAndSize
#define Bytes_AS_STRING PyString_AS_STRING
#define Bytes_GET_SIZE PyString_GET_SIZE

#define IsTextObj(x) (PyString_Check(x) || PyUnicode_Check(x))
#define IsFileObj(x) PyFile_Check(x)

#define PyState_FindModule(obj) (NULL)

#endif  /* PY_VERSION_HEX >= 0x03000000 */ 

#endif /* _PYGAME_PYTHONCOMPAT_H_ */
