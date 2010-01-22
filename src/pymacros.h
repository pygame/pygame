/**
 * Useful python macros.
 */

#include <Python.h>

#define ADD_OBJ_OR_FAIL(mod,name,type,mark)                       \
    Py_INCREF(&type);                                             \
    if (PyModule_AddObject (mod, name, (PyObject *) &type) == -1) \
    {                                                             \
        Py_DECREF(&type);                                         \
        goto mark;                                                \
    }
