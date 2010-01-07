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
#define PYGAME_MATH_INTERNAL

#include "pgbase.h"
#include "mathmod.h"
#include "pgmath.h"
#include "mathbase_doc.h"

static PyMethodDef _math_methods[] = {
    { NULL, NULL, 0, NULL },
};

double
_ScalarProduct (const double *coords1, const double *coords2, Py_ssize_t size)
{
    Py_ssize_t i;
    double ret = 0.f;
    for (i = 0; i < size; i++)
        ret += coords1[i] * coords2[i];
    return ret;
}

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_base (void)
#else
PyMODINIT_FUNC initbase (void)
#endif
{
    PyObject *mod = NULL;
    PyObject *c_api_obj;
    static void *c_api[PYGAME_MATH_SLOTS];

#ifdef IS_PYTHON_3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "base",
        DOC_BASE,
        -1,
        _math_methods,
        NULL, NULL, NULL, NULL
    };
#endif
    PyVector_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready (&PyVector_Type) < 0)
        goto fail;
    Py_INCREF (&PyVector_Type);
    PyVector2_Type.tp_base = &PyVector_Type; 
    if (PyType_Ready (&PyVector2_Type) < 0)
        goto fail;
    Py_INCREF (&PyVector2_Type);
    PyVector3_Type.tp_base = &PyVector_Type; 
    if (PyType_Ready (&PyVector3_Type) < 0)
        goto fail;
    Py_INCREF (&PyVector3_Type);    

#ifdef IS_PYTHON_3
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("base", _math_methods, DOC_BASE);
#endif
    if (!mod)
        goto fail;

    PyModule_AddObject (mod, "Vector", (PyObject *) &PyVector_Type);
    PyModule_AddObject (mod, "Vector2", (PyObject *) &PyVector2_Type);
    PyModule_AddObject (mod, "Vector3", (PyObject *) &PyVector3_Type);

    /* Export C API */
    vector_export_capi (c_api);
    vector2_export_capi (c_api);
    vector3_export_capi (c_api);

    c_api_obj = PyCObject_FromVoidPtr ((void *) c_api, NULL);
    if (c_api_obj)
        PyModule_AddObject (mod, PYGAME_MATH_ENTRY, c_api_obj);
        
    if (import_pygame2_base () < 0)
        goto fail;
    MODINIT_RETURN(mod);

fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
