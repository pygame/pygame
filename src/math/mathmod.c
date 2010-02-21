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

#include "pymacros.h"
#include "pgbase.h"
#include "mathmod.h"
#include "pgmath.h"
#include "mathbase_doc.h"

static PyObject* _frompolar (PyObject* mod, PyObject *args);
static PyObject* _fromspherical (PyObject* mod, PyObject *args);

static PyMethodDef _math_methods[] = {
    { "vector_from_polar", _frompolar, METH_VARARGS,
      DOC_BASE_VECTOR_FROM_POLAR },
    { "vector_from_spherical", _fromspherical, METH_VARARGS,
      DOC_BASE_VECTOR_FROM_SPHERICAL },
    { NULL, NULL, 0, NULL },
};

static PyObject*
_frompolar (PyObject* mod, PyObject *args)
{
    double r, phi, c1, c2;

    if (!PyArg_ParseTuple (args, "dd:vector_from_polar", &r, &phi))
        return NULL;
    c1 = r * cos (phi);
    c2 = r * sin (phi);
    return PyVector2_New (c1, c2);
}

static PyObject*
_fromspherical (PyObject* mod, PyObject *args)
{
    double r, phi, theta, c1, c2, c3;

    if (!PyArg_ParseTuple (args, "ddd:vector_from_spherical", &r, &theta, &phi))
        return NULL;

    c1 = r * sin (theta) * cos (phi);
    c2 = r * sin (theta) * sin (phi);
    c3 = r * cos (theta);

    return PyVector3_New (c1, c2, c3);
}

/* Internally used methods */
double
_ScalarProduct (const double *coords1, const double *coords2, Py_ssize_t size)
{
    Py_ssize_t i;
    double ret = 0.f;
    for (i = 0; i < size; i++)
        ret += coords1[i] * coords2[i];
    return ret;
}

/* C API */
double*
VectorCoordsFromObj (PyObject *object, Py_ssize_t *dims)
{
    double *coords= NULL;

    if (!object || !dims)
    {
        PyErr_SetString (PyExc_ValueError, "arguments must not be NULL");
        return NULL;
    }

    if (PyVector_Check (object))
    {
        *dims = ((PyVector*)object)->dim;
        coords = PyMem_New (double, *dims);
        if (!coords)
            return NULL;
        memcpy (coords, ((PyVector*)object)->coords, sizeof (double) * (*dims));
        return coords;
    }
    else if (PySequence_Check (object))
    {
        Py_ssize_t i;

        *dims = PySequence_Size (object);
        if ((*dims) < 2)
        {
            PyErr_SetString (PyExc_ValueError,
                "sequence must be greater than 1");
            return NULL;
        }
        coords = PyMem_New (double, *dims);
        if (!coords)
            return NULL;

        for (i = 0; i < (*dims); i++)
        {
            if (!DoubleFromSeqIndex (object, i, &(coords[i])))
            {
                PyMem_Free (coords);
                return NULL;
            }
        }
        return coords;
    }
    else
    {
        PyErr_SetString (PyExc_TypeError,
            "object must be a vector or sequence");
        return NULL;
    }
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
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("base", _math_methods, DOC_BASE);
#endif
    if (!mod)
        goto fail;

    PyVectorIter_Type.tp_new = &PyType_GenericNew;
    PyVectorIter_Type.tp_iter = &PyObject_SelfIter;
    if (PyType_Ready (&PyVectorIter_Type) < 0)
        goto fail;
    if (PyType_Ready (&PyVector_Type) < 0)
        goto fail;
    PyVector2_Type.tp_base = &PyVector_Type; 
    if (PyType_Ready (&PyVector2_Type) < 0)
        goto fail;
    PyVector3_Type.tp_base = &PyVector_Type; 
    if (PyType_Ready (&PyVector3_Type) < 0)
        goto fail;

    ADD_OBJ_OR_FAIL (mod, "Vector", PyVector_Type, fail);
    ADD_OBJ_OR_FAIL (mod, "Vector2", PyVector2_Type, fail);
    ADD_OBJ_OR_FAIL (mod, "Vector3", PyVector3_Type, fail);

    /* Export C API */
    c_api[PYGAME_MATH_FIRSTSLOT] = (void *)VectorCoordsFromObj;

    vector_export_capi (c_api);
    vector2_export_capi (c_api);
    vector3_export_capi (c_api);

    c_api_obj = PyCObject_FromVoidPtr ((void *) c_api, NULL);
    if (c_api_obj)
    {
        if (PyModule_AddObject (mod, PYGAME_MATH_ENTRY, c_api_obj) == -1)
        {
            Py_DECREF (c_api_obj);
            goto fail;
        }
    }
   
    if (import_pygame2_base () < 0)
        goto fail;
    MODINIT_RETURN(mod);

fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
