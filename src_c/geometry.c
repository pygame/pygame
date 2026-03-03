#include "circle.c"
#include "line.c"
#include "geometry_common.c"

static PyMethodDef geometry_methods[] = {{NULL, NULL, 0, NULL}};

MODINIT_DEFINE(geometry)
{
    PyObject *module, *apiobj;
    static void *c_api[PYGAMEAPI_GEOMETRY_NUMSLOTS];

    static struct PyModuleDef _module = {
        .m_base = PyModuleDef_HEAD_INIT,
        .m_name = "geometry",
        .m_doc = "Module for the Line, Circle and Polygon objects\n",
        .m_size = -1,
        .m_methods = geometry_methods,
    };

    import_pygame_base();
    if (PyErr_Occurred()) {
        return NULL;
    }

    import_pygame_rect();
    if (PyErr_Occurred()) {
        return NULL;
    }

    module = PyModule_Create(&_module);
    if (!module) {
        return NULL;
    }

    if (PyModule_AddType(module, &pgCircle_Type)) {
        Py_DECREF(module);
        return NULL;
    }

    if (PyModule_AddType(module, &pgLine_Type)) {
        Py_DECREF(module);
        return NULL;
    }

    c_api[0] = &pgCircle_Type;
    c_api[1] = &pgLine_Type;
    apiobj = encapsulate_api(c_api, "geometry");
    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj)) {
        Py_XDECREF(apiobj);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
