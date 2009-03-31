/*
  pygame physics - Pygame physics module

  Copyright (C) 2008 Zhang Fan

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

#define PHYSICS_PHYSICSMOD_INTERNAL

#include "physicsmod.h"
#include "pgphysics.h"

#if PY_VERSION_HEX >= 0x03000000
PyMODINIT_FUNC PyInit_physics (void)
#else
PyMODINIT_FUNC initphysics (void)
#endif
{
    static void* c_api[PHYSICS_SLOTS];
    PyObject *mod = NULL, *c_api_obj;

#if PY_VERSION_HEX >= 0x03000000
    static struct PyModuleDef _physicsmodule = {
        PyModuleDef_HEAD_INIT, "physics", "", -1, NULL,
        NULL, NULL, NULL, NULL
    };
#endif
    
    /* Complete types */
    if (PyType_Ready (&PyWorld_Type) < 0)
        goto fail;
    if (PyType_Ready (&PyBody_Type) < 0)
        goto fail;
    PyShape_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready (&PyShape_Type) < 0)
        goto fail;
    PyRectShape_Type.tp_base = &PyShape_Type;
    if (PyType_Ready (&PyRectShape_Type) < 0)
        goto fail;
    PyJoint_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready (&PyJoint_Type) < 0)
        goto fail;
    PyContact_Type.tp_base = &PyJoint_Type;
    if (PyType_Ready (&PyContact_Type) < 0)
        goto fail;

    Py_INCREF (&PyWorld_Type);
    Py_INCREF (&PyBody_Type);
    Py_INCREF (&PyJoint_Type);
    Py_INCREF (&PyContact_Type);
    Py_INCREF (&PyShape_Type);
    Py_INCREF (&PyRectShape_Type);

#if PY_VERSION_HEX < 0x03000000
    mod = Py_InitModule3 ("physics", NULL, NULL);
#else
    mod = PyModule_Create (&_physicsmodule);
#endif
    if (!mod)
        goto fail;

    PyModule_AddObject (mod, "World", (PyObject *) &PyWorld_Type);
    PyModule_AddObject (mod, "Body", (PyObject *) &PyBody_Type);
    PyModule_AddObject (mod, "Joint", (PyObject *) &PyJoint_Type);
    PyModule_AddObject (mod, "Contact", (PyObject *) &PyContact_Type);
    PyModule_AddObject (mod, "Shape", (PyObject *) &PyShape_Type);
    PyModule_AddObject (mod, "RectShape", (PyObject *) &PyRectShape_Type);

    /* Export the C API */
    math_export_capi (c_api);
    aabbox_export_capi (c_api);
    body_export_capi (c_api);
    shape_export_capi (c_api);
    rectshape_export_capi (c_api);
    joint_export_capi (c_api);
    contact_export_capi (c_api);
    world_export_capi (c_api);

    c_api_obj = PyCObject_FromVoidPtr ((void *) c_api, NULL);
    if (c_api_obj)
        PyModule_AddObject (mod, PHYSICS_ENTRY, c_api_obj);
    if (import_pygame2_base () < 0)
        goto fail;
    MODINIT_RETURN(mod);

fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
