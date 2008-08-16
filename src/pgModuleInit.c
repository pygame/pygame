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

#include "pgDeclare.h"
#include "pgVector2.h"
#include "pgWorldObject.h"
#include "pgBodyObject.h"
#include "pgJointObject.h"
#include "pgShapeObject.h"

PyMODINIT_FUNC
initphysics(void) 
{
    PyObject* mod;
    PyObject* c_api_obj;
    static void *c_api[PHYSICS_API_SLOTS];

    /* Check the types. */
    if (PyType_Ready(&PyWorld_Type) < 0)
        return;
    if (PyType_Ready(&PyBody_Type) < 0)
        return;
    if (PyType_Ready(&PyJoint_Type) < 0)
        return;
    PyDistanceJoint_Type.tp_base = &PyJoint_Type;
    if (PyType_Ready(&PyDistanceJoint_Type) < 0)
        return;
	PyRevoluteJoint_Type.tp_base = &PyJoint_Type;
	if (PyType_Ready(&PyRevoluteJoint_Type) < 0)
		return;
    PyContact_Type.tp_base = &PyJoint_Type;
    if (PyType_Ready(&PyContact_Type) < 0)
        return;
    if (PyType_Ready(&PyShape_Type) < 0)
        return;
    PyRectShape_Type.tp_base = &PyShape_Type;
    if (PyType_Ready(&PyRectShape_Type) < 0)
        return;

    /* Increase their ref counts. */
    Py_INCREF (&PyWorld_Type);
    Py_INCREF (&PyBody_Type);
    Py_INCREF (&PyJoint_Type);
    Py_INCREF (&PyDistanceJoint_Type);
	Py_INCREF (&PyRevoluteJoint_Type);
    Py_INCREF (&PyContact_Type);
    Py_INCREF (&PyShape_Type);
    Py_INCREF (&PyRectShape_Type);

    /* Init the module and add the object types. */
    mod = Py_InitModule3("physics", NULL,
        "A simple 2D physics module");
    PyModule_AddObject (mod, "World", (PyObject *) &PyWorld_Type);
    PyModule_AddObject (mod, "Body", (PyObject *) &PyBody_Type);
    PyModule_AddObject (mod, "Joint", (PyObject *) &PyJoint_Type);
    PyModule_AddObject (mod, "DistanceJoint",
        (PyObject *) &PyDistanceJoint_Type);
	/*PyModule_AddObject (mod, "RevoluteJoint",
		(PyObject *) &PyRevoluteJoint_Type);*/
    PyModule_AddObject (mod, "Shape", (PyObject *) &PyShape_Type);
    PyModule_AddObject (mod, "RectShape", (PyObject *) &PyRectShape_Type);

    /* C API */
    PyMath_ExportCAPI (c_api);
    PyBodyObject_ExportCAPI (c_api);
    PyJointObject_ExportCAPI (c_api);
    PyShapeObject_ExportCAPI (c_api);
    PyWorldObject_ExportCAPI (c_api);
    c_api_obj = PyCObject_FromVoidPtr ((void *) c_api, NULL);
    if (c_api_obj)
        PyModule_AddObject (mod, PHYSICS_CAPI_ENTRY, c_api_obj);
}
