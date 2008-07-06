#include "pgBodyObject.h"
#include "pgWorldObject.h"

extern PyTypeObject pgWorldType;
extern PyTypeObject pgBodyType;
//Joint types
extern PyTypeObject pgJointType;
extern PyTypeObject pgDistanceJointType;

//help functions
extern PyMethodDef pgHelpMethods[];


#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initphysics(void) 
{
	PyObject* mod;

	/* Check the types. */
	if (PyType_Ready(&pgWorldType) < 0)
		return;
	if (PyType_Ready(&pgBodyType) < 0)
		return;
	if (PyType_Ready(&pgJointType) < 0)
		return;
        pgDistanceJointType.tp_base = &pgJointType;
	if (PyType_Ready(&pgDistanceJointType) < 0)
		return;

		/* Increase their ref counts. */
	Py_INCREF (&pgWorldType);
	Py_INCREF (&pgBodyType);
	Py_INCREF (&pgJointType);
	Py_INCREF (&pgDistanceJointType);

	/* Init the module and add the object types. */
	mod = Py_InitModule3("physics", pgHelpMethods, "Simple 2D physics module");
	PyModule_AddObject (mod, "World", (PyObject *) &pgWorldType);
	PyModule_AddObject (mod, "Body", (PyObject *) &pgBodyType);
	PyModule_AddObject (mod, "Joint", (PyObject *) &pgJointType);
	PyModule_AddObject (mod, "DistanceJoint", (PyObject *) &pgDistanceJointType);
	
}

