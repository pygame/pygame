#include "pgBodyObject.h"
#include "pgWorldObject.h"

extern PyTypeObject pgWorldType;
extern PyTypeObject pgBodyType;


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

		/* Increase their ref counts. */
	Py_INCREF (&pgWorldType);
	Py_INCREF (&pgBodyType);

	/* Init the module and add the object types. */
	mod = Py_InitModule3("physics", NULL, "Simple 2D physics module");
	PyModule_AddObject (mod, "world", (PyObject *) &pgWorldType);
	PyModule_AddObject (mod, "body", (PyObject *) &pgBodyType);

}