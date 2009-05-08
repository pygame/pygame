#include <Python.h>
#include "bitmask.h"

#define PYGAMEAPI_MASK_NUMSLOTS 1
#define PYGAMEAPI_LOCAL_ENTRY "_PYGAME_C_API"

typedef struct {
  PyObject_HEAD
  bitmask_t *mask;
} PyMaskObject;

#define PyMask_AsBitmap(x) (((PyMaskObject*)x)->mask)

#ifndef PYGAMEAPI_MASK_INTERNAL

#define PyMask_Type     (*(PyTypeObject*)PyMASK_C_API[0])
#define PyMask_Check(x) ((x)->ob_type == &PyMask_Type)

#define import_pygame_mask() {                                                                 \
	PyObject *module = PyImport_ImportModule(IMPPREFIX "mask");                               \
	if (module != NULL) {                                                                  \
		PyObject *dict  = PyModule_GetDict(module);                                    \
		PyObject *c_api = PyDict_GetItemString(dict, PYGAMEAPI_LOCAL_ENTRY);           \
		if(PyCObject_Check(c_api)) {                                                   \
			void** localptr = (void**) PyCObject_AsVoidPtr(c_api);                 \
			memcpy(PyMASK_C_API, localptr, sizeof(void*)*PYGAMEAPI_MASK_NUMSLOTS); \
		} Py_DECREF(module);                                                           \
	}                                                                                      \
}

#endif /* !defined(PYGAMEAPI_MASK_INTERNAL) */

static void* PyMASK_C_API[PYGAMEAPI_MASK_NUMSLOTS] = {NULL};

