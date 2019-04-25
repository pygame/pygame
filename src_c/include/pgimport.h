#ifndef PGIMPORT_H
#define PGIMPORT_H

/* Prefix when initializing module */
#define MODPREFIX ""
/* Prefix when importing module */
#define IMPPREFIX "pygame."

#ifdef __SYMBIAN32__

/* On Symbian there is no pygame package. The extensions are built-in or in
 * sys\bin. */
#undef MODPREFIX
#undef IMPPREFIX
#define MODPREFIX "pygame_"
#define IMPPREFIX "pygame_"

#endif /* __SYMBIAN32__ */

#include "pgcompat.h"

#define PYGAMEAPI_LOCAL_ENTRY "_PYGAME_C_API"
#define PG_CAPSULE_NAME(m) (IMPPREFIX m "." PYGAMEAPI_LOCAL_ENTRY)

/*
 * fill API slots defined by PYGAMEAPI_DEFINE_SLOTS/PYGAMEAPI_EXTERN_SLOTS
 */
#define _IMPORT_PYGAME_MODULE(module)                                        \
    {                                                                        \
        PyObject *_module = PyImport_ImportModule(IMPPREFIX #module);        \
                                                                             \
        if (_module != NULL) {                                               \
            PyObject *_c_api =                                               \
                PyObject_GetAttrString(_module, PYGAMEAPI_LOCAL_ENTRY);      \
                                                                             \
            Py_DECREF(_module);                                              \
            if (_c_api != NULL && PyCapsule_CheckExact(_c_api)) {            \
                void **localptr = (void **)PyCapsule_GetPointer(             \
                    _c_api, PG_CAPSULE_NAME(#module));                       \
                _PGSLOTS_ ## module = localptr;                              \
            }                                                                \
            Py_XDECREF(_c_api);                                              \
        }                                                                    \
    }

#define PYGAMEAPI_IS_IMPORTED(module) (_PGSLOTS_ ## module != NULL)

/*
 * source file must include one of these in order to use _IMPORT_PYGAME_MODULE.
 * this is set by import_pygame_*() functions.
 * disable with NO_PYGAME_C_API
 */
#define PYGAMEAPI_DEFINE_SLOTS(module) \
    void ** _PGSLOTS_ ## module = NULL
#define PYGAMEAPI_EXTERN_SLOTS(module) \
    extern void **_PGSLOTS_ ## module
#define PYGAMEAPI_GET_SLOT(module, index) \
    _PGSLOTS_ ## module [(index)]

/*
 * disabled API with NO_PYGAME_C_API; do nothing instead
 */
#ifdef NO_PYGAME_C_API

#undef PYGAMEAPI_DEFINE_SLOTS
#undef PYGAMEAPI_EXTERN_SLOTS

#define PYGAMEAPI_DEFINE_SLOTS(module)
#define PYGAMEAPI_EXTERN_SLOTS(module)

/* intentionally leave this defined to cause a compiler error *
#define PYGAMEAPI_GET_SLOT(api_root, index)
#undef PYGAMEAPI_GET_SLOT*/

#undef _IMPORT_PYGAME_MODULE
#define _IMPORT_PYGAME_MODULE(module)

#endif /* NO_PYGAME_C_API */

#endif /* ~PGIMPORT_H */
