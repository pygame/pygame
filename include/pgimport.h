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

#if PG_HAVE_CAPSULE
#define encapsulate_api(ptr, module) \
    PyCapsule_New(ptr, PG_CAPSULE_NAME(module), NULL)
#else /* ~PG_HAVE_CAPSULE */
#define encapsulate_api(ptr, module) PyCObject_FromVoidPtr(ptr, NULL)
#endif /* ~PG_HAVE_CAPSULE */

#define PYGAMEAPI_LOCAL_ENTRY "_PYGAME_C_API"
#define PG_CAPSULE_NAME(m) (IMPPREFIX m "." PYGAMEAPI_LOCAL_ENTRY)

/*
 * fill API slots defined by PYGAMEAPI_DEFINE_SLOTS/PYGAMEAPI_EXTERN_SLOTS
 */
#define _IMPORT_PYGAME_MODULE(module, MODULE, api_root)                      \
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
                                                                             \
                if (localptr != NULL) {                                      \
                    memcpy(api_root + PYGAMEAPI_##MODULE##_FIRSTSLOT,        \
                           localptr,                                         \
                           sizeof(void **) * PYGAMEAPI_##MODULE##_NUMSLOTS); \
                }                                                            \
            }                                                                \
            Py_XDECREF(_c_api);                                              \
        }                                                                    \
    }

/*
 * source file must include one of these in order to use _IMPORT_PYGAME_MODULE
 * this array is filled by import_pygame_*() functions
 * disable with NO_PYGAME_C_API
 */
#define PYGAMEAPI_DEFINE_SLOTS(api_root, numslots) \
    void *api_root[(numslots)] = {NULL}
#define PYGAMEAPI_EXTERN_SLOTS(api_root, numslots) \
    extern void *api_root[(numslots)]

#define PYGAMEAPI_GET_SLOT(api_root, index) \
    api_root[(index)]

/*
 * disabled API with NO_PYGAME_C_API; do nothing instead
 */
#ifdef NO_PYGAME_C_API

#undef PYGAMEAPI_DEFINE_SLOTS
#undef PYGAMEAPI_EXTERN_SLOTS

#define PYGAMEAPI_DEFINE_SLOTS(api_root, numslots)
#define PYGAMEAPI_EXTERN_SLOTS(api_root, numslots)

/* intentionally leave this defined to cause a compiler error *
#define PYGAMEAPI_GET_SLOT(api_root, index)
#undef PYGAMEAPI_GET_SLOT*/

#undef _IMPORT_PYGAME_MODULE
#define _IMPORT_PYGAME_MODULE(module, MODULE, api_root)

#endif /* NO_PYGAME_C_API */

#endif /* ~PGIMPORT_H */
