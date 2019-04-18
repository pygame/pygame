#ifndef PGIMPORT_INTERNAL_H
#define PGIMPORT_INTERNAL_H

#include "include/pgimport.h"

#if PG_HAVE_CAPSULE
#define encapsulate_api(array, module) \
    PyCapsule_New(array, PG_CAPSULE_NAME(module), NULL)
#else /* ~PG_HAVE_CAPSULE */
#define encapsulate_api(array, module) \
    PyCObject_FromVoidPtr(array, NULL)
#endif /* ~PG_HAVE_CAPSULE */

#define _encapsulate_api_safe(retptr, array,                       \
                              module, MODULE,                      \
                              api_root)                            \
    do {                                                           \
        PG_STATIC_ASSERT( sizeof(array) / sizeof(void*) ==         \
                          PYGAMEAPI_##MODULE##_NUMSLOTS,           \
                          encapsulate_wrong_size );                \
        *retptr = encapsulate_api(array, #module);                 \
    } while(0)

#define encapsulate_api_safe(retptr, array, module, MODULE) \
    _encapsulate_api_safe(retptr, array, module, MODULE, PyGAME_C_API)

#endif /* ~PGIMPORT_INTERNAL_H */
