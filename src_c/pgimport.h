#ifndef PGIMPORT_INTERNAL_H
#define PGIMPORT_INTERNAL_H

#include "include/pgimport.h"

#if PG_HAVE_CAPSULE
#define encapsulate_api(ptr, module) \
    PyCapsule_New(ptr, PG_CAPSULE_NAME(module), NULL)
#else /* ~PG_HAVE_CAPSULE */
#define encapsulate_api(ptr, module) PyCObject_FromVoidPtr(ptr, NULL)
#endif /* ~PG_HAVE_CAPSULE */

#endif /* PGIMPORT_INTERNAL_H */
