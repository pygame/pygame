/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners
  Copyright (C) 2007  Rene Dudfield, Richard Goedeken 

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

  Pete Shinners
  pete@shinners.org
*/

/* Bufferproxy module C api.
   Depends on pygame.h being included first.
 */
#if !defined(PG_VIEW_HEADER)

#include "pgarrinter.h"

typedef int (*PgBufferProxy_PreludeCallback)(PyObject *);
typedef void (*PgBufferProxy_PostscriptCallback)(PyObject *);

/* Bufferproxy flags */
#define BUFFERPROXY_CONTIGUOUS    1
#define BUFFERPROXY_C_ORDER       2
#define BUFFERPROXY_F_ORDER       4 

#define PYGAMEAPI_VIEW_NUMSLOTS 5
#define PYGAMEAPI_VIEW_FIRSTSLOT 0

#if !(defined(PYGAMEAPI_VIEW_INTERNAL) || defined(NO_PYGAME_C_API))
static void *PgBUFFERPROXY_C_API[PYGAMEAPI_VIEW_NUMSLOTS];

typedef PyObject *(*_pgbufferproxy_new_t)(Py_buffer *,
                                   int,
                                   PgBufferProxy_PreludeCallback,
                                   PgBufferProxy_PostscriptCallback);
typedef PyObject *(*_pgbufferproxy_get_obj_t)(PyObject *);
typedef int *(*_pg_getarrayinterface_t)(PyObject *,
                                        PyObject **,
                                        PyArrayInterface **);
typedef PyObject *(*_pg_arraystructasdict_t)(PyArrayInterface *inter_p);

#define PgBufferProxy_Type (*(PyTypeObject*)PgBUFFERPROXY_C_API[0])
#define PgBufferProxy_New (*(_pgbufferproxy_new_t)PgBUFFERPROXY_C_API[1])
#define PgBufferProxy_GetParent \
    (*(_pgbufferproxy_get_obj_t)PgBUFFERPROXY_C_API[2])
#define Pg_GetArrayInterface (*(_pg_getarrayinterface_t)PgBUFFERPROXY_C_API[3])
#define Pg_ArrayStructAsDict (*(_pg_arraystructasdict_t)PgBUFFERPROXY_C_API[4])
#define PgBufferProxy_Check(x) ((x)->ob_type == (PgBufferProxy_Type))
#define import_pygame_view() \
    _IMPORT_PYGAME_MODULE(_view, VIEW, PgBUFFERPROXY_C_API)

#endif /* #if !(defined(PYGAMEAPI_VIEW_INTERNAL) || ... */

#define PG_VIEW_HEADER

#endif /* #if !defined(PG_VIEW_HEADER) */
