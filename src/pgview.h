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

typedef int (*PgBufproxy_CallbackBefore)(PyObject *);
typedef void (*PgBufproxy_CallbackAfter)(PyObject *);

/* Bufferproxy flags */
#define BUFPROXY_CONTIGUOUS    1
#define BUFPROXY_C_ORDER       2
#define BUFPROXY_F_ORDER       4

#define PYGAMEAPI_VIEW_NUMSLOTS 6
#define PYGAMEAPI_VIEW_FIRSTSLOT 0

#if !(defined(PYGAMEAPI_VIEW_INTERNAL) || defined(NO_PYGAME_C_API))
static void *PgBUFPROXY_C_API[PYGAMEAPI_VIEW_NUMSLOTS];

typedef PyObject *(*_pgbufproxy_new_t)(Py_buffer *,
                                   int,
                                   PgBufproxy_CallbackBefore,
                                   PgBufproxy_CallbackAfter);
typedef PyObject *(*_pgbufproxy_get_obj_t)(PyObject *);
typedef int (*_pgbufproxy_trip_t)(PyObject *);
typedef int (*_pg_getarrayinterface_t)(PyObject *,
                                       PyObject **,
                                       PyArrayInterface **);
typedef PyObject *(*_pg_arraystructasdict_t)(PyArrayInterface *inter_p);

#define PgBufproxy_Type (*(PyTypeObject*)PgBUFPROXY_C_API[0])
#define PgBufproxy_New (*(_pgbufproxy_new_t)PgBUFPROXY_C_API[1])
#define PgBufproxy_GetParent \
    (*(_pgbufproxy_get_obj_t)PgBUFPROXY_C_API[2])
#define Pg_GetArrayInterface (*(_pg_getarrayinterface_t)PgBUFPROXY_C_API[3])
#define Pg_ArrayStructAsDict (*(_pg_arraystructasdict_t)PgBUFPROXY_C_API[4])
#define PgBufproxy_Trip (*(_pgbufproxy_trip_t)PgBUFPROXY_C_API[5])
#define PgBufproxy_Check(x) ((x)->ob_type == (PgBufproxy_Type))
#define import_pygame_view() \
    _IMPORT_PYGAME_MODULE(_view, VIEW, PgBUFPROXY_C_API)

#endif /* #if !(defined(PYGAMEAPI_VIEW_INTERNAL) || ... */

#define PG_VIEW_HEADER

#endif /* #if !defined(PG_VIEW_HEADER) */
