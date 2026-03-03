/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners
  Copyright (C) 2008 Marcus von Appen

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

/*
 *  internal surface locking support for python objects
 */
#define PYGAMEAPI_SURFLOCK_INTERNAL

#include "pygame.h"

#include "pgcompat.h"

static int
pgSurface_Lock(pgSurfaceObject *);
static int
pgSurface_Unlock(pgSurfaceObject *);
static int
pgSurface_LockBy(pgSurfaceObject *, PyObject *);
static int
pgSurface_UnlockBy(pgSurfaceObject *, PyObject *);

static void
pgSurface_Prep(pgSurfaceObject *surfobj)
{
    struct pgSubSurface_Data *data = ((pgSurfaceObject *)surfobj)->subsurface;
    if (data != NULL) {
        pgSurface_LockBy((pgSurfaceObject *)data->owner, (PyObject *)surfobj);
    }
}

static void
pgSurface_Unprep(pgSurfaceObject *surfobj)
{
    struct pgSubSurface_Data *data = ((pgSurfaceObject *)surfobj)->subsurface;
    if (data != NULL) {
        pgSurface_UnlockBy((pgSurfaceObject *)data->owner,
                           (PyObject *)surfobj);
    }
}

static int
pgSurface_Lock(pgSurfaceObject *surfobj)
{
    return pgSurface_LockBy(surfobj, (PyObject *)surfobj);
}

static int
pgSurface_Unlock(pgSurfaceObject *surfobj)
{
    return pgSurface_UnlockBy(surfobj, (PyObject *)surfobj);
}

static int
pgSurface_LockBy(pgSurfaceObject *surfobj, PyObject *lockobj)
{
    PyObject *ref;
    pgSurfaceObject *surf = (pgSurfaceObject *)surfobj;

    if (surf->locklist == NULL) {
        surf->locklist = PyList_New(0);
        if (surf->locklist == NULL) {
            return 0;
        }
    }
    ref = PyWeakref_NewRef(lockobj, NULL);
    if (ref == NULL) {
        return 0;
    }
    if (ref == Py_None) {
        Py_DECREF(ref);
        return 0;
    }
    if (0 != PyList_Append(surf->locklist, ref)) {
        Py_DECREF(ref);
        return 0; /* Exception already set. */
    }
    Py_DECREF(ref);

    if (surf->subsurface != NULL) {
        pgSurface_Prep(surfobj);
    }
    if (!PG_LockSurface(surf->surf)) {
        PyErr_SetString(PyExc_RuntimeError, "error locking surface");
        return 0;
    }
    return 1;
}

static int
pgSurface_UnlockBy(pgSurfaceObject *surfobj, PyObject *lockobj)
{
    PG_DECLARE_EXCEPTION_SAVER

    pgSurfaceObject *surf = (pgSurfaceObject *)surfobj;
    int found = 0;
    int noerror = 1;
    int weakref_getref_result;

    if (surf->locklist != NULL) {
        PyObject *item, *ref;
        Py_ssize_t len = PyList_Size(surf->locklist);
        while (--len >= 0 && !found) {
            item = PyList_GetItem(surf->locklist, len);

            weakref_getref_result = PyWeakref_GetRef(item, &ref);
            if (weakref_getref_result == -1) {
                noerror = 0;
            }
            if (weakref_getref_result == 1) {
                if (ref == lockobj) {
                    // Need to cache any currently set exceptions before
                    // calling PySequence_DelItem
                    PG_SAVE_EXCEPTION

                    if (PySequence_DelItem(surf->locklist, len) == -1) {
                        Py_DECREF(ref);
                        // Restore the previously set exception before
                        // returning
                        PG_UNSAVE_EXCEPTION
                        return 0;
                    }
                    else {
                        found = 1;
                    }
                    // Restore the previously set exception
                    PG_UNSAVE_EXCEPTION
                }
                Py_DECREF(ref);
            }
        }

        /* Clear dead references */
        len = PyList_Size(surf->locklist);
        while (--len >= 0) {
            item = PyList_GetItem(surf->locklist, len);

            weakref_getref_result = PyWeakref_GetRef(item, &ref);
            if (weakref_getref_result == -1) {
                noerror = 0;
            }
            else if (weakref_getref_result == 0) {
                // Need to cache any currently set exceptions before calling
                // PySequence_DelItem
                PG_SAVE_EXCEPTION
                if (PySequence_DelItem(surf->locklist, len) == -1) {
                    noerror = 0;
                }
                else {
                    found++;
                }
                // Restore the previously set exception
                PG_UNSAVE_EXCEPTION
            }
            else if (weakref_getref_result == 1) {
                Py_DECREF(ref);
            }
        }
    }

    if (!found) {
        return noerror;
    }

    /* Release all found locks. */
    while (found > 0) {
        if (surf->surf != NULL) {
            SDL_UnlockSurface(surf->surf);
        }
        if (surf->subsurface != NULL) {
            pgSurface_Unprep(surfobj);
        }
        found--;
    }

    return noerror;
}

static PyMethodDef _surflock_methods[] = {{NULL, NULL, 0, NULL}};

/*DOC*/ static char _surflock_doc[] =
    /*DOC*/ "Surface locking support";

MODINIT_DEFINE(surflock)
{
    PyObject *module, *apiobj;
    static void *c_api[PYGAMEAPI_SURFLOCK_NUMSLOTS];

    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "surflock",
                                         _surflock_doc,
                                         -1,
                                         _surflock_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};

    /* Create the module and add the functions */
    module = PyModule_Create(&_module);
    if (module == NULL) {
        return NULL;
    }

    /* export the c api */
    c_api[0] = pgSurface_Prep;
    c_api[1] = pgSurface_Unprep;
    c_api[2] = pgSurface_Lock;
    c_api[3] = pgSurface_Unlock;
    c_api[4] = pgSurface_LockBy;
    c_api[5] = pgSurface_UnlockBy;
    apiobj = encapsulate_api(c_api, "surflock");
    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj)) {
        Py_XDECREF(apiobj);
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
