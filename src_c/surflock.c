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
pgSurface_Lock(PyObject *);
static int
pgSurface_Unlock(PyObject *);
static int
pgSurface_LockBy(PyObject *, PyObject *);
static int
pgSurface_UnlockBy(PyObject *, PyObject *);

static void
_lifelock_dealloc(PyObject *);

static void
pgSurface_Prep(PyObject *surfobj)
{
    struct pgSubSurface_Data *data = ((pgSurfaceObject *)surfobj)->subsurface;
    if (data != NULL) {
        SDL_Surface *surf = pgSurface_AsSurface(surfobj);
        SDL_Surface *owner = pgSurface_AsSurface(data->owner);
        pgSurface_LockBy(data->owner, surfobj);
        surf->pixels = ((char *)owner->pixels) + data->pixeloffset;
    }
}

static void
pgSurface_Unprep(PyObject *surfobj)
{
    struct pgSubSurface_Data *data = ((pgSurfaceObject *)surfobj)->subsurface;
    if (data != NULL) {
        pgSurface_UnlockBy(data->owner, surfobj);
    }
}

static int
pgSurface_Lock(PyObject *surfobj)
{
    return pgSurface_LockBy(surfobj, surfobj);
}

static int
pgSurface_Unlock(PyObject *surfobj)
{
    return pgSurface_UnlockBy(surfobj, surfobj);
}

static int
pgSurface_LockBy(PyObject *surfobj, PyObject *lockobj)
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
    PyList_Append(surf->locklist, ref);
    Py_DECREF(ref);

    if (surf->subsurface != NULL) {
        pgSurface_Prep(surfobj);
    }
    if (SDL_LockSurface(surf->surf) == -1) {
        PyErr_SetString(PyExc_RuntimeError, "error locking surface");
        return 0;
    }
    return 1;
}

static int
pgSurface_UnlockBy(PyObject *surfobj, PyObject *lockobj)
{
    pgSurfaceObject *surf = (pgSurfaceObject *)surfobj;
    int found = 0;
    int noerror = 1;

    if (surf->locklist != NULL) {
        PyObject *item, *ref;
        Py_ssize_t len = PyList_Size(surf->locklist);
        while (--len >= 0 && !found) {
            item = PyList_GetItem(surf->locklist, len);
            ref = PyWeakref_GetObject(item);
            if (ref == lockobj) {
                if (PySequence_DelItem(surf->locklist, len) == -1) {
                    return 0;
                }
                else {
                    found = 1;
                }
            }
        }

        /* Clear dead references */
        len = PyList_Size(surf->locklist);
        while (--len >= 0) {
            item = PyList_GetItem(surf->locklist, len);
            ref = PyWeakref_GetObject(item);
            if (ref == Py_None) {
                if (PySequence_DelItem(surf->locklist, len) == -1) {
                    noerror = 0;
                }
                else {
                    found++;
                }
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

static PyTypeObject pgLifetimeLock_Type = {
    TYPE_HEAD(NULL, 0) "SurfLifeLock", /* name */
    sizeof(pgLifetimeLockObject),      /* basic size */
    0,                                 /* tp_itemsize */
    _lifelock_dealloc,                 /* tp_dealloc*/
    NULL,                              /* tp_print */
    NULL,                              /* tp_getattr */
    NULL,                              /* tp_setattr */
    NULL,                              /* tp_compare */
    NULL,                              /* tp_repr */
    NULL,                              /* tp_as_number */
    NULL,                              /* tp_as_sequence */
    NULL,                              /* tp_as_mapping */
    NULL,                              /* tp_hash */
    NULL,                              /* tp_call */
    NULL,                              /* tp_str */
    NULL,                              /* tp_getattro */
    NULL,                              /* tp_setattro */
    NULL,                              /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_WEAKREFS,
    NULL,                                     /* tp_doc */
    NULL,                                     /* tp_traverse */
    NULL,                                     /* tp_clear */
    NULL,                                     /* tp_richcompare */
    offsetof(pgLifetimeLockObject, weakrefs), /* tp_weaklistoffset */
    NULL,                                     /* tp_iter */
    NULL,                                     /* tp_iternext */
    NULL,                                     /* tp_methods */
    NULL,                                     /* tp_members */
    NULL,                                     /* tp_getset */
    NULL,                                     /* tp_base */
    NULL,                                     /* tp_dict */
    NULL,                                     /* tp_descr_get */
    NULL,                                     /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    NULL,                                     /* tp_init */
    NULL,                                     /* tp_alloc */
    NULL,                                     /* tp_new */
    NULL,                                     /* tp_free */
    NULL,                                     /* tp_is_gc */
    NULL,                                     /* tp_bases */
    NULL,                                     /* tp_mro */
    NULL,                                     /* tp_cache */
    NULL,                                     /* tp_subclasses */
    NULL,                                     /* tp_weaklist */
    NULL                                      /* tp_del */
};

/* lifetimelock object internals */
static void
_lifelock_dealloc(PyObject *self)
{
    pgLifetimeLockObject *lifelock = (pgLifetimeLockObject *)self;

    if (lifelock->weakrefs != NULL) {
        PyObject_ClearWeakRefs(self);
    }

    pgSurface_UnlockBy(lifelock->surface, lifelock->lockobj);
    Py_DECREF(lifelock->surface);
    PyObject_DEL(self);
}

static PyObject *
pgSurface_LockLifetime(PyObject *surfobj, PyObject *lockobj)
{
    pgLifetimeLockObject *life;
    if (surfobj == NULL) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    life = PyObject_NEW(pgLifetimeLockObject, &pgLifetimeLock_Type);
    if (life != NULL) {
        life->surface = surfobj;
        life->lockobj = lockobj;
        life->weakrefs = NULL;
        Py_INCREF(surfobj);
        if (!pgSurface_LockBy(surfobj, lockobj)) {
            return NULL;
        }
    }
    return (PyObject *)life;
}

static PyMethodDef _surflock_methods[] = {{NULL, NULL, 0, NULL}};

/*DOC*/ static char _surflock_doc[] =
    /*DOC*/ "Surface locking support";

MODINIT_DEFINE(surflock)
{
    PyObject *module, *dict, *apiobj;
    int ecode;
    static void *c_api[PYGAMEAPI_SURFLOCK_NUMSLOTS];

#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "surflock",
                                         _surflock_doc,
                                         -1,
                                         _surflock_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};
#endif

    if (PyType_Ready(&pgLifetimeLock_Type) < 0) {
        MODINIT_ERROR;
    }

    /* Create the module and add the functions */
#if PY3
    module = PyModule_Create(&_module);
#else
    module =
        Py_InitModule3(MODPREFIX "surflock", _surflock_methods, _surflock_doc);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    dict = PyModule_GetDict(module);

    /* export the c api */
    c_api[0] = &pgLifetimeLock_Type;
    c_api[1] = pgSurface_Prep;
    c_api[2] = pgSurface_Unprep;
    c_api[3] = pgSurface_Lock;
    c_api[4] = pgSurface_Unlock;
    c_api[5] = pgSurface_LockBy;
    c_api[6] = pgSurface_UnlockBy;
    c_api[7] = pgSurface_LockLifetime;
    apiobj = encapsulate_api(c_api, "surflock");
    if (apiobj == NULL) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF(apiobj);
    if (ecode) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN(module);
}
