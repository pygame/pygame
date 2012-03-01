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

static int PySurface_Lock (PyObject* surfobj);
static int PySurface_Unlock (PyObject* surfobj);
static int PySurface_LockBy (PyObject* surfobj, PyObject* lockobj);
static int PySurface_UnlockBy (PyObject* surfobj, PyObject* lockobj);

static void _lifelock_dealloc (PyObject* self);

static void
PySurface_Prep (PyObject* surfobj)
{
    struct SubSurface_Data* data = ((PySurfaceObject*) surfobj)->subsurface;
    if (data)
    {
        SDL_Surface* surf = PySurface_AsSurface (surfobj);
        SDL_Surface* owner = PySurface_AsSurface (data->owner);
        PySurface_LockBy (data->owner, surfobj);
        surf->pixels = ((char*) owner->pixels) + data->pixeloffset;
    }
}

static void
PySurface_Unprep (PyObject* surfobj)
{
    struct SubSurface_Data* data = ((PySurfaceObject*) surfobj)->subsurface;
    if (data)
        PySurface_UnlockBy (data->owner, surfobj);
}

static int
PySurface_Lock (PyObject* surfobj)
{
    return PySurface_LockBy (surfobj, surfobj);
}

static int
PySurface_Unlock (PyObject* surfobj)
{
    return PySurface_UnlockBy (surfobj, surfobj);
}

static int
PySurface_LockBy (PyObject* surfobj, PyObject* lockobj)
{
    PyObject *ref;
    PySurfaceObject* surf = (PySurfaceObject*) surfobj;

    if (!surf->locklist)
    {
        surf->locklist = PyList_New (0);
        if (!surf->locklist)
            return 0;
    }
    ref = PyWeakref_NewRef (lockobj, NULL);
    if (!ref)
        return 0;
    if (ref == Py_None)
    {
        Py_DECREF (ref);
        return 0;
    }
    PyList_Append (surf->locklist, ref);
    Py_DECREF (ref);

    if (surf->subsurface)
        PySurface_Prep (surfobj);
    if (SDL_LockSurface (surf->surf) == -1)
    {
        PyErr_SetString (PyExc_RuntimeError, "error locking surface");
        return 0;
    }
    return 1;
}

static int
PySurface_UnlockBy (PyObject* surfobj, PyObject* lockobj)
{
    PySurfaceObject* surf = (PySurfaceObject*) surfobj;
    int found = 0;
    int noerror = 1;

    if (surf->locklist)
    {
        PyObject *item, *ref;
        Py_ssize_t len = PyList_Size (surf->locklist);
        while (--len >= 0 && !found)
        {
            item = PyList_GetItem (surf->locklist, len);
            ref = PyWeakref_GetObject (item);
            if (ref == lockobj)
            {
                if (PySequence_DelItem (surf->locklist, len) == -1)
                    return 0;
                else
                    found = 1;
            }
        }

        /* Clear dead references */
        len = PyList_Size (surf->locklist);
        while (--len >= 0)
        {
            item = PyList_GetItem (surf->locklist, len);
            ref = PyWeakref_GetObject (item);
            if (ref == Py_None)
            {
                if (PySequence_DelItem (surf->locklist, len) == -1)
                    noerror = 0;
                else
                    found++;
            }
        }
    }

    if (!found)
        return noerror;

    /* Release all found locks. */
    while (found > 0)
    {
        if (surf->surf != NULL)
            SDL_UnlockSurface (surf->surf);
        if (surf->subsurface)
            PySurface_Unprep (surfobj);
        found--;
    }

    return noerror;
}


static PyTypeObject PyLifetimeLock_Type =
{
    TYPE_HEAD (NULL, 0)
    "SurfLifeLock",             /*name*/
    sizeof(PyLifetimeLock),     /*basic size*/
    0,                          /* tp_itemsize */
    _lifelock_dealloc,          /* tp_dealloc*/
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    0,                          /* tp_repr */
    0,                          /* tp_as_number */
    0,                          /* tp_as_sequence */
    0,                          /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    0,                          /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_WEAKREFS,
    0,                          /* tp_doc */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    offsetof (PyLifetimeLock, weakrefs),  /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    0,                          /* tp_methods */
    0,                          /* tp_members */
    0,                          /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    0,                          /* tp_init */
    0,                          /* tp_alloc */
    0,                          /* tp_new */
    0,                          /* tp_free */
    0,                          /* tp_is_gc */
    0,                          /* tp_bases */
    0,                          /* tp_mro */
    0,                          /* tp_cache */
    0,                          /* tp_subclasses */
    0,                          /* tp_weaklist */
    0                           /* tp_del */
};

/* lifetimelock object internals */
static void
_lifelock_dealloc (PyObject* self)
{
    PyLifetimeLock* lifelock = (PyLifetimeLock*) self;

    if (lifelock->weakrefs)
        PyObject_ClearWeakRefs (self);

    PySurface_UnlockBy (lifelock->surface, lifelock->lockobj);
    Py_DECREF (lifelock->surface);
    PyObject_DEL (self);
}

static PyObject*
PySurface_LockLifetime (PyObject* surfobj, PyObject *lockobj)
{
    PyLifetimeLock* life;
    if (!surfobj)
        return RAISE (PyExc_SDLError, SDL_GetError ());
    
    life = PyObject_NEW (PyLifetimeLock, &PyLifetimeLock_Type);
    if (life)
    {
        life->surface = surfobj;
        life->lockobj = lockobj;
        life->weakrefs = NULL;
        Py_INCREF (surfobj);
        if (!PySurface_LockBy (surfobj, lockobj))
            return NULL;
    }
    return (PyObject*) life;
}

static PyMethodDef _surflock_methods[] =
{
    { NULL, NULL, 0, NULL }
};

/*DOC*/ static char _surflock_doc[] =
/*DOC*/     "Surface locking support";

MODINIT_DEFINE (surflock)
{
    PyObject *module, *dict, *apiobj;
    int ecode;
    static void* c_api[PYGAMEAPI_SURFLOCK_NUMSLOTS];
    
#if PY3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "surflock",
        _surflock_doc,
        -1,
        _surflock_methods,
        NULL, NULL, NULL, NULL
    };
#endif

    if (PyType_Ready (&PyLifetimeLock_Type) < 0) {
        MODINIT_ERROR;
    }

    /* Create the module and add the functions */
#if PY3
    module = PyModule_Create (&_module);
#else
    module = Py_InitModule3 (MODPREFIX "surflock", 
                             _surflock_methods, 
                             _surflock_doc);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    dict = PyModule_GetDict (module);
    
    /* export the c api */
    c_api[0] = &PyLifetimeLock_Type;
    c_api[1] = PySurface_Prep;
    c_api[2] = PySurface_Unprep;
    c_api[3] = PySurface_Lock;
    c_api[4] = PySurface_Unlock;
    c_api[5] = PySurface_LockBy;
    c_api[6] = PySurface_UnlockBy;
    c_api[7] = PySurface_LockLifetime;
    apiobj = encapsulate_api (c_api, "surflock");
    if (apiobj == NULL) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString (dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF (apiobj);
    if (ecode) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN (module);
}
