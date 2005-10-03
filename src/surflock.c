/*
    pygame - Python Game Library
    Copyright (C) 2000-2001  Pete Shinners

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



static int PySurface_Lock(PyObject* surfobj);
static int PySurface_Unlock(PyObject* surfobj);



static void PySurface_Prep(PyObject* surfobj)
{
	struct SubSurface_Data* data = ((PySurfaceObject*)surfobj)->subsurface;
	if(data)
	{
		SDL_Surface* surf = PySurface_AsSurface(surfobj);
		SDL_Surface* owner = PySurface_AsSurface(data->owner);
		PySurface_Lock(data->owner);
		surf->pixels = ((char*)owner->pixels) + data->pixeloffset;
	}
}

static void PySurface_Unprep(PyObject* surfobj)
{
	struct SubSurface_Data* data = ((PySurfaceObject*)surfobj)->subsurface;
	if(data)
		PySurface_Unlock(data->owner);
}

static int PySurface_Lock(PyObject* surfobj)
{
	PySurfaceObject* surf = (PySurfaceObject*)surfobj;
	if(surf->subsurface)
		PySurface_Prep(surfobj);
	if(SDL_LockSurface(surf->surf) == -1)
	{
		PyErr_SetString(PyExc_RuntimeError, "error locking surface");
		return 0;
	}
	return 1;
}


static int PySurface_Unlock(PyObject* surfobj)
{
	PySurfaceObject* surf = (PySurfaceObject*)surfobj;
	SDL_UnlockSurface(surf->surf);
	if(surf->subsurface)
		PySurface_Unprep(surfobj);
	return 1;
}





/* lifetimelock object internals */

typedef struct {
	PyObject_HEAD
	PyObject* surface;
} PyLifetimeLockObject;

static void lifelock_dealloc(PyObject* self)
{
	PyLifetimeLockObject* lifelock = (PyLifetimeLockObject*)self;

	PySurface_Unlock(lifelock->surface);
	Py_DECREF(lifelock->surface);

	PyObject_DEL(self);
}

static PyTypeObject PyLifetimeLock_Type =
{
	PyObject_HEAD_INIT(NULL)
	0,                           /*size*/
	"SurfLifeLock",              /*name*/
	sizeof(PyLifetimeLockObject),/*basic size*/
	0,                           /*itemsize*/
	lifelock_dealloc,            /*dealloc*/
};


static PyObject* PySurface_LockLifetime(PyObject* surf)
{
	PyLifetimeLockObject* life;
	if(!surf) return RAISE(PyExc_SDLError, SDL_GetError());

	if(!PySurface_Lock(surf))
		return NULL;

	life = PyObject_NEW(PyLifetimeLockObject, &PyLifetimeLock_Type);
	if(life)
	{
		life->surface = surf;
		Py_INCREF(surf);
	}
	return (PyObject*)life;
}





static PyMethodDef surflock__builtins__[] =
{
	{NULL, NULL}
};


PYGAME_EXPORT
void initsurflock(void)
{
	PyObject *module, *dict, *apiobj;
	static void* c_api[PYGAMEAPI_SURFLOCK_NUMSLOTS];

	PyType_Init(PyLifetimeLock_Type);


	/* Create the module and add the functions */
	module = Py_InitModule3("surflock", surflock__builtins__, "Surface locking support");
	dict = PyModule_GetDict(module);

	/* export the c api */
	c_api[0] = PySurface_Prep;
	c_api[1] = PySurface_Unprep;
	c_api[2] = PySurface_Lock;
	c_api[3] = PySurface_Unlock;
	c_api[4] = PySurface_LockLifetime;
	apiobj = PyCObject_FromVoidPtr(c_api, NULL);
	PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
	Py_DECREF(apiobj);
}
