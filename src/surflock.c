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



/*extra Surface documentation*/
#if 0 /*extra help, only for docs, not docstrings*/
    /*DOC*/ static char doc_Surface_EXTRA[] =
    /*DOC*/    "Any functions that directly access a surface's pixel data will\n"
    /*DOC*/    "need that surface to be lock()'ed. These functions can lock()\n"
    /*DOC*/    "and unlock() the surfaces themselves without assistance. But, if\n"
    /*DOC*/    "a function will be called many times, there will be a lot of overhead\n"
    /*DOC*/    "for multiple locking and unlocking of the surface. It is best to lock\n"
    /*DOC*/    "the surface manually before making the function call many times, and\n"
    /*DOC*/    "then unlocking when you are finished. All functions that need a locked\n"
    /*DOC*/    "surface will say so in their docs.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Also remember that you will want to leave the surface locked for the\n"
    /*DOC*/    "shortest amount of time needed.\n"
    /*DOC*/    "\n"
    /*DOC*/    "\n"
    /*DOC*/    "Here is the quick breakdown of how packed pixels work (don't worry if\n"
    /*DOC*/    "you don't quite understand this, it is only here for informational\n"
    /*DOC*/    "purposes, it is not needed). Each colorplane mask can be used to\n"
    /*DOC*/    "isolate the values for a colorplane from the packed pixel color.\n"
    /*DOC*/    "Therefore PACKED_COLOR & RED_MASK == REDPLANE. Note that the\n"
    /*DOC*/    "REDPLANE is not exactly the red color value, but it is the red\n"
    /*DOC*/    "color value bitwise left shifted a certain amount. The losses and\n"
    /*DOC*/    "masks can be used to convert back and forth between each\n"
    /*DOC*/    "colorplane and the actual color for that plane. Here are the\n"
    /*DOC*/    "final formulas used be map and unmap.\n"
    /*DOC*/    "PACKED_COLOR = RED>>losses[0]<<shifts[0] |\n"
    /*DOC*/    "      GREEN>>losses[1]<<shifts[1] | BLUE>>losses[2]<<shifts[2]\n"
    /*DOC*/    "RED = PACKED_COLOR & masks[0] >> shifts[0] << losses[0]\n"
    /*DOC*/    "GREEN = PACKED_COLOR & masks[1] >> shifts[1] << losses[1]\n"
    /*DOC*/    "BLUE = PACKED_COLOR & masks[2] >> shifts[2] << losses[2]\n"
    /*DOC*/    "There is also an alpha channel for some Surfaces.\n"
#endif







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
	if(!surf->lockcount && (surf->subsurface || !surf->surf->pixels))
	{
		if(SDL_LockSurface(surf->surf) == -1)
		{
			PyErr_SetString(PyExc_RuntimeError, "error locking surface");
			return 0;
		}
		surf->didlock = 1;
	}
	surf->lockcount++;
	return 1;
}


static int PySurface_Unlock(PyObject* surfobj)
{
	PySurfaceObject* surf = (PySurfaceObject*)surfobj;
	surf->lockcount--;
	if(!surf->lockcount && surf->didlock)
	{
		surf->didlock = 0;
		SDL_UnlockSurface(surf->surf);
	}
	if(surf->lockcount < 0)
	{
		PyErr_SetString(PyExc_RuntimeError, "attempt to unlock an unlocked surface");
		return 0;
	}
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
	0,						     /*size*/
	"SurfLifeLock",              /*name*/
	sizeof(PyLifetimeLockObject),/*basic size*/
	0,						     /*itemsize*/
	lifelock_dealloc,		     /*dealloc*/
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
