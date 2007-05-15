/*
    Copyright (C) 2002-2007 Ulf Ekstrom except for the bitcount function.
    This wrapper code was originally written by Danny van Bruggen(?) for
    the SCAM library, it was then converted by Ulf Ekstrom to wrap the
    bitmask library, a spinoff from SCAM.

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

*/

#include "SDL.h"
#include "pygame.h"
#include "pygamedocs.h"
#include "bitmask.h"


typedef struct {
  PyObject_HEAD
  bitmask_t *mask;
} PyMaskObject;

staticforward PyTypeObject PyMask_Type;
#define PyMask_Check(x) ((x)->ob_type == &PyMask_Type)
#define PyMask_AsBitmap(x) (((PyMaskObject*)x)->mask)



/* mask object methods */

static PyObject* mask_get_size(PyObject* self, PyObject* args)
{
	bitmask_t *mask = PyMask_AsBitmap(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return Py_BuildValue("(ii)", mask->w, mask->h);
}

static PyObject* mask_get_at(PyObject* self, PyObject* args)
{
	bitmask_t *mask = PyMask_AsBitmap(self);
        int x, y, val;

	if(!PyArg_ParseTuple(args, "(ii)", &x, &y))
		return NULL;
	if (x >= 0 && x < mask->w && y >= 0 && y < mask->h)
	  val = bitmask_getbit(mask, x, y);
	else
	  return NULL;

        return PyInt_FromLong(val);
}

static PyObject* mask_set_at(PyObject* self, PyObject* args)
{
	bitmask_t *mask = PyMask_AsBitmap(self);
        int x, y, value = 1;

	if(!PyArg_ParseTuple(args, "(ii)|i", &x, &y, &value))
		return NULL;
	if (x >= 0 && x < mask->w && y >= 0 && y < mask->h)
	  {
	    if (value)
	      bitmask_setbit(mask, x, y);
	    else
	      bitmask_clearbit(mask, x, y);
	  }
	else
	  {
	    return NULL;
	  }
	Py_INCREF(Py_None);
        return Py_None;
}

static PyObject* mask_overlap(PyObject* self, PyObject* args)
{
	bitmask_t *mask = PyMask_AsBitmap(self);
        bitmask_t *othermask;
        PyObject *maskobj;
        int x, y, val;
	int xp,yp;

	if(!PyArg_ParseTuple(args, "O!(ii)", &PyMask_Type, &maskobj, &x, &y))
		return NULL;
	othermask = PyMask_AsBitmap(maskobj);

	val = bitmask_overlap_pos(mask, othermask, x, y, &xp, &yp);
	if (val)
	  return Py_BuildValue("(ii)", xp,yp);
	else
	  {
	    Py_INCREF(Py_None);
	    return Py_None;	  
	  }
}


static PyObject* mask_overlap_area(PyObject* self, PyObject* args)
{
	bitmask_t *mask = PyMask_AsBitmap(self);
        bitmask_t *othermask;
        PyObject *maskobj;
        int x, y, val;

	if(!PyArg_ParseTuple(args, "O!(ii)", &PyMask_Type, &maskobj, &x, &y))
		return NULL;
	othermask = PyMask_AsBitmap(maskobj);

	val = bitmask_overlap_area(mask, othermask, x, y);
        return PyInt_FromLong(val);
}

static PyMethodDef maskobj_builtins[] =
{
	{ "get_size", mask_get_size, METH_VARARGS, DOC_MASKGETSIZE },
	{ "get_at", mask_get_at, METH_VARARGS, DOC_MASKGETAT },
	{ "set_at", mask_set_at, METH_VARARGS, DOC_MASKSETAT },
	{ "overlap", mask_overlap, METH_VARARGS, DOC_MASKOVERLAP },
	{ "overlap_area", mask_overlap_area, METH_VARARGS, DOC_MASKOVERLAPAREA },

	{ NULL, NULL }
};



/*mask object internals*/

static void mask_dealloc(PyObject* self)
{
	bitmask_t *mask = PyMask_AsBitmap(self);
	bitmask_free(mask);
	PyObject_DEL(self);
}


static PyObject* mask_getattr(PyObject* self, char* attrname)
{
	return Py_FindMethod(maskobj_builtins, self, attrname);
}


static PyTypeObject PyMask_Type = 
{
	PyObject_HEAD_INIT(NULL)
	0,
	"Mask",
	sizeof(PyMaskObject),
	0,
	mask_dealloc,	
	0,
	mask_getattr,
	0,
	0,
	0,
	0,
	NULL,
	0, 
	(hashfunc)NULL,
	(ternaryfunc)NULL,
	(reprfunc)NULL,
	0L,0L,0L,0L,
	DOC_PYGAMEMASK /* Documentation string */
};



/*mask module methods*/

static PyObject* Mask(PyObject* self, PyObject* args)
{
	bitmask_t *mask;
	int w,h;
        PyMaskObject *maskobj;
	if(!PyArg_ParseTuple(args, "(ii)", &w, &h))
		return NULL;
        mask = bitmask_create(w,h);

	if(!mask)
	  return NULL; /*RAISE(PyExc_Error, "cannot create bitmask");*/
        
        /*create the new python object from mask*/        
	maskobj = PyObject_New(PyMaskObject, &PyMask_Type);
        if(maskobj)
        	maskobj->mask = mask;
	return (PyObject*)maskobj;
}



static PyMethodDef mask_builtins[] =
{
	{ "Mask", Mask, 1, DOC_PYGAMEMASK },

	{ NULL, NULL }
};



void initmask(void)
{
  PyObject *module, *dict;
  PyType_Init(PyMask_Type);
  
  /* create the module */
  module = Py_InitModule3("mask", mask_builtins, DOC_PYGAMEMASK);
  dict = PyModule_GetDict(module);
  PyDict_SetItemString(dict, "MaskType", (PyObject *)&PyMask_Type);
}

