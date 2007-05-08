/*
  This wrapper code was originally written by Danny van Bruggen(?) for
  the SCAM library, it was then converted by Ulf Ekstrom to wrap the
  bitmask library, a spinoff from SCAM.

  This file is released under the LGPL licence, see COPYING for 
  details.
*/

#include <Python.h>
#include "bitmask.h"

typedef struct {
  PyObject_HEAD
  bitmask_t *mask;
} PyMaskObject;

staticforward PyTypeObject PyMask_Type;
#define PyMask_Check(x) ((x)->ob_type == &PyMask_Type)
#define PyMask_AsBitmap(x) (((PyMaskObject*)x)->mask)


#define PyType_Init(x)                                          \
{                                                               \
    x.ob_type = &PyType_Type;                                   \
}


/* mask object methods */

static char doc_mask_get_size[] = "Mask.get_size() -> width,height";
static PyObject* mask_get_size(PyObject* self, PyObject* args)
{
	bitmask_t *mask = PyMask_AsBitmap(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return Py_BuildValue("(ii)", mask->w, mask->h);
}

static char doc_mask_get_at[] = "Mask.get_at((x,y)) -> int";
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

static char doc_mask_set_at[] = "Mask.set_at((x,y),value)";
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

static char doc_mask_overlap[] = "Mask.overlap(othermask, offset) -> x,y";
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


static char doc_mask_overlap_area[] = "Mask.overlap_area(othermask, offset) -> numpixels";
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
	{ "get_size", mask_get_size, METH_VARARGS, doc_mask_get_size },
	{ "get_at", mask_get_at, METH_VARARGS, doc_mask_get_at },
	{ "set_at", mask_set_at, METH_VARARGS, doc_mask_set_at },
	{ "overlap", mask_overlap, METH_VARARGS, doc_mask_overlap },
	{ "overlap_area", mask_overlap_area, METH_VARARGS, doc_mask_overlap_area },

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


static char doc_mask_object[] = "2d bitmask";
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
	doc_mask_object /* Documentation string */
};



/*mask module methods*/

static char doc_Mask[] = "mask.Mask((width, height)) -> Mask";
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
	maskobj = PyObject_NEW(PyMaskObject, &PyMask_Type);
        if(maskobj)
        	maskobj->mask = mask;
	return (PyObject*)maskobj;
}



static PyMethodDef mask_builtins[] =
{
	{ "Mask", Mask, 1, doc_Mask },

	{ NULL, NULL }
};




static char doc_mask_module[] = "wrapper for the bitmask pixel collision library";

void initmask(void)
{
  PyObject *module, *dict;
  PyType_Init(PyMask_Type);
  
  /* create the module */
  module = Py_InitModule3("mask", mask_builtins, doc_mask_module);
  dict = PyModule_GetDict(module);
  PyDict_SetItemString(dict, "MaskType", (PyObject *)&PyMask_Type);
}

