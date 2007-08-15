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
	  {
            PyErr_Format(PyExc_IndexError, "%d, %d is out of bounds", x, y);
	    return NULL;
	  }

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
            PyErr_Format(PyExc_IndexError, "%d, %d is out of bounds", x, y);
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

/*
def maskFromSurface(surface, threshold = 127):
    mask = pygame.Mask(surface.get_size())
    key = surface.get_colorkey()
    if key:
        for y in range(surface.get_height()):
            for x in range(surface.get_width()):
                if surface.get_at((x+0.1,y+0.1)) != key:
                    mask.set_at((x,y),1)
    else:
        for y in range(surface.get_height()):
            for x in range (surface.get_width()):
                if surface.get_at((x,y))[3] > threshold:
                    mask.set_at((x,y),1)
    return mask
*/

PyObject* mask_from_surface(PyObject* self, PyObject* args)
{
        //TODO:
	bitmask_t *mask;
	SDL_Surface* surf;
        PyObject* surfobj;
        PyMaskObject *maskobj;

        int x, y, threshold;
        Uint8 *pixels;

        SDL_PixelFormat *format;
        Uint32 color;
        Uint8 *pix;
        Uint8 r, g, b, a;

        printf("hi there1\n");

        //TODO: set threshold as 127 default argument.
        threshold = 127;

        /* get the surface from the passed in arguments. 
         *   surface, threshold
         */

        if (!PyArg_ParseTuple (args, "O!|(i)", &PySurface_Type, &surfobj, &threshold))
            return NULL;

        printf("hi there2\n");
	surf = PySurface_AsSurface(surfobj);

        printf("hi there3\n");

        /* lock the surface, release the GIL. */
        PySurface_Lock (surfobj);

        /* get the size from the surface, and create the mask. */
        mask = bitmask_create(surf->w, surf->h);

	if(!mask)
	  return NULL; /*RAISE(PyExc_Error, "cannot create bitmask");*/
        
        


        /* TODO: this is the slow, but easy to code way.  Could make the loop 
         *         just increment a pointer depending on the format.  
         *         It's faster than in python anyhow.
         */
        pixels = (Uint8 *) surf->pixels;
        format = surf->format;

        for(y=0; y < surf->h; y++) {
            for(x=0; x < surf->w; x++) {
                switch (format->BytesPerPixel)
                {
                case 1:
                    color = (Uint32)*((Uint8 *) pixels + y * surf->pitch + x);
                    break;
                case 2:
                    color = (Uint32)*((Uint16 *) (pixels + y * surf->pitch) + x);
                    break;
                case 3:
                    pix = ((Uint8 *) (pixels + y * surf->pitch) + x * 3);
            #if SDL_BYTEORDER == SDL_LIL_ENDIAN
                    color = (pix[0]) + (pix[1] << 8) + (pix[2] << 16);
            #else
                    color = (pix[2]) + (pix[1] << 8) + (pix[0] << 16);
            #endif
                    break;
                default:                  /* case 4: */
                    color = *((Uint32 *) (pixels + y * surf->pitch) + x);
                    break;
                }


                if (surf->flags & SDL_SRCCOLORKEY) {

                    SDL_GetRGBA (color, format, &r, &g, &b, &a);

                    /* no colorkey, so we check the threshold of the alpha */
                    if (a > threshold) {
                        bitmask_setbit(mask, x, y);
                    }
                } else {
                    /*  test against the colour key. */
                    if (format->colorkey != color) {
                        bitmask_setbit(mask, x, y);
                    }
                }
            }
        }



        /* unlock the surface, release the GIL.
         */
        PySurface_Unlock (surfobj);

        /*create the new python object from mask*/        
	maskobj = PyObject_New(PyMaskObject, &PyMask_Type);
        if(maskobj)
            maskobj->mask = mask;


	return (PyObject*)maskobj;
}





static PyMethodDef maskobj_builtins[] =
{
	{ "get_size", mask_get_size, METH_VARARGS, DOC_PYGAMEMASKGETSIZE},
	{ "get_at", mask_get_at, METH_VARARGS, DOC_PYGAMEMASKGETAT },
	{ "set_at", mask_set_at, METH_VARARGS, DOC_PYGAMEMASKSETAT },
	{ "overlap", mask_overlap, METH_VARARGS, DOC_PYGAMEMASKOVERLAP },
	{ "overlap_area", mask_overlap_area, METH_VARARGS, DOC_PYGAMEMASKOVERLAPAREA },

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
	{ "Mask", Mask, 1, DOC_PYGAMEMASKPYGAMEMASK},
	{ "from_surface", mask_from_surface, METH_VARARGS, DOC_PYGAMEMASKPYGAMEMASKFROMSURFACE},
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

