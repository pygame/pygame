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

#include <stddef.h>
#include "pygame.h"
#include "pygamedocs.h"
#include "pgarrinter.h"
#include <SDL_byteorder.h>

static int
_get_array_interface(PyObject *obj,
		     PyObject **cobj_p,
		     PyArrayInterface **inter_p)
{
    if (!(*cobj_p = PyObject_GetAttrString(obj, "__array_struct__"))) {
	if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
		PyErr_Clear();
		PyErr_SetString(PyExc_ValueError,
				"no C-struct array interface");
	    }
	return 0;
    }

    if (!PyCObject_Check(*cobj_p) ||  /* conditional 'or's */
	!(*inter_p = PyCObject_AsVoidPtr(*cobj_p)) ||
	(*inter_p)->two != 2) {
	Py_DECREF(*cobj_p);
	PyErr_SetString(PyExc_ValueError, "invalid array interface");
	return 0;
    }
    return 1;
}

/*macros used to blit arrays*/
#define COPYMACRO_2D(DST, SRC)                                           \
    for (loopy = 0; loopy < sizey; ++loopy)                              \
    {                                                                    \
        DST* imgrow = (DST*)(((char*)surf->pixels)+loopy*surf->pitch);   \
        Uint8* datarow = (Uint8*)array_data + stridey * loopy;           \
        for (loopx = 0; loopx < sizex; ++loopx)                          \
            *(imgrow + loopx) = (DST)*(SRC*)(datarow + stridex * loopx); \
    }

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
#define COPYMACRO_2D_24_PIXEL(pix, data, SRC)    			\
    *pix++ = *data;					                \
    *pix++ = *(data + 1);					        \
    *pix++ = *(data + 2);
#else
#define COPYMACRO_2D_24_PIXEL(pix, data, SRC)    			\
    *pix++ = *(data + sizeof (SRC) - 1);				\
    *pix++ = *(data + sizeof (SRC) - 2);				\
    *pix++ = *(data + sizeof (SRC) - 3);
#endif

#define COPYMACRO_2D_24(SRC)                                            \
    for (loopy = 0; loopy < sizey; ++loopy)                             \
    {                                                                   \
        Uint8 *pix = ((Uint8 *)surf->pixels) + loopy * surf->pitch;     \
        Uint8 *data = (Uint8 *)array_data + stridey * loopy;            \
        Uint8 *end = pix + 3 * sizex;                                   \
        while (pix != end) {                                            \
            COPYMACRO_2D_24_PIXEL(pix, data, SRC)			\
            data += stridex;                                            \
        }                                                               \
    }

#define COPYMACRO_3D(DST, SRC)                                          \
    for (loopy = 0; loopy < sizey; ++loopy)                             \
    {                                                                   \
        DST* pix = (DST*)(((char*)surf->pixels) + surf->pitch * loopy); \
        char* data = array_data + stridey * loopy;                      \
        for (loopx = 0; loopx < sizex; ++loopx) {                       \
            *pix++ = (DST)(*(SRC*)(data) >> Rloss << Rshift) |          \
                (*(SRC*)(data+stridez) >> Gloss << Gshift) |            \
                (*(SRC*)(data+stridez2) >> Bloss << Bshift) |           \
                alpha;                                                  \
            data += stridex;                                            \
        }                                                               \
    }

#define COPYMACRO_3D_24(SRC)                                            \
    for (loopy = 0; loopy < sizey; ++loopy)                             \
    {                                                                   \
        Uint8 *pix = ((Uint8*)surf->pixels) + surf->pitch * loopy;      \
        Uint8 *data = (Uint8*)array_data + stridey * loopy;             \
	Uint8 *end = pix + 3 * sizex;                                   \
        while (pix != end) {                                            \
            *pix++ = (Uint8)*(SRC*)(data + stridez_0);                  \
            *pix++ = (Uint8)*(SRC*)(data + stridez_1);                  \
            *pix++ = (Uint8)*(SRC*)(data + stridez_2);                  \
            data += stridex;                                            \
        }                                                               \
    }

static PyObject*
blit_array(PyObject* self, PyObject* arg)
{
    PyObject *surfobj, *arrayobj;
    PyObject *cobj;
    PyArrayInterface *inter;
    void *array_data;
    SDL_Surface* surf;
    SDL_PixelFormat* format;
    int loopx, loopy;
    int stridex, stridey, stridez=0, stridez2=0, sizex, sizey;
    int Rloss, Gloss, Bloss, Rshift, Gshift, Bshift;
    char typekind, *cptr;
    const char *validtypekinds = "iuSV";
    
    if (!PyArg_ParseTuple(arg, "O!O", &PySurface_Type, &surfobj, &arrayobj)) {
        return NULL;
    }
    surf = PySurface_AsSurface(surfobj);
    format = surf->format;
    
    if (!_get_array_interface(arrayobj, &cobj, &inter)) {
        return 0;
    }

    typekind = inter->typekind;
    for (cptr = validtypekinds; *cptr && *cptr != typekind; ++cptr) {
	;
    }
    if (!*cptr) {
	Py_DECREF(cobj);
	PyErr_Format(PyExc_ValueError, "unsupported array type '%c'",
		     typekind);
	return NULL;
    }

    if (!(inter->nd == 2 || (inter->nd == 3 && inter->shape[2] == 3)))
        return RAISE(PyExc_ValueError, "must be a valid 2d or 3d array\n");

    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
        return RAISE(PyExc_ValueError, "unsupport bit depth for surface");

    stridex = inter->strides[0];
    stridey = inter->strides[1];
    if (inter->nd == 3) {
        stridez = inter->strides[2];
        stridez2 = stridez*2;
    }
    sizex = inter->shape[0];
    sizey = inter->shape[1];
    Rloss = format->Rloss; Gloss = format->Gloss; Bloss = format->Bloss;
    Rshift = format->Rshift; Gshift = format->Gshift; Bshift = format->Bshift;

    if (sizex != surf->w || sizey != surf->h) {
        Py_DECREF(cobj);
        return RAISE(PyExc_ValueError, "array must match surface dimensions");
    }
    if (!PySurface_LockBy(surfobj, (PyObject *) arrayobj)) {
        Py_DECREF(cobj);
        return NULL;
    }
    
    array_data = inter->data;

    switch (surf->format->BytesPerPixel) {
    case 1:
        if (inter->nd == 2) {
            switch (inter->itemsize) {
            case sizeof (Uint8):
                COPYMACRO_2D(Uint8, Uint8);
                break;
            case sizeof (Uint16):
                COPYMACRO_2D(Uint8, Uint16);
                break;
            case sizeof (Uint32):
                COPYMACRO_2D(Uint8, Uint32);
                break;
            case sizeof (Uint64):
                COPYMACRO_2D(Uint8, Uint64);
                break;
            default:
                Py_DECREF(cobj);
                if (!PySurface_UnlockBy(surfobj, (PyObject *) arrayobj)) {
                    return NULL;
                }
                return RAISE(PyExc_ValueError,
                             "unsupported datatype for array\n");
            }
        }
        else {
            Py_DECREF(cobj);
            if (!PySurface_UnlockBy(surfobj, (PyObject *) arrayobj)) {
                return NULL;
            }
            return RAISE(PyExc_ValueError,
                         "unsupported datatype for array\n");
        }
        break;
    case 2:
        if (inter->nd == 2) {
            switch (inter->itemsize) {
            case sizeof (Uint16):
                COPYMACRO_2D(Uint16, Uint16);
                break;
            case sizeof (Uint32):
                COPYMACRO_2D(Uint16, Uint32);
                break;
            case sizeof (Uint64):
                COPYMACRO_2D(Uint16, Uint64);
                break;
            default:
                Py_DECREF(cobj);
                if (!PySurface_UnlockBy(surfobj, (PyObject *) arrayobj)) {
                    return NULL;
                }
                return RAISE(PyExc_ValueError,
                             "unsupported datatype for array\n");
            }
        }
        else {
	    Uint16 alpha = 0;
	    if (format->Amask) {
		alpha = 255 >> format->Aloss << format->Ashift;
	    }
            switch (inter->itemsize) {
            case sizeof (Uint8):
                COPYMACRO_3D(Uint16, Uint8);
                break;
            case sizeof (Uint16):
                COPYMACRO_3D(Uint16, Uint16);
                break;
            case sizeof (Uint32):
                COPYMACRO_3D(Uint16, Uint32);
                break;
            case sizeof (Uint64):
                COPYMACRO_3D(Uint16, Uint64);
                break;
            default:
                Py_DECREF(cobj);
                if (!PySurface_UnlockBy(surfobj, (PyObject *) arrayobj)) {
                    return NULL;
                }
                return RAISE(PyExc_ValueError,
                             "unsupported datatype for array\n");
            }
        }
        break;
    case 3:
        /* Assumption: The rgb components of a 24 bit pixel are in
           separate bytes.
	*/
        if (inter->nd == 2) {
            switch (inter->itemsize) {
            case sizeof (Uint32):
                COPYMACRO_2D_24(Uint32);
                break;
            case sizeof (Uint64):
                COPYMACRO_2D_24(Uint64);
                break;
            default:
                Py_DECREF(cobj);
                if (!PySurface_UnlockBy(surfobj, (PyObject *) arrayobj)) {
                    return NULL;
                }
                return RAISE(PyExc_ValueError,
                             "unsupported datatype for array\n");
            }
        }
        else {
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
	    size_t stridez_0 = (format->Rmask & 0x0000ff ? 0        : 
			        format->Gmask & 0x0000ff ? stridez  :
                                                           stridez2   );
	    size_t stridez_1 = (format->Rmask & 0x00ff00 ? 0        :
			        format->Gmask & 0x00ff00 ? stridez  :
                                                           stridez2   );
	    size_t stridez_2 = (format->Rmask & 0xff0000 ? 0        :
			        format->Gmask & 0xff0000 ? stridez  :
                                                           stridez2   );
#else
	    size_t stridez_2 = (format->Rmask & 0x0000ff ? 0        : 
			        format->Gmask & 0x0000ff ? stridez  :
                                                           stridez2   );
	    size_t stridez_1 = (format->Rmask & 0x00ff00 ? 0        :
			        format->Gmask & 0x00ff00 ? stridez  :
                                                           stridez2   );
	    size_t stridez_0 = (format->Rmask & 0xff0000 ? 0        :
			        format->Gmask & 0xff0000 ? stridez  :
                                                           stridez2   );
#endif
            switch (inter->itemsize) {
            case sizeof (Uint8):
                COPYMACRO_3D_24(Uint8);
                break;
            case sizeof (Uint16):
                COPYMACRO_3D_24(Uint16);
                break;
            case sizeof (Uint32):
                COPYMACRO_3D_24(Uint32);
                break;
            case sizeof (Uint64):
                COPYMACRO_3D_24(Uint64);
                break;
            default:
                Py_DECREF(cobj);
                if (!PySurface_UnlockBy(surfobj, (PyObject *) arrayobj)) {
                    return NULL;
                }
                return RAISE(PyExc_ValueError,
                             "unsupported datatype for array\n");
            }
        }
        break;
    case 4:
        if (inter->nd == 2) {
            switch (inter->itemsize) {
            case sizeof (Uint32):
                COPYMACRO_2D(Uint32, Uint32);
                break;
            case sizeof (Uint64):
                COPYMACRO_2D(Uint32, Uint64);
                break;
            default:
                Py_DECREF(cobj);
                if (!PySurface_UnlockBy(surfobj, (PyObject *) arrayobj)) {
                    return NULL;
                }
                return RAISE(PyExc_ValueError,
                            "unsupported datatype for array\n");
            }
        }
        else {
	    Uint32 alpha = 0;
	    if (format->Amask) {
		alpha = 255 >> format->Aloss << format->Ashift;
	    }
            switch (inter->itemsize) {
            case sizeof (Uint8):
                COPYMACRO_3D(Uint32, Uint8);
                break;
            case sizeof (Uint16):
                COPYMACRO_3D(Uint32, Uint16);
                break;
            case sizeof (Uint32):
                COPYMACRO_3D(Uint32, Uint32);
                break;
            case sizeof (Uint64):
                COPYMACRO_3D(Uint32, Uint64);
                break;
            default:
                Py_DECREF(cobj);
                if (!PySurface_UnlockBy(surfobj, (PyObject *) arrayobj)) {
                    return NULL;
                }
                return RAISE(PyExc_ValueError,
                             "unsupported datatype for array\n");
            }
        }
        break;
    default:
        Py_DECREF(cobj);
        if (!PySurface_UnlockBy(surfobj, (PyObject *) arrayobj)) {
            return NULL;
        }
        return RAISE(PyExc_RuntimeError, "unsupported bit depth for image");
    }
    
    Py_DECREF(cobj);
    if (!PySurface_UnlockBy(surfobj, (PyObject *) arrayobj)) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyMethodDef surfarray_builtins[] =
{
    { "blit_array", blit_array, METH_VARARGS, DOC_PYGAMESURFARRAYBLITARRAY },
    { NULL, NULL, 0, NULL}
};

PYGAME_EXPORT
void init_arraysurfarray(void)
{
    PyObject *module;
    
    module = Py_InitModule3("_arraysurfarray", surfarray_builtins,
                            DOC_PYGAMESURFARRAY);
    import_pygame_base();
    import_pygame_surface();
}
