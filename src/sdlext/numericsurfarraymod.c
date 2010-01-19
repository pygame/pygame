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

*/

#define PYGAME_SDLSURFARRAY_INTERNAL

#include "pgsdl.h"
#include "numeric_arrayobject.h"
#include "sdlextnumericsurfarray_doc.h"

#ifndef IS_PYTHON_3

/*macros used to blit arrays*/
#define COPYMACRO_2D(DST, SRC)                                          \
    Py_BEGIN_ALLOW_THREADS;                                             \
    for (loopy = 0; loopy < sizey; ++loopy)                             \
    {                                                                   \
        DST* imgrow = (DST*)(((char*)surf->pixels)+loopy*surf->pitch);  \
        Uint8* datarow = (Uint8*)array->data + stridey * loopy;         \
        for (loopx = 0; loopx < sizex; ++loopx)                         \
            *(imgrow + loopx) = (DST)*(SRC*)(datarow + stridex * loopx);\
    }                                                                   \
    Py_END_ALLOW_THREADS;

#define COPYMACRO_2D_24(SRC)                                            \
    Py_BEGIN_ALLOW_THREADS;                                             \
    for (loopy = 0; loopy < sizey-1; ++loopy)                           \
    {                                                                   \
        Uint8* imgrow = ((Uint8*)surf->pixels)+loopy*surf->pitch;       \
        Uint8* datarow = (Uint8*)array->data + stridey * loopy;         \
        for (loopx = 0; loopx < sizex; ++loopx)                         \
            *(int*)(imgrow + loopx*3) =                                 \
                (int)*(SRC*)(datarow + stridex * loopx)<<8;             \
    }                                                                   \
    {                                                                   \
        char* imgrow = ((char*)surf->pixels)+loopy*surf->pitch;         \
        char* datarow = array->data + stridey * loopy;                  \
        for (loopx = 0; loopx < sizex-1; ++loopx)                       \
            *(int*)(imgrow + loopx*3) =                                 \
                ((int)*(SRC*)(datarow + stridex * loopx))<<8;           \
    }                                                                   \
    Py_END_ALLOW_THREADS;


#define COPYMACRO_3D(DST, SRC)                                          \
    Py_BEGIN_ALLOW_THREADS;                                             \
    for (loopy = 0; loopy < sizey; ++loopy)                             \
    {                                                                   \
        DST* data = (DST*)(((char*)surf->pixels) + surf->pitch * loopy);\
        char* pix = array->data + stridey * loopy;                      \
        for (loopx = 0; loopx < sizex; ++loopx) {                       \
            *data++ = (DST)(*(SRC*)(pix) >> Rloss << Rshift) |          \
                (*(SRC*)(pix+stridez) >> Gloss << Gshift) |             \
                (*(SRC*)(pix+stridez2) >> Bloss << Bshift);             \
            pix += stridex;                                             \
        }                                                               \
    }                                                                   \
    Py_END_ALLOW_THREADS;

#define COPYMACRO_3D_24(SRC)                                            \
    Py_BEGIN_ALLOW_THREADS;                                             \
    for (loopy = 0; loopy < sizey; ++loopy)                             \
    {                                                                   \
        Uint8* data = ((Uint8*)surf->pixels) + surf->pitch * loopy;     \
        Uint8* pix = (Uint8*)array->data + stridey * loopy;             \
        for (loopx = 0; loopx < sizex; ++loopx) {                       \
            *data++ = (Uint8)*(SRC*)(pix+stridez2);                     \
            *data++ = (Uint8)*(SRC*)(pix+stridez);                      \
            *data++ = (Uint8)*(SRC*)(pix);                              \
            pix += stridex;                                             \
        }                                                               \
    }                                                                   \
    Py_END_ALLOW_THREADS;

static PyObject* _pixels3d (PyObject* self, PyObject* arg);
static PyObject* _pixels2d (PyObject* self, PyObject* arg);
static PyObject* _pixels_alpha (PyObject* self, PyObject* arg);
static PyObject* _array3d (PyObject* self, PyObject* arg);
static PyObject* _array2d (PyObject* self, PyObject* arg);
static PyObject* _array_alpha (PyObject* self, PyObject* arg);
static PyObject* _array_colorkey (PyObject* self, PyObject* arg);
static PyObject* _map_array (PyObject* self, PyObject* arg);
static PyObject* _blit_array (PyObject* self, PyObject* arg);
static PyObject* _make_surface (PyObject* self, PyObject* arg);

static PyMethodDef _surfarray_methods[] =
{
    { "pixels2d", _pixels2d, METH_O, DOC_NUMERICSURFARRAY_PIXELS2D },
    { "pixels3d", _pixels3d, METH_O, DOC_NUMERICSURFARRAY_PIXELS3D },
    { "pixels_alpha", _pixels_alpha, METH_O,
      DOC_NUMERICSURFARRAY_PIXELS_ALPHA },
    { "array2d", _array2d, METH_O, DOC_NUMERICSURFARRAY_ARRAY2D },
    { "array3d", _array3d, METH_O, DOC_NUMERICSURFARRAY_ARRAY3D },
    { "array_alpha", _array_alpha, METH_O,
      DOC_NUMERICSURFARRAY_ARRAY_ALPHA },
    { "array_colorkey", _array_colorkey, METH_O,
      DOC_NUMERICSURFARRAY_ARRAY_COLORKEY },
    { "map_array", _map_array, METH_VARARGS,
      DOC_NUMERICSURFARRAY_MAP_ARRAY },
    { "blit_array", _blit_array, METH_VARARGS,
      DOC_NUMERICSURFARRAY_BLIT_ARRAY },
    { "make_surface", _make_surface, METH_O,
      DOC_NUMERICSURFARRAY_MAKE_SURFACE },
    { NULL, NULL, 0, NULL}
};

static PyObject*
_pixels3d (PyObject* self, PyObject* arg)
{
    int dim[3];
    PyObject* array;
    SDL_Surface* surf;
    char* startpixel;
    int pixelstep;
    const int lilendian = (SDL_BYTEORDER == SDL_LIL_ENDIAN);
    PyObject* lifelock;
    int rgb = 0;
    
    if (!PySDLSurface_Check (arg))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    surf = ((PySDLSurface*)arg)->surface;
    
    if (surf->format->BytesPerPixel <= 2 || surf->format->BytesPerPixel > 4)
    {
        PyErr_SetString (PyExc_ValueError,
            "unsupported bit depth for 3D reference array");
        return NULL;
    }

    /*must discover information about how data is packed*/
    if (surf->format->Rmask == 0xff<<16 &&
        surf->format->Gmask == 0xff<<8 &&
        surf->format->Bmask == 0xff)
    {
        pixelstep = (lilendian ? -1 : 1);
        rgb = 1;
    }
    else if (surf->format->Bmask == 0xff<<16 &&
             surf->format->Gmask == 0xff<<8 &&
             surf->format->Rmask == 0xff)
    {
        pixelstep = (lilendian ? 1 : -1);
        rgb = 0;
    }
    else
    {
        PyErr_SetString (PyExc_ValueError,
            "unsupport colormasks for 3D reference array");
        return NULL;
    }

    /*create the referenced array*/
    dim[0] = surf->w;
    dim[1] = surf->h;
    dim[2] = 3; /*could be 4 if alpha in the house*/

    /* Pass a dummy string and set it correctly later */
    array = PyArray_FromDimsAndData (3, dim, PyArray_UBYTE, "");
    if (!array)
        return NULL;

    lifelock = PySDLSurface_AcquireLockObj (arg, array);
    if (!lifelock)
    {
        Py_DECREF (array);
        return NULL;
    }
    
    if (rgb)
        startpixel = ((char*) surf->pixels) + (lilendian ? 2 : 0);
    else
        startpixel = ((char*) surf->pixels) + (lilendian ? 0 : 2);
    if (!lilendian && surf->format->BytesPerPixel == 4)
        ++startpixel;

    ((PyArrayObject*) array)->flags = OWN_DIMENSIONS|OWN_STRIDES|SAVESPACE;
    ((PyArrayObject*) array)->strides[2] = pixelstep;
    ((PyArrayObject*) array)->strides[1] = surf->pitch;
    ((PyArrayObject*) array)->strides[0] = surf->format->BytesPerPixel;
    ((PyArrayObject*) array)->base = lifelock;
    ((PyArrayObject*) array)->data = startpixel;
    
    return array;
}

static PyObject*
_pixels2d (PyObject* self, PyObject* arg)
{
    int types[] = { PyArray_UBYTE, PyArray_SHORT, 0, PyArray_INT };
    int dim[3];
    int type;
    PyObject *array;
    SDL_Surface* surf;
    PyObject* lifelock;
    
    if (!PySDLSurface_Check (arg))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    surf = ((PySDLSurface*) arg)->surface;
    
    if (surf->format->BytesPerPixel == 3 || surf->format->BytesPerPixel < 1 
        || surf->format->BytesPerPixel > 4)
    {
        PyErr_SetString (PyExc_ValueError,
            "unsupported bit depth for 2D reference array");
        return NULL;
    }
    
    dim[0] = surf->w;
    dim[1] = surf->h;
    type = types[surf->format->BytesPerPixel-1];
    array = PyArray_FromDimsAndData (2, dim, type, "");
    if (!array)
        return NULL;

    lifelock = PySDLSurface_AcquireLockObj (arg, array);
    if (!lifelock)
    {
        Py_DECREF (array);
        return NULL;
    }
    ((PyArrayObject*) array)->strides[1] = surf->pitch;
    ((PyArrayObject*) array)->strides[0] = surf->format->BytesPerPixel;
    ((PyArrayObject*) array)->flags = OWN_DIMENSIONS|OWN_STRIDES;
    ((PyArrayObject*) array)->base = lifelock;
    ((PyArrayObject*) array)->data = (char*) surf->pixels;
    
    return array;
}

static PyObject*
_pixels_alpha (PyObject* self, PyObject* arg)
{
    int dim[3];
    PyObject *array;
    PyObject* lifelock;
    SDL_Surface* surf;
    char* startpixel;
    int startoffset;

    const int lilendian = (SDL_BYTEORDER == SDL_LIL_ENDIAN);

    if (!PySDLSurface_Check (arg))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    surf = ((PySDLSurface*)arg)->surface;
    
    if (surf->format->BytesPerPixel != 4)
    {
        PyErr_SetString (PyExc_ValueError,
            "unsupported bit depth for alpha array");
        return NULL;
    }

    /*must discover information about how data is packed*/
    if (surf->format->Amask == 0xff << 24)
        startoffset = (lilendian ? 3 : 0);
    else if (surf->format->Amask == 0xff)
        startoffset = (lilendian ? 0 : 3);
    else
    {
        PyErr_SetString (PyExc_ValueError,
            "unsupport colormasks for alpha reference array");
        return NULL;
    }

    dim[0] = surf->w;
    dim[1] = surf->h;
    array = PyArray_FromDimsAndData (2, dim, PyArray_UBYTE, "");
    if (!array)
        return NULL;
    
    lifelock = PySDLSurface_AcquireLockObj (arg, array);
    if (!lifelock)
    {
        Py_DECREF (array);
        return NULL;
    }
    startpixel = ((char*) surf->pixels) + startoffset;
    ((PyArrayObject*) array)->strides[1] = surf->pitch;
    ((PyArrayObject*) array)->strides[0] = surf->format->BytesPerPixel;
    ((PyArrayObject*) array)->flags = OWN_DIMENSIONS|OWN_STRIDES;
    ((PyArrayObject*) array)->base = lifelock;
    ((PyArrayObject*) array)->data = startpixel;
    
    return array;
}

static PyObject*
_array2d (PyObject* self, PyObject* arg)
{
    int dim[2], loopy;
    Uint8* data;
    PyObject *surfobj, *array;
    SDL_Surface* surf;
    int stridex, stridey;
    
    if (!PySDLSurface_Check (arg))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    surf = ((PySDLSurface*)arg)->surface;

    dim[0] = surf->w;
    dim[1] = surf->h;
    
    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
    {
        PyErr_SetString (PyExc_ValueError,
            "unsupport bit depth for surface array");
        return NULL;
    }
    array = PyArray_FromDims (2, dim, PyArray_INT);
    if (!array)
        return NULL;
    
    stridex = ((PyArrayObject*) array)->strides[0];
    stridey = ((PyArrayObject*) array)->strides[1];

    if (!PySDLSurface_AddRefLock (arg, array))
    {
        Py_DECREF (array);
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS; /* TODO: will that work? */

    switch (surf->format->BytesPerPixel)
    {
    case 1:
        for (loopy = 0; loopy < surf->h; ++loopy)
        {
            Uint8* pix = (Uint8*)(((char*) surf->pixels) + loopy * surf->pitch);
            Uint8* end = (Uint8*)(((char*) pix) + surf->w);
            data = ((Uint8*)((PyArrayObject*) array)->data) + stridey*loopy;
            while (pix < end)
            {
                *(Uint32*)data = *pix++;
                data += stridex;
            }
        }
        break;
    case 2:
        for (loopy = 0; loopy < surf->h; ++loopy)
        {
            Uint16* pix = (Uint16*)(((char*)surf->pixels) + loopy *surf->pitch);
            Uint16* end = (Uint16*)(((char*) pix) + surf->w * 2);
            data = ((Uint8*)((PyArrayObject*) array)->data) + stridey * loopy;
            while (pix < end)
            {
                *(Uint32*)data = *pix++;
                data += stridex;
            }
        }
        break;
    case 3:
        for (loopy = 0; loopy < surf->h; ++loopy)
        {
            Uint8* pix = (Uint8*)(((char*)surf->pixels)+loopy*surf->pitch);
            Uint8* end = pix+surf->w*3;
            data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
            while (pix < end)
            {
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                *(Uint32*)data = pix[0] + (pix[1]<<8) + (pix[2]<<16);
#else
                *(Uint32*)data = pix[2] + (pix[1]<<8) + (pix[0]<<16);
#endif
                pix += 3;
                data += stridex;
            }
        }
        break;
    default: /*case 4*/
        for (loopy = 0; loopy < surf->h; ++loopy)
        {
            Uint32* pix = (Uint32*)(((char*)surf->pixels)+loopy*surf->pitch);
            Uint32* end = (Uint32*)(((char*)pix)+surf->w*4);
            data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
            while (pix < end)
            {
                *(Uint32*)data = *pix++;
                data += stridex;
            }
        }
        break;
    }

    Py_END_ALLOW_THREADS;

    PySDLSurface_RemoveRefLock (arg, array);
    return array;
}

static PyObject*
_array3d (PyObject* self, PyObject* arg)
{
    int dim[3], loopy;
    Uint8* data;
    PyObject *array;
    SDL_Surface* surf;
    SDL_PixelFormat* format;
    int Rmask, Gmask, Bmask, Rshift, Gshift, Bshift, Rloss, Gloss, Bloss;
    int stridex, stridey;
    SDL_Color* palette;
    
    if (!PySDLSurface_Check (arg))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    surf = ((PySDLSurface*)arg)->surface;
    
    format = surf->format;
    dim[0] = surf->w;
    dim[1] = surf->h;
    dim[2] = 3;
    Rmask = format->Rmask;
    Gmask = format->Gmask;
    Bmask = format->Bmask;
    Rshift = format->Rshift;
    Gshift = format->Gshift;
    Bshift = format->Bshift;
    Rloss = format->Rloss;
    Gloss = format->Gloss;
    Bloss = format->Bloss;
    
    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
    {
        PyErr_SetString (PyExc_ValueError,
            "unsupported bit depth for surface array");
        return NULL;
    }
    
    array = PyArray_FromDims (3, dim, PyArray_UBYTE);
    if (!array)
        return NULL;
    
    stridex = ((PyArrayObject*) array)->strides[0];
    stridey = ((PyArrayObject*) array)->strides[1];

    if (format->BytesPerPixel == 1)
    {
        if (!format->palette)
        {
            Py_DECREF (array);
            PyErr_SetString (PyExc_RuntimeError, "8bit surface has no palette");
            return NULL;
        }
    }

    if (!PySDLSurface_AddRefLock (arg, array))
    {
        Py_DECREF (array);
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS; /* TODO: will that work? */

    switch (format->BytesPerPixel)
    {
    case 1:
        palette = format->palette->colors;
        for (loopy = 0; loopy < surf->h; ++loopy)
        {
            Uint8* pix = (Uint8*)(((char*) surf->pixels) + loopy * surf->pitch);
            Uint8* end =
                (Uint8*)(((char*)pix) + surf->w * surf->format->BytesPerPixel);
            data = ((Uint8*)((PyArrayObject*)array)->data) + stridey * loopy;
            while (pix < end)
            {
                SDL_Color* c = palette + (*pix++);
                data[0] = c->r;
                data[1] = c->g;
                data[2] = c->b;
                data += stridex;
            }
        }
        break;
    case 2:
        for (loopy = 0; loopy < surf->h; ++loopy)
        {
            Uint16* pix = (Uint16*)(((char*)surf->pixels) + loopy *surf->pitch);
            Uint16* end =
                (Uint16*)(((char*)pix) + surf->w * surf->format->BytesPerPixel);
            data = ((Uint8*)((PyArrayObject*)array)->data) + stridey * loopy;
            while (pix < end)
            {
                Uint32 color = *pix++;
                data[0] = ((color&Rmask)>>Rshift)<<Rloss;
                data[1] = ((color&Gmask)>>Gshift)<<Gloss;
                data[2] = ((color&Bmask)>>Bshift)<<Bloss;
                data += stridex;
            }
        }
        break;
    case 3:
        for (loopy = 0; loopy < surf->h; ++loopy)
        {
            Uint8* pix = (Uint8*)(((char*)surf->pixels)+loopy*surf->pitch);
            Uint8* end =
                (Uint8*)(((char*)pix)+surf->w*surf->format->BytesPerPixel);
            data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
            while (pix < end)
            {
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                Uint32 color = (pix[0]) + (pix[1]<<8) + (pix[2]<<16); pix += 3;
#else
                Uint32 color = (pix[2]) + (pix[1]<<8) + (pix[0]<<16); pix += 3;
#endif
                /*assume no loss on 24bit*/
                data[0] = ((color&Rmask)>>Rshift)/*<<Rloss*/;
                data[1] = ((color&Gmask)>>Gshift)/*<<Gloss*/;
                data[2] = ((color&Bmask)>>Bshift)/*<<Bloss*/;
                data += stridex;
            }
        }
        break;
    default: /*case 4*/
        for (loopy = 0; loopy < surf->h; ++loopy)
        {
            Uint32* pix = (Uint32*)(((char*)surf->pixels)+loopy*surf->pitch);
            Uint32* end =
                (Uint32*)(((char*)pix)+surf->w*surf->format->BytesPerPixel);
            data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
            while (pix < end)
            {
                Uint32 color = *pix++;
                /*assume no loss on 32bit*/
                data[0] = ((color&Rmask)>>Rshift)/*<<Rloss*/;
                data[1] = ((color&Gmask)>>Gshift)/*<<Gloss*/;
                data[2] = ((color&Bmask)>>Bshift)/*<<Bloss*/;
                data += stridex;
            }
        }
        break;
    }

    Py_END_ALLOW_THREADS;

    PySDLSurface_RemoveRefLock (arg, array);
    return array;
}

static PyObject*
_array_alpha (PyObject* self, PyObject* arg)
{
    int dim[2], loopy;
    Uint8* data;
    Uint32 color;
    PyObject *array;
    SDL_Surface* surf;
    int stridex, stridey;
    int Ashift, Amask, Aloss;
    
    if (!PySDLSurface_Check (arg))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    surf = ((PySDLSurface*)arg)->surface;
    
    dim[0] = surf->w;
    dim[1] = surf->h;
    
    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
    {
        PyErr_SetString (PyExc_ValueError,
            "unsupported bit depth for alpha array");
        return NULL;
    }

    array = PyArray_FromDims (2, dim, PyArray_UBYTE);
    if (!array)
        return NULL;
    
    Amask = surf->format->Amask;
    Ashift = surf->format->Ashift;
    Aloss = surf->format->Aloss;
    
    if (!Amask || surf->format->BytesPerPixel == 1) /*no pixel alpha*/
    {
        memset (((PyArrayObject*) array)->data, 255,
            (size_t) surf->w * surf->h);
        return array;
    }
    
    stridex = ((PyArrayObject*) array)->strides[0];
    stridey = ((PyArrayObject*) array)->strides[1];
    
    if (!PySDLSurface_AddRefLock (arg, array))
    {
        Py_DECREF (array);
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS; /* TODO: working? */

    switch (surf->format->BytesPerPixel)
    {
    case 2:
        for (loopy = 0; loopy < surf->h; ++loopy)
        {
            Uint16* pix = (Uint16*)(((char*)surf->pixels)+loopy*surf->pitch);
            Uint16* end = (Uint16*)(((char*)pix)+surf->w*2);
            data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
            while (pix < end)
            {
                color = *pix++;
                *data = (color & Amask) >> Ashift << Aloss;
                data += stridex;
            }
        }
        break;
    case 3:
        for (loopy = 0; loopy < surf->h; ++loopy)
        {
            Uint8* pix = (Uint8*)(((char*)surf->pixels)+loopy*surf->pitch);
            Uint8* end = pix+surf->w*3;
            data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
            while (pix < end)
            {
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                color = pix[0] + (pix[1]<<8) + (pix[2]<<16);
#else
                color = pix[2] + (pix[1]<<8) + (pix[0]<<16);
#endif
                *data = (color & Amask) >> Ashift << Aloss;
                pix += 3;
                data += stridex;
            }
        }
        break;
    default: /*case 4*/
        for (loopy = 0; loopy < surf->h; ++loopy)
        {
            Uint32* pix = (Uint32*)(((char*)surf->pixels)+loopy*surf->pitch);
            Uint32* end = (Uint32*)(((char*)pix)+surf->w*4);
            data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
            while (pix < end)
            {
                color = *pix++;
                /*assume no loss in 32bit*/
                *data = (color & Amask) >> Ashift /*<< Aloss*/;
                data += stridex;
            }
        }
        break;
    }

    Py_END_ALLOW_THREADS;
    
    PySDLSurface_RemoveRefLock (arg, array);
    return array;
}

static PyObject*
_array_colorkey (PyObject* self, PyObject* arg)
{
    int dim[2], loopy;
    Uint8* data;
    Uint32 color, colorkey;
    PyObject *array;
    SDL_Surface* surf;
    int stridex, stridey;
    
    if (!PySDLSurface_Check (arg))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    surf = ((PySDLSurface*)arg)->surface;
    
    dim[0] = surf->w;
    dim[1] = surf->h;
    
    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
    {
        PyErr_SetString (PyExc_ValueError,
            "unsupport bit depth for colorkey array");
        return NULL;
    }

    array = PyArray_FromDims (2, dim, PyArray_UBYTE);
    if(!array)
        return NULL;
    
    colorkey = surf->format->colorkey;
    if (!(surf->flags & SDL_SRCCOLORKEY)) /*no pixel alpha*/
    {
        memset (((PyArrayObject*)array)->data, 255,
            (size_t) surf->w * surf->h);
        return array;
    }
    
    stridex = ((PyArrayObject*) array)->strides[0];
    stridey = ((PyArrayObject*) array)->strides[1];
    
    if (!PySDLSurface_AddRefLock (arg, array))
    {
        Py_DECREF (array);
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS; /* TODO:...*/ 

    switch (surf->format->BytesPerPixel)
    {
    case 1:
        for (loopy = 0; loopy < surf->h; ++loopy)
        {
            Uint8* pix = (Uint8*)(((char*)surf->pixels)+loopy*surf->pitch);
            Uint8* end = (Uint8*)(((char*)pix)+surf->w);
            data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
            while (pix < end)
            {
                color = *pix++;
                *data = (color != colorkey) * 255;
                data += stridex;
            }
        }
        break;
    case 2:
        for (loopy = 0; loopy < surf->h; ++loopy)
        {
            Uint16* pix = (Uint16*)(((char*)surf->pixels)+loopy*surf->pitch);
            Uint16* end = (Uint16*)(((char*)pix)+surf->w*2);
            data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
            while (pix < end)
            {
                color = *pix++;
                *data = (color != colorkey) * 255;
                data += stridex;
            }
        }
        break;
    case 3:
        for (loopy = 0; loopy < surf->h; ++loopy)
        {
            Uint8* pix = (Uint8*)(((char*)surf->pixels)+loopy*surf->pitch);
            Uint8* end = pix+surf->w*3;
            data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
            while (pix < end)
            {
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                color = pix[0] + (pix[1]<<8) + (pix[2]<<16);
#else
                color = pix[2] + (pix[1]<<8) + (pix[0]<<16);
#endif
                *data = (color != colorkey) * 255;
                pix += 3;
                data += stridex;
            }
        }
        break;
    default: /*case 4*/
        for (loopy = 0; loopy < surf->h; ++loopy)
        {
            Uint32* pix = (Uint32*)(((char*)surf->pixels)+loopy*surf->pitch);
            Uint32* end = (Uint32*)(((char*)pix)+surf->w*4);
            data = ((Uint8*)((PyArrayObject*)array)->data) + stridey*loopy;
            while (pix < end)
            {
                color = *pix++;
                *data = (color != colorkey) * 255;
                data += stridex;
            }
        }
        break;
    }

    Py_END_ALLOW_THREADS;

    PySDLSurface_RemoveRefLock (arg, array);
    return array;
}

static PyObject*
_map_array (PyObject* self, PyObject* arg)
{
    int* data;
    PyObject *surfobj, *arrayobj, *newarray;
    SDL_Surface* surf;
    SDL_PixelFormat* format;
    PyArrayObject* array;
    int loopx, loopy;
    int stridex, stridey, stridez, stridez2, sizex, sizey;
    int dims[2];
    
    if (!PyArg_ParseTuple (arg, "OO", &surfobj, &arrayobj))
        return NULL;
    if (!PySDLSurface_Check (surfobj))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PyArray_Check (arrayobj))
    {
        PyErr_SetString (PyExc_TypeError, "array must be a numeric array");
        return NULL;
    }

    surf = ((PySDLSurface*)surfobj)->surface;
    format = surf->format;
    array = (PyArrayObject*) arrayobj;
    
    if (!array->nd || array->dimensions[array->nd - 1] != 3)
    {
        PyErr_SetString (PyExc_ValueError,
            "array must be a 3d array of 3-value color data");
        return NULL;
    }

    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
    {
        PyErr_SetString (PyExc_ValueError,
            "unsupported bit depth for surface array");
        return NULL;
    }

    switch (array->nd)
    {
    case 3: /*image of colors*/
        dims[0] = array->dimensions[0];
        dims[1] = array->dimensions[1];
        newarray = PyArray_FromDims (2, dims, PyArray_INT);
        if (!newarray)
            return NULL;
        data = (int*)((PyArrayObject*) newarray)->data;
        stridex = array->strides[0];
        stridey = array->strides[1];
        stridez = array->strides[2];
        sizex = array->dimensions[0];
        sizey = array->dimensions[1];
        break;
    case 2: /*list of colors*/
        dims[0] = array->dimensions[0];
        newarray = PyArray_FromDims (1, dims, PyArray_INT);
        if (!newarray)
            return NULL;
        data = (int*)((PyArrayObject*) newarray)->data;
        stridex = 0;
        stridey = array->strides[0];
        stridez = array->strides[1];
        sizex = 1;
        sizey = array->dimensions[0];
        break;
#if 1 /*kinda like a scalar here, use normal map_rgb*/
    case 1: /*single color*/
        dims[0] = 1;
        newarray = PyArray_FromDims (1, dims, PyArray_INT);
        if (!newarray)
            return NULL;
        data = (int*)((PyArrayObject*) newarray)->data;
        stridex = 0;
        stridey = 0;
        stridez = array->strides[0];
        sizex = 1;
        sizey = 1;
        break;
#endif
    default:
        PyErr_SetString (PyExc_ValueError, "unsupported array shape");
        return NULL;
    }

    stridez2 = stridez * 2;

    switch (array->descr->elsize)
    {
    case sizeof (char):
        Py_BEGIN_ALLOW_THREADS;
        for (loopx = 0; loopx < sizex; ++loopx)
        {
            char* col = array->data + stridex * loopx;
            for (loopy = 0; loopy < sizey; ++loopy)
            {
                char* pix = col + stridey * loopy;
                *data++ = (*((unsigned char*)(pix)) >>
                           format->Rloss << format->Rshift) |
                    (*((unsigned char*)(pix+stridez)) >>
                     format->Gloss << format->Gshift) |
                    (*((unsigned char*)(pix+stridez2)) >>
                     format->Bloss << format->Bshift);
            }
        }
        Py_END_ALLOW_THREADS;
        break;
    case sizeof (short):
        Py_BEGIN_ALLOW_THREADS;
        for (loopx = 0; loopx < sizex; ++loopx)
        {
            char* col = array->data + stridex * loopx;
            for (loopy = 0; loopy < sizey; ++loopy)
            {
                char* pix = col + stridey * loopy;
                *data++ = (*((unsigned short*)(pix)) >>
                           format->Rloss << format->Rshift) |
                    (*((unsigned short*)(pix+stridez)) >>
                     format->Gloss << format->Gshift) |
                    (*((unsigned short*)(pix+stridez2)) >>
                     format->Bloss << format->Bshift);
            }
        }
        Py_END_ALLOW_THREADS;
        break;
    case sizeof(int):
        Py_BEGIN_ALLOW_THREADS;
        for (loopx = 0; loopx < sizex; ++loopx)
        {
            char* col = array->data + stridex * loopx;
            for (loopy = 0; loopy < sizey; ++loopy)
            {
                char* pix = col + stridey * loopy;
                *data++ = (*((int*)(pix)) >>
                           format->Rloss << format->Rshift) |
                    (*((int*)(pix+stridez)) >>
                     format->Gloss << format->Gshift) |
                    (*((int*)(pix+stridez2)) >>
                     format->Bloss << format->Bshift);
            }
        }
        Py_END_ALLOW_THREADS;
        break;
    default:
        Py_DECREF (newarray);
        PyErr_SetString (PyExc_ValueError,
            "unsupported bytesperpixel for array\n");
        return NULL;
    }

    return newarray;
}

static PyObject*
_blit_array (PyObject* self, PyObject* arg)
{
    PyObject *surfobj, *arrayobj;
    SDL_Surface* surf;
    SDL_PixelFormat* format;
    PyArrayObject* array;
    int loopx, loopy;
    int stridex, stridey, stridez=0, stridez2=0, sizex, sizey;
    int Rloss, Gloss, Bloss, Rshift, Gshift, Bshift;
    
    if (!PyArg_ParseTuple (arg, "OO", &surfobj, &arrayobj))
        return NULL;
    if (!PySDLSurface_Check (surfobj))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PyArray_Check (arrayobj))
    {
        PyErr_SetString (PyExc_TypeError, "array must be a numeric array");
        return NULL;
    }

    surf = ((PySDLSurface*)surfobj)->surface;
    format = surf->format;
    array = (PyArrayObject*) arrayobj;
    
    if (!(array->nd == 2 || (array->nd == 3 && array->dimensions[2] == 3)))
    {
        PyErr_SetString (PyExc_ValueError, "must be a valid 2d or 3d array");
        return NULL;
    }

    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
    {
        PyErr_SetString (PyExc_ValueError, "unsupported bit depth for surface");
        return NULL;
    }

    stridex = array->strides[0];
    stridey = array->strides[1];
    if (array->nd == 3)
    {
        stridez = array->strides[2];
        stridez2 = stridez*2;
    }
    sizex = array->dimensions[0];
    sizey = array->dimensions[1];
    Rloss = format->Rloss; Gloss = format->Gloss; Bloss = format->Bloss;
    Rshift = format->Rshift; Gshift = format->Gshift; Bshift = format->Bshift;
    
    if (sizex != surf->w || sizey != surf->h)
    {
        PyErr_SetString (PyExc_ValueError,
            "array must match surface dimensions");
        return NULL;
    }

    if (!PySDLSurface_AddRefLock (surfobj, (PyObject *) array))
        return NULL;
    
    switch (surf->format->BytesPerPixel)
    {
    case 1:
        if (array->nd == 2)
        {
            switch (array->descr->elsize)
            {
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
                PySDLSurface_RemoveRefLock (surfobj, (PyObject *) array);
                PyErr_SetString (PyExc_ValueError,
                    "unsupported datatype for array\n");
                return NULL;
            }
        }
        break;
    case 2:
        if (array->nd == 2)
        {
            switch (array->descr->elsize)
            {
            case sizeof (Uint8):
                COPYMACRO_2D(Uint16, Uint8);
                break;
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
                PySDLSurface_RemoveRefLock (surfobj, (PyObject *) array);
                PyErr_SetString (PyExc_ValueError,
                    "unsupported datatype for array\n");
                return NULL;
            }
        }
        else
        {
            switch (array->descr->elsize)
            {
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
                PySDLSurface_RemoveRefLock (surfobj, (PyObject *) array);
                PyErr_SetString (PyExc_ValueError,
                    "unsupported datatype for array\n");
                return NULL;
            }
        }
        break;
    case 3:
        if (array->nd == 2)
        {
            switch (array->descr->elsize)
            {
            case sizeof (Uint8):
                COPYMACRO_2D_24(Uint8);
                break;
            case sizeof (Uint16):
                COPYMACRO_2D_24(Uint16);
                break;
            case sizeof (Uint32):
                COPYMACRO_2D_24(Uint32);
                break;
            case sizeof (Uint64):
                COPYMACRO_2D_24(Uint64);
                break;
            default:
                PySDLSurface_RemoveRefLock (surfobj, (PyObject *) array);
                PyErr_SetString (PyExc_ValueError,
                    "unsupported datatype for array\n");
                return NULL;
            }
        }
        else
        {
            switch (array->descr->elsize)
            {
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
                PySDLSurface_RemoveRefLock (surfobj, (PyObject *) array);
                PyErr_SetString (PyExc_ValueError,
                    "unsupported datatype for array\n");
                return NULL;
            }
        }
        break;
    case 4:
        if (array->nd == 2)
        {
            switch (array->descr->elsize)
            {
            case sizeof (Uint8):
                COPYMACRO_2D(Uint32, Uint8);
                break;
            case sizeof (Uint16):
                COPYMACRO_2D(Uint32, Uint16);
                break;
            case sizeof (Uint32):
                COPYMACRO_2D(Uint32, Uint32);
                break;
            case sizeof (Uint64):
                COPYMACRO_2D(Uint32, Uint64);
                break;
            default:
                PySDLSurface_RemoveRefLock (surfobj, (PyObject *) array);
                PyErr_SetString (PyExc_ValueError,
                    "unsupported datatype for array\n");
                return NULL;
            }
        }
        else
        {
            switch (array->descr->elsize)
            {
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
                PySDLSurface_RemoveRefLock (surfobj, (PyObject *) array);
                PyErr_SetString (PyExc_ValueError,
                    "unsupported datatype for array\n");
                return NULL;
            }
        }
        break;
    default:
        PySDLSurface_RemoveRefLock (surfobj, (PyObject *) array);
        PyErr_SetString (PyExc_RuntimeError, "unsupported bit depth for image");
        return NULL;
    }
    
    PySDLSurface_RemoveRefLock (surfobj, (PyObject *) array);
    Py_RETURN_NONE;
}

static PyObject*
_make_surface (PyObject* self, PyObject* arg)
{
    PyObject *surfobj, *args;
    SDL_Surface* surf;
    PyArrayObject* array;
    int sizex, sizey, bitsperpixel;
    Uint32 rmask, gmask, bmask;
    
    if (!PyArray_Check (arg))
    {
        PyErr_SetString (PyExc_TypeError, "array must be a numeric array");
        return NULL;
    }
    array = (PyArrayObject*) arg;
    
    if (!(array->nd == 2 || (array->nd == 3 && array->dimensions[2] == 3)))
    {
        PyErr_SetString (PyExc_ValueError, "must be a valid 2d or 3d array\n");
        return NULL;
    }
    if (array->descr->type_num > PyArray_LONG)
    {
        PyErr_SetString (PyExc_ValueError,
            "invalid array datatype for surface");
        return NULL;
    }
    
    if (array->nd == 2)
    {
        bitsperpixel = 8;
        rmask = 0xFF >> 6 << 5;
        gmask = 0xFF >> 5 << 2;
        bmask = 0xFF >> 6;
    }
    else
    {
        bitsperpixel = 32;
        rmask = 0xFF<<16;
        gmask = 0xFF<<8;
        bmask = 0xFF;
    }
    sizex = array->dimensions[0];
    sizey = array->dimensions[1];

    surf = SDL_CreateRGBSurface (0, sizex, sizey, bitsperpixel, rmask, gmask,
        bmask, 0);
    if (!surf)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }

    surfobj = PySDLSurface_NewFromSDLSurface (surf);
    if (!surfobj)
    {
        SDL_FreeSurface (surf);
        return NULL;
    }
    
    args = Py_BuildValue ("(OO)", surfobj, array);
    if (!args)
    {
        Py_DECREF (surfobj);
        return NULL;
    }
    _blit_array (NULL, args);
    Py_DECREF (args);
    
    if (PyErr_Occurred ())
    {
        Py_DECREF (surfobj);
        return NULL;
    }
    return surfobj;
}

PyMODINIT_FUNC
initnumericsurfarray (void)
{
    PyObject *mod;

    mod = Py_InitModule3 ("numericsurfarray", _surfarray_methods,
        DOC_NUMERICSURFARRAY);

    import_pygame2_base ();
    import_pygame2_sdl_base ();
    import_pygame2_sdl_video ();
    import_array ();
}

#endif /* !IS_PYTHON_3 */
