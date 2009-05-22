/*
 *   A part of the pydfb module. Provides interface for basic overlay
 *			manipulation functions.
 *      In this implementation overlay can use only one layer out of maximum
 *      possible. After getting overlay, no overlay can be created.
 *
 *					Copyright (C) 2002-2003  Dmitry Borisov
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the Free
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Dmitry Borisov
 */

#include <Python.h>
#include "pygame.h"
#include "pgcompat.h"
#include "pygamedocs.h"

typedef struct
{
    PyObject_HEAD
    SDL_Overlay *cOverlay;
    GAME_Rect cRect;
} PyGameOverlay;

static void
overlay_dealloc (PyGameOverlay *self)
{
    if (SDL_WasInit (SDL_INIT_VIDEO) && self->cOverlay)
        SDL_FreeYUVOverlay (self->cOverlay);

    PyObject_Free ((PyObject*)self);
}

static PyObject*
Overlay_SetLocation (PyGameOverlay *self, PyObject *args)
{
    GAME_Rect *rect, temp;
    
    rect = GameRect_FromObject (args, &temp);
    if (!rect)
        return RAISE (PyExc_TypeError, "Invalid rectstyle argument");
    
    self->cRect.x = rect->x;
    self->cRect.y = rect->y;
    self->cRect.w = rect->w;
    self->cRect.h = rect->h;

    Py_RETURN_NONE;
}

static PyObject*
Overlay_Display (PyGameOverlay *self, PyObject *args)
{
    SDL_Rect cRect;
    // Parse data params for frame
    int ls_y, ls_u, ls_v, y;
    unsigned char *src_y=0, *src_u=0, *src_v=0;
	
    if (PyTuple_Size (args))
    {
        if (!PyArg_ParseTuple (args, "(s#s#s#)", &src_y, &ls_y, &src_u, &ls_u,
                               &src_v, &ls_v))
            return NULL;
    }

    if (src_y)
    {
        Uint8 *dst_y=0, *dst_u=0, *dst_v=0;
        SDL_LockYUVOverlay (self->cOverlay);

        // No clipping at this time( only support for YUV420 )

        dst_y = self->cOverlay->pixels[0];
        dst_v = self->cOverlay->pixels[1];
        dst_u = self->cOverlay->pixels[2];

        for (y = 0; y < self->cOverlay->h; y++)
        {
            memcpy (dst_y, src_y, self->cOverlay->w);

            src_y += ls_y / self->cOverlay->h;
            dst_y += self->cOverlay->pitches[0];

            if (!(y & 1))
            {
                src_u += (ls_u * 2)/self->cOverlay->h;
                src_v += (ls_v * 2)/self->cOverlay->h;
                dst_u += self->cOverlay->pitches[ 1 ];
                dst_v += self->cOverlay->pitches[ 2 ];
            }
            else
            {
                memcpy (dst_u, src_u, (ls_u * 2) / self->cOverlay->h);
                memcpy (dst_v, src_v, (ls_v * 2) / self->cOverlay->h);
            }
        }

        SDL_UnlockYUVOverlay (self->cOverlay);
    }

    cRect.x = self->cRect.x;
    cRect.y = self->cRect.y;
    cRect.w = self->cRect.w;
    cRect.h = self->cRect.h;
    SDL_DisplayYUVOverlay (self->cOverlay, &cRect);

    Py_RETURN_NONE;
}

static PyObject*
Overlay_GetHardware (PyGameOverlay *self)
{
    return PyInt_FromLong (self->cOverlay->hw_overlay);
}

PyObject*
Overlay_New (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    int pixelformat;
    PyGameOverlay *self;
    int w, h;
    SDL_Surface *screen;
    if (!PyArg_ParseTuple (args, "i(ii)", &pixelformat, &w, &h))
        return NULL;

    if (!SDL_WasInit (SDL_INIT_VIDEO))
        return RAISE
            (PyExc_SDLError,
             "cannot create overlay without pygame.display initialized");

    screen = SDL_GetVideoSurface ();
    if (!screen)
        return RAISE (PyExc_SDLError, "Display mode not set");
        
    // Create new Overlay object
    self= (PyGameOverlay *)type->tp_alloc (type, 0);
    if (!self)
        return NULL;

    // Create layer with desired format
    self->cOverlay = SDL_CreateYUVOverlay (w, h, pixelformat, screen);
    if (!self->cOverlay)
        return RAISE (PyExc_SDLError, "Cannot create overlay");

    self->cRect.x= 0;
    self->cRect.y= 0;
    self->cRect.w= w;
    self->cRect.h= h;

    return (PyObject*)self;
}

static PyMethodDef Overlay_methods[] = {
    { "set_location", (PyCFunction) Overlay_SetLocation, METH_VARARGS,
      DOC_OVERLAYSETLOCATION },
    { "display", (PyCFunction) Overlay_Display, METH_VARARGS,
      DOC_OVERLAYDISPLAY },
    { "get_hardware", (PyCFunction) Overlay_GetHardware, METH_NOARGS,
      DOC_OVERLAYGETHARDWARE },
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

PyTypeObject PyOverlay_Type =
{
    TYPE_HEAD (NULL, 0)
    "pygame.overlay",        /*tp_name*/
    sizeof(PyGameOverlay),      /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor) overlay_dealloc,	/*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,   /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    DOC_PYGAMEOVERLAY,           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    Overlay_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,    /* tp_init */
    0,       /* tp_alloc */
    Overlay_New,			   /* tp_new */
};

static PyMethodDef _overlay_methods[] =
{
    { NULL, NULL, 0, NULL }
};


MODINIT_DEFINE (overlay)
{
    PyObject *module;

#if PY3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "overlay",
        NULL,
        -1,
        _overlay_methods,
        NULL, NULL, NULL, NULL
    };
#endif

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }
    import_pygame_rect ();    
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }

    if (PyType_Ready (&PyOverlay_Type) == -1) {
        MODINIT_ERROR;
    }

    /* create the module */
#if PY3
    module = PyModule_Create (&_module);
#else
    module = Py_InitModule (MODPREFIX "overlay", _overlay_methods );
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }

    /* create the module reference */
    Py_INCREF ((PyObject *)&PyOverlay_Type);
    if (PyModule_AddObject (module, "Overlay",
                            (PyObject *)&PyOverlay_Type) == -1) {
      Py_DECREF ((PyObject *)&PyOverlay_Type);
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN (module);
}
