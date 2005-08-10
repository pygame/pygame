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

typedef struct
{
  PyObject_HEAD
	SDL_Overlay *cOverlay;
	GAME_Rect cRect;
} PyGameOverlay;


static void
overlay_dealloc(PyGameOverlay *self)
{
	if(SDL_WasInit(SDL_INIT_VIDEO) && self->cOverlay)
		SDL_FreeYUVOverlay(self->cOverlay);

	PyObject_Free((PyObject*)self);
}



    /*DOC*/ static char doc_Overlay_SetLocation[] =
    /*DOC*/    "Overlay.set_location(rectstyle) -> None\n"
    /*DOC*/    "set overlay location\n"
    /*DOC*/    "\n"
    /*DOC*/    "Sets location for the overlay on a screen.\n"
    /*DOC*/    "This does not move or redraw any currently displayed data,\n"
    /*DOC*/    "it only sets the position for newly display() calls.\n"
    /*DOC*/ ;

static PyObject* Overlay_SetLocation(PyGameOverlay *self, PyObject *args)
{
        GAME_Rect *rect, temp;
    
        rect = GameRect_FromObject(args, &temp);
        if(!rect)
            return RAISE(PyExc_TypeError, "Invalid rectstyle argument");
        
        self->cRect.x = rect->x;
        self->cRect.y = rect->y;
        self->cRect.w = rect->w;
        self->cRect.h = rect->h;

	RETURN_NONE
}



    /*DOC*/ static char doc_Overlay_Display[] =
    /*DOC*/    "Overlay.display(y, u, v) -> None\n"
    /*DOC*/    "display the yuv data\n"
    /*DOC*/    "\n"
    /*DOC*/    "Display the yuv data in SDL's overlay planes. The y, u, and v\n"
    /*DOC*/    "arguments represents strings of byte data.\n"
    /*DOC*/ ;

static PyObject* Overlay_Display(PyGameOverlay *self, PyObject *args)
{
	// Parse data params for frame
	int ls_y, ls_u, ls_v, y;
	unsigned char *dst_y, *dst_u, *dst_v, *src_y, *src_u, *src_v;
	if(!PyArg_ParseTuple(args, "(s#s#s#)", &src_y, &ls_y, &src_u, &ls_u, &src_v, &ls_v))
		return NULL;

	{
		SDL_Rect cRect= { self->cRect.x, self->cRect.y, self->cRect.w, self->cRect.h };
		SDL_LockYUVOverlay( self->cOverlay );

		// No clipping at this time( only support for YUV420 )
		dst_y = (char*)self->cOverlay->pixels[ 0 ];
		dst_v = (char*)self->cOverlay->pixels[ 1 ];
		dst_u = (char*)self->cOverlay->pixels[ 2 ];
		for (y=0; y< self->cOverlay->h; y++)
		{
			memcpy( dst_y, src_y, self->cOverlay->w );

			src_y += ls_y / self->cOverlay->h;
			dst_y += self->cOverlay->pitches[ 0 ];

			if (y & 1) {
				src_u += ( ls_u* 2 )/self->cOverlay->h;
				src_v += ( ls_v* 2 )/self->cOverlay->h;
				dst_u += self->cOverlay->pitches[ 1 ];
				dst_v += self->cOverlay->pitches[ 2 ];
			}
			else
			{
				memcpy( dst_u, src_u, ( ls_u* 2 )/self->cOverlay->h );
				memcpy( dst_v, src_v, ( ls_v* 2 )/self->cOverlay->h );
			}
		}

		SDL_UnlockYUVOverlay( self->cOverlay );
                SDL_DisplayYUVOverlay( self->cOverlay, &cRect);
	}
	RETURN_NONE
}



PyObject* Overlay_New(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	int pixelformat;
	PyGameOverlay *self;
	int w, h;
        SDL_Surface *screen;
	if(!PyArg_ParseTuple(args, "i(ii)", &pixelformat, &w, &h))
		return NULL;

	if(!SDL_WasInit(SDL_INIT_VIDEO))
		return RAISE(PyExc_SDLError, "cannot create overlay without pygame.display initialized");

        screen = SDL_GetVideoSurface();
        if(!screen)
            return RAISE(PyExc_SDLError, "Display mode not set");
        
	// Create new Overlay object
	self= (PyGameOverlay *)type->tp_alloc(type, 0);
        if( !self )
		return NULL;

	// Create layer with desired format
	self->cOverlay = SDL_CreateYUVOverlay(w, h, pixelformat, screen);
	if( !self->cOverlay )
		return RAISE(PyExc_SDLError, "Cannot create overlay");

	self->cRect.x= 0;
	self->cRect.y= 0;
	self->cRect.w= w;
	self->cRect.h= h;

	return (PyObject*)self;
}


static PyMethodDef Overlay_methods[] = {
  {"set_location", (PyCFunction)Overlay_SetLocation, METH_VARARGS, doc_Overlay_SetLocation},
  {"display", (PyCFunction)Overlay_Display, METH_VARARGS, doc_Overlay_Display},
	{NULL}  /* Sentinel */
};


    /*DOC*/ static char doc_Overlay[] =
    /*DOC*/    "pygame.Overlay(pixeltype, [width, height]) -> Overlay\n"
    /*DOC*/    "Create a new video overlay object\n"
    /*DOC*/    "\n"
    /*DOC*/    "This creates a new Overlay object. Overlays represent a basic\n"
    /*DOC*/    "interface for putting YUV image data into the graphics card's\n"
    /*DOC*/    "video overlay planes. This is a low level object intended for\n"
    /*DOC*/    "use by people who know what they are doing, and have pregenerated\n"
    /*DOC*/    "YUV image data.\n"
    /*DOC*/    "The pixeltype argument must be one of the pygame constants;\n"
    /*DOC*/    "YV12_OVERLAY, IYUV_OVERLAY, YUV2_OVERLAY, UYVY_OVERLAY, or YVYU_OVERLAY.\n"
    /*DOC*/    "\n"
    /*DOC*/ ;

#if 0
    /*DOC*/ static char doc_Overlay_MODULE[] =
    /*DOC*/    "pygame.Overlay(pixeltype, [width, height]) -> Overlay\n"
    /*DOC*/    "Create a new video overlay object\n"
    /*DOC*/    "\n"
    /*DOC*/    "This creates a new Overlay object. Overlays represent a basic\n"
    /*DOC*/    "interface for putting YUV image data into the graphics card's\n"
    /*DOC*/    "video overlay planes. This is a low level object intended for\n"
    /*DOC*/    "use by people who know what they are doing, and have pregenerated\n"
    /*DOC*/    "YUV image data.\n"
    /*DOC*/    "The pixeltype argument must be one of the pygame constants;\n"
    /*DOC*/    "YV12_OVERLAY, IYUV_OVERLAY, YUV2_OVERLAY, UYVY_OVERLAY, or YVYU_OVERLAY.\n"
    /*DOC*/    "\n"
    /*DOC*/    "\n"
    /*DOC*/ ;
#endif

PyTypeObject PyOverlay_Type =
{
	PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "pygame.overlay",        /*tp_name*/
    sizeof(PyGameOverlay),      /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    0,												/*tp_dealloc*/
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
    doc_Overlay,           /* tp_doc */
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

static PyMethodDef overlay_methods[] =
{
	{ NULL, NULL }
};


PYGAME_EXPORT
void initoverlay(void)
{
	PyObject *module;
	module = Py_InitModule("overlay", overlay_methods );

	PyOverlay_Type.ob_type = &PyType_Type;
	PyOverlay_Type.tp_dealloc = (destructor)overlay_dealloc;
	PyOverlay_Type.tp_alloc =PyType_GenericAlloc;
	PyOverlay_Type.tp_getattro = PyObject_GenericGetAttr;
	Py_INCREF((PyObject *)&PyOverlay_Type);
	PyType_Init(PyOverlay_Type);

    /* create the module reference */
	PyModule_AddObject(module, "Overlay", (PyObject *)&PyOverlay_Type);

	import_pygame_base();
	import_pygame_rect();    
}
