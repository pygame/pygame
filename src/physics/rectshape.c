/*
  pygame physics - Pygame physics module

  Copyright (C) 2008 Zhang Fan

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

#define PHYSICS_RECTSHAPE_INTERNAL

#include "physicsmod.h"
#include "pgphysics.h"

static PyVector2* _get_vertices (PyShape *shape, Py_ssize_t *count);
static AABBox* _get_aabbox (PyShape *shape);
static int _update (PyShape *shape, PyBody *body);

static PyObject *_rectshape_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);
static int _rectshape_init (PyRectShape *self, PyObject *args, PyObject *kwds);

static PyObject* _rectshape_getrect (PyRectShape *self, void *closure);

/**
 * Getters/Setters
 */
static PyGetSetDef _rectshape_getsets[] =
{
    { "rect",  (getter) _rectshape_getrect, NULL, NULL, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

PyTypeObject PyRectShape_Type =
{
    TYPE_HEAD(NULL, 0)
    "physics.RectShape",        /* tp_name */
    sizeof (PyShape),           /* tp_basicsize */
    0,                          /* tp_itemsize */
    0,                          /* tp_dealloc */
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
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "",
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    0,                          /* tp_methods */
    0,                          /* tp_members */
    _rectshape_getsets,         /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc)_rectshape_init,  /* tp_init */
    0,                          /* tp_alloc */
    _rectshape_new,             /* tp_new */
    0,                          /* tp_free */
    0,                          /* tp_is_gc */
    0,                          /* tp_bases */
    0,                          /* tp_mro */
    0,                          /* tp_cache */
    0,                          /* tp_subclasses */
    0,                          /* tp_weaklist */
    0,                          /* tp_del */
#if PY_VERSION_HEX >= 0x02060000
    0                           /* tp_version_tag */
#endif
};

static PyVector2*
_get_vertices (PyShape *shape, Py_ssize_t *count)
{
    PyRectShape *r = (PyRectShape*)shape;
    PyVector2* vertices;
    if (!r || !count)
        return NULL;
    
    vertices = PyMem_New (PyVector2, 4);
    if (!vertices)
        return NULL;

    /* Return in CCW order starting from the bottomleft. */
    vertices[0] = r->bottomleft;
    vertices[1] = r->bottomright;
    vertices[2] = r->topright;
    vertices[3] = r->topleft;

    *count = 4;
    return vertices;
}

static AABBox*
_get_aabbox (PyShape *shape)
{
    AABBox *box;
    PyRectShape *r = (PyRectShape*)shape;
    
    if (!r)
        return NULL;
    
    box = PyMem_New (AABBox, 1);
    if (!box)
        return NULL;
    
    box->top = r->box.top;
    box->left = r->box.left;
    box->bottom = r->box.bottom;
    box->right = r->box.right;
    
    return box;
}

static int
_update (PyShape *shape, PyBody *body)
{
    PyRectShape *r = (PyRectShape*)shape;
    PyVector2 gp[4];
    
    if (!shape || !body)
        return 0;

    /* Update the aabbox. */
    PyBody_GetGlobalPos (body, r->bottomleft, gp[0]);
    PyBody_GetGlobalPos (body, r->bottomright, gp[1]);
    PyBody_GetGlobalPos (body, r->topright, gp[2]);
    PyBody_GetGlobalPos (body, r->topleft, gp[3]);
    AABBox_ExpandTo (&(r->box), &(gp[0]));
    AABBox_ExpandTo (&(r->box), &(gp[1]));
    AABBox_ExpandTo (&(r->box), &(gp[2]));
    AABBox_ExpandTo (&(r->box), &(gp[3]));
    
    return 1;
}

static PyObject *_rectshape_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds)
{
    PyRectShape *shape = (PyRectShape*) type->tp_alloc (type, 0);
    if (!shape)
        return NULL;
    shape->shape.get_aabbox = _get_aabbox;
    shape->shape.get_vertices = _get_vertices;
    shape->shape.update = _update;
    shape->shape.type = RECT;
    
    AABBox_Reset (&(shape->box));
    
    return (PyObject*) shape;
}

static int
_rectshape_init (PyRectShape *self, PyObject *args, PyObject *kwds)
{
    PyObject *tuple;
    AABBox *box;

    if (PyShape_Type.tp_init ((PyObject*)self, args, kwds) < 0)
        return -1;
    
    if (!PyArg_ParseTuple (args, "O", &tuple))
        return -1;

    box = AABBox_FromSequence (tuple);
    if (!box)
        return -1;

    self->box.top = box->top;
    self->box.left = box->left;
    self->box.right = box->right;
    self->box.bottom = box->bottom;

    PyMem_Free (box);
    return 0;
}

/* Getters/Setters */
static PyObject*
_rectshape_getrect (PyRectShape *self, void *closure)
{
    AABBox box;
    
    AABBox_Reset (&box);
    AABBox_ExpandTo (&box, &(self->topleft));
    AABBox_ExpandTo (&box, &(self->topright));
    AABBox_ExpandTo (&box, &(self->bottomleft));
    AABBox_ExpandTo (&box, &(self->bottomright));
    
    return AABBox_AsFRect (&box);
}

/* C API */
PyObject*
PyRectShape_New (AABBox box)
{
    /* TODO: is anything correctly initialised? */
    PyRectShape *shape = PyObject_New (PyRectShape, &PyRectShape_Type);
    if (!shape)
        return NULL;

    shape->box.top = box.top;
    shape->box.left = box.left;
    shape->box.right = box.right;
    shape->box.bottom = box.bottom;
    return (PyObject*) shape;
}

void
rectshape_export_capi (void **capi)
{
    capi[PHYSICS_SHAPE_FIRSTSLOT + 0] = &PyRectShape_Type;
    capi[PHYSICS_SHAPE_FIRSTSLOT + 1] = &PyRectShape_New;
}
