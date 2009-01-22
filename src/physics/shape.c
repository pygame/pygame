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

#define PHYSICS_SHAPE_INTERNAL

#include "physicsmod.h"
#include "pgphysics.h"

static void _shape_dealloc (PyShape *shape);

static PyObject* _shape_getdict (PyShape *shape, void *closure);
static PyObject* _shape_getinertia (PyShape *shape, void *closure);
static PyObject* _shape_getrotation (PyShape *shape, void *closure);

static PyObject* _shape_getvertices (PyShape *shape);
static PyObject* _shape_getaabbox (PyShape *shape);
static PyObject* _shape_collide (PyShape *shape, PyObject *args);
static PyObject* _shape_update (PyShape *shape, PyObject *args);

/**
 * Methods, which are bound to the PyShape type.
 */
static PyMethodDef _shape_methods[] =
{
    { "get_vertices", (PyCFunction) _shape_getvertices, METH_NOARGS, NULL },
    { "get_aabbox", (PyCFunction) _shape_getaabbox, METH_NOARGS, NULL },
    { "collide", (PyCFunction) _shape_collide, METH_VARARGS, NULL },
    { "update", (PyCFunction) _shape_update, METH_VARARGS, NULL },
    { NULL, NULL, 0, NULL }
};

/**
 * Getters/Setters
 */
static PyGetSetDef _shape_getsets[] =
{
    { "__dict__", (getter) _shape_getdict, NULL, NULL, NULL },
    { "inertia",  (getter) _shape_getinertia, NULL, NULL, NULL },
    { "rotation", (getter) _shape_getrotation, NULL, NULL, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

PyTypeObject PyShape_Type =
{
    TYPE_HEAD(NULL, 0)
    "physics.Shape",            /* tp_name */
    sizeof (PyShape),           /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _shape_dealloc, /* tp_dealloc */
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
    _shape_methods,             /* tp_methods */
    0,                          /* tp_members */
    _shape_getsets,             /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    offsetof (PyShape, dict),   /* tp_dictoffset */
    0,                          /* tp_init */
    0,                          /* tp_alloc */
    0,                          /* tp_new */
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

static void
_shape_dealloc (PyShape *shape)
{
    Py_XDECREF (shape->dict);
    ((PyObject*)shape)->ob_type->tp_free ((PyObject*)shape);
}

/* Getters/Setters */
static PyObject*
_shape_getdict (PyShape *shape, void *closure)
{
    if (!shape->dict)
    {
        shape->dict = PyDict_New ();
        if (!shape->dict)
            return NULL;
    }
    Py_INCREF (shape->dict);
    return shape->dict;
}

static PyObject*
_shape_getinertia (PyShape *shape, void *closure)
{
    return PyFloat_FromDouble (shape->inertia);
}

static PyObject*
_shape_getrotation (PyShape *shape, void *closure)
{
    return PyFloat_FromDouble (RAD2DEG (shape->rotation));
}

/* Methods */
static PyObject*
_shape_getvertices (PyShape *shape)
{
    if (shape->get_vertices)
    {
        Py_ssize_t i, count;
        PyObject *list, *tuple;
        PyVector2 *vertices;

        vertices = shape->get_vertices (shape, &count);
        if (!vertices)
        {
            /* TODO: does get_vertices() set the error? */
            return NULL;
        }

        list = PyList_New (count);
        if (!list)
        {
            PyMem_Free (vertices);
            return NULL;
        }

        for (i = 0; i < count; i++)
        {
            tuple = PyVector2_AsTuple (vertices[i]);
            if (!tuple)
            {
                Py_DECREF (list);
                PyMem_Free (vertices);
                return NULL;
            }
            PyList_SET_ITEM (list, i, tuple);
        }
        PyMem_Free (vertices);
        return list;
    }

    PyErr_SetString (PyExc_NotImplementedError, "method not implemented");
    return NULL;
}

static PyObject*
_shape_getaabbox (PyShape *shape)
{
    if (shape->get_aabbox)
    {
        PyObject *rect;
        AABBox* box = shape->get_aabbox (shape);
        rect = AABBox_AsFRect (box);
        PyMem_Free (box);
        return rect;
    }
    PyErr_SetString (PyExc_NotImplementedError, "method not implemented");
    return NULL;
}

static PyObject*
_shape_collide (PyShape *shape, PyObject *args)
{
    PyObject *shape2;
    PyVector2 pos = { 0, 0 };
    int swap, refid;
    collisionfunc collider = NULL;

    if (!PyArg_ParseTuple (args, "O!:collide", &PyShape_Type, &shape2))
        return NULL;
    collider = PyCollision_GetCollisionFunc (shape->type,
        ((PyShape*)shape2)->type, &swap);

    if (collider)
    {
        if (swap)
        {
            PyShape *tmp = shape;
            shape = (PyShape*) shape2;
            shape2 = (PyObject*) tmp;
        }
        return collider (shape, pos, 0.0, (PyShape*)shape2, pos, 0.0, &refid);
    }

    PyErr_SetString (PyExc_NotImplementedError, "method not implemented");
    return NULL;
}

static PyObject*
_shape_update (PyShape *shape, PyObject *args)
{
    if (shape->update)
    {
        PyObject *body;
        if (!PyArg_ParseTuple (args, "O!:update", &PyBody_Type, &body))
            return NULL;
        shape->update (shape, (PyBody*)body);
        Py_RETURN_NONE;
    }
    PyErr_SetString (PyExc_NotImplementedError, "method not implemented");
    return NULL;
}

/* C API */
PyObject*
PyShape_Collide (PyObject *shape1, PyVector2 pos1, double rot1,
    PyObject *shape2, PyVector2 pos2, double rot2, int *refid)
{
    if (!PyShape_Check (shape1) || !PyShape_Check (shape2))
    {
        PyErr_SetString (PyExc_TypeError,
            "shape arguments must be Shape objects");
        return NULL;
    }

    return PyShape_Collide_FAST ((PyShape*)shape1, pos1, rot1, (PyShape*)shape2,
        pos2, rot2, refid);
}

PyObject*
PyShape_Collide_FAST (PyShape *shape1, PyVector2 pos1, double rot1,
    PyShape *shape2, PyVector2 pos2, double rot2, int *refid)
{
    int swap = 0;
    collisionfunc LHCollider = NULL;
    
    /* The boxes overlap - run the exact collision checks. */
    LHCollider = PyCollision_GetCollisionFunc (shape1->type, shape2->type,
        &swap);
    
    if (!LHCollider)
    {
        /* No collisionfunc found, use the shape's implementation. */
        return PyObject_CallMethod ((PyObject*)shape1, "collide", "O",
            (PyObject*) shape2, NULL);
    }

    if (swap)
    {
        PyShape *tmp = shape1;
        shape1 = shape2;
        shape2 = tmp;
    }

    return LHCollider (shape1, pos1, rot1, shape2, pos2, rot2, refid);
}

int
PyShape_Update (PyObject *shape, PyObject *body)
{
    if (!PyShape_Check (shape))
    {
        PyErr_SetString (PyExc_TypeError, "shape must be a Shape");
        return 0;
    }
    if (!PyBody_Check (body))
    {
        PyErr_SetString (PyExc_TypeError, "body must be a Body");
        return 0;
    }
    return PyShape_Update_FAST ((PyShape*)shape, (PyBody*) body);
}

int
PyShape_Update_FAST (PyShape *shape, PyBody *body)
{
    int retval = 1;
    PyObject *result;
    
    if (shape->update)
        return shape->update (shape, body);
        
    /* No update method, use the shape's implementation. */
    result =  PyObject_CallMethod ((PyObject*)shape, "update", "O",
        (PyObject*) body, NULL);
    if (!result)
        retval = 0;
    Py_XDECREF (result);
    return retval;
}

AABBox*
PyShape_GetAABBox (PyObject *shape)
{
    if (!PyShape_Check (shape))
    {
        PyErr_SetString (PyExc_TypeError, "shape must be a Shape");
        return NULL;
    }
    return PyShape_GetAABBox_FAST ((PyShape*)shape);
}

AABBox*
PyShape_GetAABBox_FAST (PyShape *shape)
{
    PyObject *result;
    AABBox *box = NULL;
    
    if (shape->get_aabbox)
        return shape->get_aabbox (shape);

    result =  PyObject_CallMethod ((PyObject*)shape, "get_aabbox", NULL);
    if (!result)
        return NULL;
    box = AABBox_FromSequence (result);
    Py_XDECREF (result);
    return box;
}

PyVector2*
PyShape_GetVertices (PyObject *shape, Py_ssize_t *count)
{
    if (!count)
    {
        PyErr_SetString (PyExc_RuntimeError, "count argument missing");
        return NULL;
    }
    if (!PyShape_Check (shape))
    {
        PyErr_SetString (PyExc_TypeError, "shape must be a Shape");
        return NULL;
    }
    return PyShape_GetVertices_FAST ((PyShape*)shape, count);
}

PyVector2*
PyShape_GetVertices_FAST (PyShape *shape, Py_ssize_t *count)
{
    PyObject *result, *item;
    PyVector2* vertices;
    Py_ssize_t i;

    if (shape->get_vertices)
        return shape->get_vertices (shape, count);

    result = PyObject_CallMethod ((PyObject*)shape, "get_vertices", NULL);
    if (!result)
        return NULL;

    if (!PySequence_Check (result))
    {
        Py_DECREF (result);
        return NULL;
    }

    *count = PySequence_Size (result);
    if (*count == 0)
        return NULL;

    vertices = PyMem_New (PyVector2, *count);
    if (!vertices)
        return NULL;

    for (i = 0; i < *count; i++)
    {
        item = PySequence_ITEM (result, i);
        if (!PyVector2_FromSequence (item, &vertices[i]))
        {
            Py_XDECREF (item);
            PyMem_Free (vertices);
            return NULL;
        }
    }
    Py_DECREF (result);
    return vertices;
}

void
shape_export_capi (void **capi)
{
    capi[PHYSICS_SHAPE_FIRSTSLOT + 0] = &PyShape_Type;
    capi[PHYSICS_SHAPE_FIRSTSLOT + 1] = PyShape_Collide;
    capi[PHYSICS_SHAPE_FIRSTSLOT + 2] = PyShape_Collide_FAST;
    capi[PHYSICS_SHAPE_FIRSTSLOT + 3] = PyShape_Update;
    capi[PHYSICS_SHAPE_FIRSTSLOT + 4] = PyShape_Update_FAST;
    capi[PHYSICS_SHAPE_FIRSTSLOT + 5] = PyShape_GetAABBox;
    capi[PHYSICS_SHAPE_FIRSTSLOT + 6] = PyShape_GetAABBox_FAST;
    capi[PHYSICS_SHAPE_FIRSTSLOT + 7] = PyShape_GetVertices;
    capi[PHYSICS_SHAPE_FIRSTSLOT + 8] = PyShape_GetVertices_FAST;
}
