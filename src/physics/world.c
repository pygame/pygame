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

#define PHYSICS_WORLD_INTERNAL

#include "physicsmod.h"
#include "pgphysics.h"

#define MAX_ITERATION 5

static void _update_body_simulation (PyWorld *world, double step);
static int _detect_collisions (PyWorld *world, double step);
static void _correct_positions (PyWorld *world, double step);
static void _solve_joints (PyWorld *world, double step);
static void _update_positions (PyWorld *world, double step);

static void _world_dealloc (PyWorld *world);
static PyObject* _world_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);
static int _world_init (PyWorld *world, PyObject *args, PyObject *kwds);
    
/* Getters/Setters */
static PyObject* _world_getdict (PyWorld *world, void *closure);
static PyObject* _world_getdamping (PyWorld *world, void *closure);
static int _world_setdamping (PyWorld *world, PyObject *value, void *closure);
static PyObject* _world_getgravity (PyWorld *world, void *closure);
static int _world_setgravity (PyWorld *world, PyObject *value, void *closure);
static PyObject* _world_getbodies (PyWorld *world, void *closure);
static PyObject* _world_getjoints (PyWorld *world, void *closure);

/* Methods */
static PyObject* _world_update (PyWorld* world, PyObject* args);
static PyObject* _world_addbody (PyWorld* world, PyObject* args);
static PyObject* _world_removebody (PyWorld* world, PyObject* args);
static PyObject* _world_addjoint (PyWorld* world, PyObject* args);
static PyObject* _world_removejoint (PyWorld* world, PyObject* args);


/**
 * Methods, which are bound to the PyWorld type.
 */
static PyMethodDef _world_methods[] =
{
    { "update", (PyCFunction) _world_update, METH_VARARGS, NULL },
    { "add_body", (PyCFunction) _world_addbody, METH_VARARGS, NULL },
    { "add_joint", (PyCFunction) _world_addjoint, METH_VARARGS, NULL},
    { "remove_body", (PyCFunction) _world_removebody, METH_VARARGS, NULL},
    { "remove_joint", (PyCFunction) _world_removejoint, METH_VARARGS, NULL},
    { NULL, NULL, 0, NULL } /* The NULL sentinel is important! */
};

static PyGetSetDef _world_getsets[] =
{
    { "__dict__", (getter) _world_getdict, NULL, NULL, NULL },
    { "damping", (getter) _world_getdamping, (setter)_world_setdamping,
      NULL, NULL },
    { "gravity",(getter)_world_getgravity, (setter)_world_setgravity,
      NULL, NULL, },
    { "bodies", (getter)_world_getbodies, NULL, NULL, NULL },
    { "joints", (getter)_world_getjoints, NULL, NULL, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};


PyTypeObject PyWorld_Type =
{
    TYPE_HEAD(NULL, 0)
    "physics.World",            /* tp_name */
    sizeof (PyWorld),           /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _world_dealloc,/* tp_dealloc */
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
    _world_methods,             /* tp_methods */
    0,                          /* tp_members */
    _world_getsets,             /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    offsetof (PyWorld, dict),   /* tp_dictoffset */
    (initproc)_world_init,      /* tp_init */
    0,                          /* tp_alloc */
    _world_new,                 /* tp_new */
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

/**
 * Updates the velocity of the bodies attached to the world.
 *
 * @param world The PyWorld to update the bodies for.
 * @param step The time passed since the last update.
 */
static void
_update_body_simulation (PyWorld *world, double step)
{
    Py_ssize_t i;
    PyBody* body;
    Py_ssize_t size = PyList_Size (world->bodylist);

    for (i = 0; i < size; ++i)
    {
        body = (PyBody*) PyList_GET_ITEM (world->bodylist, i);
        PyBody_UpdateVelocity (body, world->gravity, step);
    }
}

/**
 * Checks the bodies and joints attached to the world for collisions and
 * updates them accordingly.
 *
 * @param world The PyWorld to check the bodies and joints for.
 * @param step The time passed since the last update.
 * @return 1 on success, 0 in case an error occured.
 */
static int
_detect_collisions (PyWorld *world, double step)
{
    Py_ssize_t i, j, body_cnt, contact_cnt;
    PyBody *refbody, *incbody;
    PyObject *contacts, *contact;

    body_cnt = PyList_Size (world->bodylist);
    contact_cnt = PyList_Size (world->contactlist);
    if (PyList_SetSlice (world->contactlist, 0, contact_cnt, NULL) == -1)
        return 0;

    /* For all pair of objects, do collision test        
     * update AABB
     */
    for (i = 0; i < body_cnt; ++i)
    {
        /* TODO: that should be done implicitly within the body code, so we 
         * can safely skip it. */
        refbody = (PyBody*) PyList_GET_ITEM (world->bodylist, i);
        PyShape_Update_FAST ((PyShape*)refbody->shape, refbody);
    }
    
    /* collision test */
    for (i = 0; i < body_cnt - 1; ++i)
    {
        refbody = (PyBody*) PyList_GET_ITEM (world->bodylist, i);
        for (j = i + 1; j < body_cnt; ++j)
        {
            incbody = (PyBody*) PyList_GET_ITEM (world->bodylist, j);
            if (refbody->isstatic && incbody->isstatic)
                continue;
            
            /* TODO: remove, once the collision handling is done */
            contacts = PyBody_CheckCollision_FAST (refbody, incbody);
            if (!contacts)
            {
                /* TODO: An error occured - what to do here? */
            }
            if (contacts == Py_None)
            {
                /* No collision */
                Py_DECREF (contacts);
            }
            else
            {
                Py_ssize_t k, len;
                len = PyList_GET_SIZE (contacts);
                for (k = 0; k < len; k++)
                    PyList_Append (world->contactlist,
                        PyList_GET_ITEM (contacts, k));
            }
        }
    }

    contact_cnt = PyList_GET_SIZE (world->contactlist);
    for (j = 0; j < MAX_ITERATION; ++j)
    {
        /* clear bias */
        for (i = 0; i < body_cnt; ++i)
        {
            refbody = (PyBody*) PyList_GET_ITEM (world->bodylist, i);
            PyVector2_Set (refbody->bias_lv, 0.f, 0.f);
            refbody->bias_w = 0.f;
        }

        /* clear impulse */
        for (i = 0; i < contact_cnt; ++i)
        {
            contact = PyList_GET_ITEM (world->contactlist, i);
            PyVector2_Set (((PyContact*)contact)->acc_moment, 0, 0);
            PyVector2_Set (((PyContact*)contact)->split_acc_moment, 0, 0);
        }

        /* collision reaction */
        for (i = 0; i < contact_cnt; ++i)
        {
            contact = PyList_GET_ITEM (world->contactlist, i);
            PyContact_Collision_FAST ((PyContact*)contact, step);
        }

        /* update V */
        for (i = 0; i < contact_cnt; ++i)
        {
            contact = PyList_GET_ITEM (world->contactlist, i);
            PyJoint_SolveConstraints_FAST ((PyJoint*)contact, step);
        }
    }
    return 1;
}

/**
 * Performs any necessary position correction for the attached bodies.
 *
 * @param world The PyWorld to update the bodies for.
 * @param stepTime The time passed since the last update.
 */
static void
_correct_positions (PyWorld *world, double step)
{
    Py_ssize_t size = PyList_Size (world->bodylist);
    Py_ssize_t i;
    PyBody* body;

    for (i = 0; i < size; ++i)
    {
        body = (PyBody*) PyList_GET_ITEM (world->bodylist, i);
        PyBody_CorrectPosition (body, step); 
    }
}

/**
 * Updates all joints attached to the world.
 *
 * @param world The PyWorld to update the joints for.
 * @param stepTime The time passed since the last update.
 */
static void
_solve_joints (PyWorld *world, double step)
{
    Py_ssize_t i;
    Py_ssize_t size = PyList_Size (world->jointlist);
    PyJoint* joint;

    for (i = 0; i < size; ++i)
    {
        joint = (PyJoint*) PyList_GET_ITEM (world->jointlist, i);
        PyJoint_SolveConstraints_FAST (joint, step);
        /* what happened here? */
/*         if (joint->SolveConstraints) */
/*             PyJoint_SolveConstraints (joint, step); */
    }
}

/**
 * Updates the positions of the bodies attached to the world.
 *
 * @param world The PyWorld to update the body positions for.
 * @param stepTime The time passed since the last update.
 */
static void
_update_positions (PyWorld *world, double step)
{
    Py_ssize_t size = PyList_Size (world->bodylist);
    Py_ssize_t i;
    PyBody* body;

    for (i = 0; i < size; ++i)
    {
        body = (PyBody*) PyList_GET_ITEM (world->bodylist, i);
        PyBody_UpdatePosition (body, step);      
    }
}

static void
_world_dealloc (PyWorld *world)
{
    Py_XDECREF (world->dict);
    Py_XDECREF (world->bodylist);
    Py_XDECREF (world->jointlist);
    Py_XDECREF (world->contactlist);
    ((PyObject*)world)->ob_type->tp_free ((PyObject *) world);
}

static PyObject*
_world_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyWorld *world = (PyWorld*) type->tp_alloc (type, 0);
    if (!world)
        return NULL;

    world->bodylist = PyList_New (0);
    if (!world->bodylist)
    {
        Py_DECREF (world);
        return NULL;
    }

    world->jointlist = PyList_New (0);
    if (!world->jointlist)
    {
        Py_DECREF (world);
        return NULL;
    }
    
    world->contactlist = PyList_New (0);
    if (!world->contactlist)
    {
        Py_DECREF (world);
        return NULL;
    }

    world->dict = NULL;
    world->damping = 0.0;
    world->totaltime = 0.0;
    world->steptime = 0.1;
    AABBox_Reset (&(world->area));
    PyVector2_Set (world->gravity,0.0,-10);
    return (PyObject*) world;
}

static int
_world_init (PyWorld *world, PyObject *args, PyObject *kwds)
{
    return 0;
}

/* Getters/Setters */
static PyObject*
_world_getdict (PyWorld *world, void *closure)
{
    if (!world->dict)
    {
        world->dict = PyDict_New ();
        if (!world->dict)
            return NULL;
    }
    Py_INCREF (world->dict);
    return world->dict;
}

static PyObject*
_world_getdamping (PyWorld *world, void *closure)
{
    return PyFloat_FromDouble (world->damping);
}

static int
_world_setdamping (PyWorld *world, PyObject *value, void *closure)
{
    double tmp;
    if (!DoubleFromObj (value, &tmp))
        return -1;
    world->damping = tmp;
    return 0;
}

static PyObject*
_world_getgravity (PyWorld *world, void *closure)
{
    return Py_BuildValue ("(ff)", world->gravity.real, world->gravity.imag);
}

static int
_world_setgravity (PyWorld *world, PyObject *value, void *closure)
{
    double real, imag;
    
    if (!PySequence_Check(value) || PySequence_Size (value) != 2)
    {
        PyErr_SetString (PyExc_TypeError, "gravity must be a x, y sequence");
        return -1;
    }
    
    if (!DoubleFromSeqIndex (value, 0, &real))
        return -1;
    if (!DoubleFromSeqIndex (value, 1, &imag))
        return -1;

    world->gravity.real = real;
    world->gravity.imag = imag;
    return 0;

}

static PyObject*
_world_getbodies (PyWorld *world, void *closure)
{
    /* Return a copy of the list, so the user cannot manipulate the bodies
     * directly. */
    return PySequence_List (world->bodylist);

}

static PyObject*
_world_getjoints (PyWorld *world, void *closure)
{
    /* Return a copy of the list, so the user cannot manipulate the joints
     * directly. */
    return PySequence_List (world->jointlist);
}

/* Methods */
static PyObject*
_world_update (PyWorld* world, PyObject* args)
{
    double dt = 0.1;

    if (!PyArg_ParseTuple (args, "|d:update", &dt))
        return NULL;
    if (dt < 0)
    {
        PyErr_SetString (PyExc_ValueError, "step time must not be negative");
        return NULL;
    }

    if (!PyWorld_Update_FAST (world, dt))
        return NULL;
    Py_RETURN_NONE;
}

static PyObject*
_world_addbody (PyWorld* world, PyObject* args)
{
    PyObject *body;
    
    if (!PyArg_ParseTuple (args, "O:add_body", &body))
        return NULL;
    if (!PyWorld_AddBody ((PyObject*)world, body))
        return NULL;
    Py_RETURN_NONE;
}

static PyObject*
_world_removebody (PyWorld* world, PyObject* args)
{
    PyObject *body;
    if (!PyArg_ParseTuple (args, "O:remove_body", &body))
        return NULL;
    if (!PyWorld_RemoveBody ((PyObject*)world, body))
        return NULL;
    Py_RETURN_NONE;
}

static PyObject*
_world_addjoint (PyWorld* world, PyObject* args)
{
    PyObject *joint;
    if (!PyArg_ParseTuple (args, "O:add_joint", &joint))
        return NULL;
    if (!PyWorld_AddJoint ((PyObject*)world, joint))
        return NULL;
    Py_RETURN_NONE;
}

static PyObject*
_world_removejoint (PyWorld* world, PyObject* args)
{
    PyObject *joint;
    if (!PyArg_ParseTuple (args, "O:remove_joint", &joint))
        return NULL;
    if (!PyWorld_RemoveJoint ((PyObject*)world, joint))
        return NULL;
    Py_RETURN_NONE;
}


/* C API */
PyObject*
PyWorld_New (void)
{
    PyWorld *world = (PyWorld*) PyWorld_Type.tp_new (&PyWorld_Type, NULL, NULL);
    if (!world)
        return NULL;

    world->dict = NULL;

    world->bodylist = PyList_New (0);
    if (!world->bodylist)
    {
        Py_DECREF (world);
        return NULL;
    }

    world->jointlist = PyList_New (0);
    if (!world->jointlist)
    {
        Py_DECREF (world);
        return NULL;
    }
    
    world->contactlist = PyList_New (0);
    if (!world->contactlist)
    {
        Py_DECREF (world);
        return NULL;
    }

    AABBox_Reset (&(world->area));
    world->damping = 0.0;
    world->totaltime = 0.0;
    world->steptime = 0.1;
    PyVector2_Set (world->gravity,0.0,-10);
    return (PyObject*) world;
}

int
PyWorld_AddBody (PyObject *world, PyObject *body)
{
    int contained;

    if (!PyWorld_Check (world))
    {
        PyErr_SetString (PyExc_TypeError, "world must be a World");
        return 0;
    }
 
    if (!PyBody_Check (body))
    {
        PyErr_SetString (PyExc_TypeError, "body must be a Body");
        return 0;
    }
    
    contained = PySequence_Contains (((PyWorld*)world)->bodylist, body);
    if (contained == 1)
    {
        PyErr_SetString (PyExc_ValueError, "body already in world");
        return 0;
    }
    else if (contained == -1)
        return 0; /* Error set by the sequence */

    if (PyList_Append (((PyWorld*)world)->bodylist, body) == 0)
        return 1;
    return 0;
}

int
PyWorld_RemoveBody (PyObject *world, PyObject *body)
{
    Py_ssize_t _index;

    if (!PyWorld_Check (world))
    {
        PyErr_SetString (PyExc_TypeError, "world must be a World");
        return 0;
    }
 
    if (!PyBody_Check (body))
    {
        PyErr_SetString (PyExc_TypeError, "body must be a Body");
        return 0;
    }

    _index = PySequence_Index (((PyWorld*)world)->bodylist, body);
    if (_index != -1)
    {
        if (PySequence_DelItem (((PyWorld*)world)->bodylist, _index) == -1)
            return 0; /* Could not delete */
        return 1;
    }

    /* Not in list. */
    return 0;
}

int
PyWorld_AddJoint (PyObject *world, PyObject *joint)
{
    int contained;

    if (!PyWorld_Check (world))
    {
        PyErr_SetString (PyExc_TypeError, "world must be a World");
        return 0;
    }
 
    if (!PyJoint_Check (joint))
    {
        PyErr_SetString (PyExc_TypeError, "joint must be a Joint");
        return 0;
    }
    
    contained = PySequence_Contains (((PyWorld*)world)->jointlist, joint);
    if (contained == 1)
    {
        PyErr_SetString (PyExc_ValueError, "joint already in world");
        return 0;
    }
    else if (contained == -1)
        return 0; /* Error set by the sequence */

    if (PyList_Append (((PyWorld*)world)->jointlist, joint) == 0)
        return 1;
    return 0;
}

int
PyWorld_RemoveJoint (PyObject *world, PyObject *joint)
{
    Py_ssize_t _index;

    if (!PyWorld_Check (world))
    {
        PyErr_SetString (PyExc_TypeError, "world must be a World");
        return 0;
    }
 
    if (!PyJoint_Check (joint))
    {
        PyErr_SetString (PyExc_TypeError, "joint must be a Joint");
        return 0;
    }

    _index = PySequence_Index (((PyWorld*)world)->jointlist, joint);
    if (_index != -1)
    {
        if (PySequence_DelItem (((PyWorld*)world)->jointlist, _index) == -1)
            return 0; /* Could not delete */
        return 1;
    }
    /* Not in list. */
    return 0;
}

int
PyWorld_Update (PyObject* world, double step)
{
    if (!PyWorld_Check (world))
    {
        PyErr_SetString (PyExc_TypeError, "world must be a World");
        return 0;
    }

    if (step < 0)
    {
        PyErr_SetString (PyExc_ValueError, "step time must not be negative");
        return 0;
    }

    return PyWorld_Update_FAST ((PyWorld*)world, step);
}

int
PyWorld_Update_FAST (PyWorld *world, double step)
{
    int i;

    _update_body_simulation ((PyWorld*) world, step);

    for (i = 0; i < MAX_ITERATION; ++i)
    {
        if (!_detect_collisions ((PyWorld*) world, step))
            return 0;
        _correct_positions ((PyWorld*) world, step);
        _solve_joints ((PyWorld*) world, step);
    }
    _update_positions ((PyWorld*) world, step);

    return 1;
}

void
world_export_capi (void **capi)
{
    capi[PHYSICS_WORLD_FIRSTSLOT] = &PyWorld_Type;
    capi[PHYSICS_WORLD_FIRSTSLOT + 1] = PyWorld_New;
    capi[PHYSICS_WORLD_FIRSTSLOT + 2] = PyWorld_AddBody;
    capi[PHYSICS_WORLD_FIRSTSLOT + 3] = PyWorld_RemoveBody;
    capi[PHYSICS_WORLD_FIRSTSLOT + 4] = PyWorld_AddJoint;
    capi[PHYSICS_WORLD_FIRSTSLOT + 5] = PyWorld_RemoveJoint;
    capi[PHYSICS_WORLD_FIRSTSLOT + 6] = PyWorld_Update;
    capi[PHYSICS_WORLD_FIRSTSLOT + 7] = PyWorld_Update_FAST;
}
