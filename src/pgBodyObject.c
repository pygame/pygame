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

#define PHYSICS_BODY_INTERNAL
#include "pgDeclare.h"
#include "pgVector2.h"
#include "pgHelpFunctions.h"
#include "pgBodyObject.h"

static void _BodyNewInternal (PyBodyObject* body);
static int _BodyInit (PyBodyObject *body, PyObject *args, PyObject *kwds);
static PyObject* _BodyNew(PyTypeObject *type, PyObject *args, PyObject *kwds);
static void _BodyDestroy(PyBodyObject* body);

static PyObject* _Body_getVelocity(PyBodyObject* body,void* closure);
static int _Body_setVelocity(PyBodyObject* body,PyObject* value,void* closure);
static PyObject* _Body_getAngularVel (PyBodyObject* body,void* closure);
static int _Body_setAngularVel(PyBodyObject* body,PyObject* value,void* closure);
static PyObject* _Body_getPosition(PyBodyObject* body,void* closure);
static int _Body_setPosition(PyBodyObject* body,PyObject* value,void* closure);
static PyObject* _Body_getForce(PyBodyObject* body,void* closure);
static int _Body_setForce(PyBodyObject* body,PyObject* value,void* closure);

static PyObject* _Body_getMass (PyBodyObject* body,void* closure);
static int _Body_setMass(PyBodyObject* body,PyObject* value,void* closure);
static PyObject* _Body_getRotation (PyBodyObject* body,void* closure);
static int _Body_setRotation (PyBodyObject* body,PyObject* value,void* closure);
static PyObject* _Body_getTorque (PyBodyObject* body,void* closure);
static int _Body_setTorque (PyBodyObject* body,PyObject* value,void* closure);
static PyObject* _Body_getRestitution (PyBodyObject* body,void* closure);
static int _Body_setRestitution (PyBodyObject* body,PyObject* value,
    void* closure);
static PyObject* _Body_getFriction (PyBodyObject* body,void* closure);
static int _Body_setFriction (PyBodyObject* body,PyObject* value,void* closure);
static PyObject* _Body_getBStatic (PyBodyObject* body,void* closure);
static int _Body_setBStatic (PyBodyObject* body,PyObject* value,void* closure);
static PyObject* _Body_getShape(PyBodyObject* body,void* closure);
static PyObject *_Body_getPointList(PyObject *self, PyObject *args);

/* C API */
static PyObject* PyBody_New(PyObject *shape);
static PyVector2 PyBody_GetGlobalPos (PyObject *body, PyVector2 point);

/**
 * Methods used by the Body object.
 */
static PyMethodDef _Body_methods[] = {
    { "get_points",_Body_getPointList,METH_VARARGS,"" },
    { NULL, NULL, 0, NULL }   /* Sentinel */
};

/**
 * Getter/Setter definitions used by the Body
 */
static PyGetSetDef _Body_getseters[] = {
    { "mass", (getter) _Body_getMass, (setter) _Body_setMass,
      "The mass of the Body.", NULL },
    { "shape",(getter)_Body_getShape, NULL,
      "The shape of the Body. (Read-only)", NULL},
    { "rotation", (getter) _Body_getRotation, (setter) _Body_setRotation,
      "Rotation", NULL },
    { "torque", (getter) _Body_getTorque, (setter) _Body_setTorque,
      "Torque", NULL },
    { "restitution", (getter) _Body_getRestitution,
      (setter) _Body_setRestitution, "Restitution", NULL },
    { "friction", (getter) _Body_getFriction, (setter) _Body_setFriction,
      "Friction", NULL },
    { "velocity",(getter)_Body_getVelocity,(setter)_Body_setVelocity,
      "velocity",NULL },
    { "angular_velocity",(getter)_Body_getAngularVel,
      (setter)_Body_setAngularVel,"Angular Velocity",NULL },
    { "position",(getter)_Body_getPosition,(setter)_Body_setPosition,
      "position",NULL },
    { "force",(getter)_Body_getForce,(setter)_Body_setForce,"force",NULL },
    { "static",(getter)_Body_getBStatic,(setter)_Body_setBStatic,
      "whether static",NULL },
    { NULL, NULL, NULL, NULL, NULL}
};

PyTypeObject PyBody_Type =
{
    PyObject_HEAD_INIT(NULL)
    0,
    "physics.Body",             /* tp_name */
    sizeof(PyBodyObject),       /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor)_BodyDestroy,   /* tp_dealloc */
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
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "",                         /* tp_doc */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _Body_methods,              /* tp_methods */
    0,                          /* tp_members */
    _Body_getseters,            /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc)_BodyInit,                  /* tp_init */
    0,                          /* tp_alloc */
    _BodyNew,                   /* tp_new */
    0,                          /* tp_free */
    0,                          /* tp_is_gc */
    0,                          /* tp_bases */
    0,                          /* tp_mro */
    0,                          /* tp_cache */
    0,                          /* tp_subclasses */
    0,                          /* tp_weaklist */
    0                           /* tp_del */
};

/**
 * Internal body initialization. Fill the fields of the Body structure
 * with the default values.
 *
 * @param body The PyBodyObject to initialize.
 */
static void _BodyNewInternal(PyBodyObject* body)
{
    body->fAngleVelocity = 0.0;
    body->fFriction = 0.0;
    body->fMass = 1.0;
    body->fRestitution = 1.0;
    body->fRotation = 0.0;
    body->fTorque = 0.0;
    body->shape = NULL;
    body->bStatic = 0;
    body->fLinearVelDamping = 0.0;
    body->fAngleVelDamping = 0.06;
    PyVector2_Set(body->vecForce,0.0,0.0);
    PyVector2_Set(body->vecImpulse,0.0,0.0);
    PyVector2_Set(body->vecLinearVelocity,0.0,0.0);
    PyVector2_Set(body->vecPosition,0.0,0.0);
}

/**
 * Creates a new PyBodyObject.
 */
static PyObject* _BodyNew(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyObject* op = type->tp_alloc(type, 0);
    if (!op)
        return NULL;

    _BodyNewInternal((PyBodyObject*)op);
    return op;
}

static int _BodyInit (PyBodyObject *body, PyObject *args, PyObject *kwds)
{
    PyObject *shape;

    if (!PyArg_ParseTuple (args, "O", &shape))
        return -1;
    if (!PyShape_Check (shape))
    {
        PyErr_SetString (PyExc_TypeError, "shape must be a Shape");
        return -1;
    }
    Py_INCREF (shape);
    body->shape = shape;

    // I = M(a^2 + b^2)/12
    // TODO:
    // This should be automatically be done by the shape.
    if (((PyShapeObject*)shape)->type == ST_RECT)
    {
        PyRectShapeObject* rsh = (PyRectShapeObject*) shape;
        double width = ABS (rsh->bottomright.real - rsh->bottomleft.real);
        double height = ABS (rsh->bottomright.imag - rsh->topright.imag);
        ((PyShapeObject*)shape)->rInertia = body->fMass *
            (width * width + height * height) / 12;
    }

    return 0;
}

/**
 * Deallocates the passed PyBodyObject.
 *
 * @param body The PyBodyObject to deallocate.
 */
static void _BodyDestroy(PyBodyObject* body)
{
    /*
     * DECREF anything related to the Body, such as the lists and
     * release any other memory hold by it.
     */

    //delete shape
    Py_XDECREF(body->shape);
    body->ob_type->tp_free((PyObject*)body);
}


//============================================================
//getter and setter functions

/**
 * Getter for Body.velocity
 */
static PyObject* _Body_getVelocity(PyBodyObject* body,void* closure)
{
    return Py_BuildValue ("(ff)", body->vecLinearVelocity.real,
        body->vecLinearVelocity.imag);
}

/**
 * Setter for Body.velocity = (x, y)
 */
static int _Body_setVelocity(PyBodyObject* body,PyObject* value,void* closure)
{
    PyObject *item;
    double real, imag;

    if (!PySequence_Check(value) || PySequence_Size (value) != 2)
    {
        PyErr_SetString (PyExc_TypeError, "velocity must be a x, y sequence");
        return -1;
    }

    item = PySequence_GetItem (value, 0);
    if (!DoubleFromObj (item, &real))
        return -1;
    item = PySequence_GetItem (value, 1);
    if (!DoubleFromObj (item, &imag))
        return -1;
    
    body->vecLinearVelocity.real = real;
    body->vecLinearVelocity.imag = imag;
    return 0;
}

/**
 * Getter for Body.position
 */
static PyObject* _Body_getPosition(PyBodyObject* body,void* closure)
{
    return Py_BuildValue ("(ff)", body->vecPosition.real,
        body->vecPosition.imag);
}

/**
 * Setter for Body.position = (x, y)
 */
static int _Body_setPosition(PyBodyObject* body,PyObject* value,void* closure)
{
    PyObject *item;
    double real, imag;

    if (!PySequence_Check(value) || PySequence_Size (value) != 2)
    {
        PyErr_SetString (PyExc_TypeError, "position must be a x, y sequence");
        return -1;
    }

    item = PySequence_GetItem (value, 0);
    if (!DoubleFromObj (item, &real))
        return -1;
    item = PySequence_GetItem (value, 1);
    if (!DoubleFromObj (item, &imag))
        return -1;
    
    body->vecPosition.real = real;
    body->vecPosition.imag = imag;
    return 0;
}

/**
 * Getter for Body.force
 */
static PyObject* _Body_getForce(PyBodyObject* body,void* closure)
{
    return Py_BuildValue ("(ff)", body->vecForce.real, body->vecForce.imag);
}

/**
 * Setter for Body.force = (x, y)
 */
static int _Body_setForce(PyBodyObject* body,PyObject* value,void* closure)
{
    PyObject *item;
    double real, imag;

    if (!PySequence_Check(value) || PySequence_Size (value) != 2)
    {
        PyErr_SetString (PyExc_TypeError, "force must be a x, y sequence");
        return -1;
    }

    item = PySequence_GetItem (value, 0);
    if (!DoubleFromObj (item, &real))
    {
        Py_DECREF (item);
        return -1;
    }
    Py_DECREF (item);
    item = PySequence_GetItem (value, 1);
    if (!DoubleFromObj (item, &imag))
    {
        Py_DECREF (item);
        return -1;
    }
    Py_DECREF (item);

    body->vecForce.real = real;
    body->vecForce.imag = imag;
    return 0;
}

/**
 * Getter for Body.mass
 */
static PyObject* _Body_getMass (PyBodyObject* body,void* closure)
{
    return PyFloat_FromDouble (body->fMass);
}

/**
 * Setter for Body.mass = x
 */
static int _Body_setMass(PyBodyObject* body,PyObject* value,void* closure)
{
    if (PyNumber_Check (value))
    {
        PyObject *tmp = PyNumber_Float (value);

        if (tmp)
        {
            double mass = PyFloat_AsDouble (tmp);
            Py_DECREF (tmp);
            if (PyErr_Occurred ())
                return -1;
            if (mass < 0)
            {
                PyErr_SetString(PyExc_ValueError, "mass must not be negative");
                return -1;
            }
            body->fMass = mass;
            return 0;
        }
    }
    PyErr_SetString (PyExc_TypeError, "mass must be a float");

    // I = M(a^2 + b^2)/12
    // TODO:
    // This should be automatically be done by the shape.
    if (((PyShapeObject*)body->shape)->type == ST_RECT)
    {
        PyRectShapeObject* rsh = (PyRectShapeObject*) body->shape;
        double width = ABS (rsh->bottomright.real - rsh->bottomleft.real);
        double height = ABS (rsh->bottomright.imag - rsh->topright.imag);
        ((PyShapeObject*)body->shape)->rInertia = body->fMass *
            (width * width + height * height) / 12;
    }

    return -1;
}

/**
 * Getter for Body.rotation.
 */
static PyObject* _Body_getRotation (PyBodyObject* body,void* closure)
{
    return PyFloat_FromDouble (RAD2DEG(body->fRotation));
}

/**
 * Setter for Body.rotation = x
 */
static int _Body_setRotation(PyBodyObject* body,PyObject* value,void* closure)
{
    if (PyNumber_Check (value))
    {
        PyObject *tmp = PyNumber_Float (value);

        if (tmp)
        {
            double rotation = PyFloat_AsDouble (tmp);
            Py_DECREF (tmp);
            if (PyErr_Occurred ())
                return -1;
            body->fRotation = DEG2RAD(rotation);
            return 0;
        }
    }
    PyErr_SetString (PyExc_TypeError, "rotation must be a float");
    return -1;
}

/**
* Getter for Body.angular_velocity
*/
static PyObject* _Body_getAngularVel (PyBodyObject* body,void* closure)
{
    return PyFloat_FromDouble (RAD2DEG(body->fAngleVelocity));
}

/**
 * Setter for Body.angular_velocity = x
 */
static int _Body_setAngularVel(PyBodyObject* body,PyObject* value,void* closure)
{
    if (PyNumber_Check (value))
    {
        PyObject *tmp = PyNumber_Float (value);

        if (tmp)
        {
            double fAngleVelocity = PyFloat_AsDouble (tmp);
            Py_DECREF (tmp);
            if (PyErr_Occurred ())
                return -1;
            body->fAngleVelocity = DEG2RAD(fAngleVelocity);
            return 0;
        }
    }
    PyErr_SetString (PyExc_TypeError, "rotation must be a float");
    return -1;
}

/**
 * Getter for Body.torque
 */
static PyObject* _Body_getTorque (PyBodyObject* body,void* closure)
{
    return PyFloat_FromDouble (body->fTorque);
}

/**
 * Setter for Body.torque = x
 */
static int _Body_setTorque (PyBodyObject* body,PyObject* value,void* closure)
{
    if (PyNumber_Check (value))
    {
        PyObject *tmp = PyNumber_Float (value);

        if (tmp)
        {
            double torque = PyFloat_AsDouble (tmp);
            Py_DECREF (tmp);
            if (PyErr_Occurred ())
                return -1;
            body->fTorque = torque;
            return 0;
        }
    }
    PyErr_SetString (PyExc_TypeError, "torque must be a float");
    return -1;
}

/**
 * Getter for Body.restitution
 */
static PyObject* _Body_getRestitution (PyBodyObject* body,void* closure)
{
    return PyFloat_FromDouble (body->fRestitution);
}

/**
 * Setter for Body.restitution = x
 */
static int _Body_setRestitution (PyBodyObject* body,PyObject* value,
    void* closure)
{
    if (PyNumber_Check (value))
    {
        PyObject *tmp = PyNumber_Float (value);

        if (tmp)
        {
            double rest = PyFloat_AsDouble (tmp);
            Py_DECREF (tmp);
            if (PyErr_Occurred ())
                return -1;
            if (rest < 0 || rest > 1)
            {
                PyErr_SetString(PyExc_ValueError,
                    "restitution must be in the range [0,1]");
                return -1;
            }
            body->fRestitution = rest;
            return 0;
        }
    }
    PyErr_SetString (PyExc_TypeError, "restitution must be a float");
    return -1;
}

/**
 * Getter for Body.friction
 */
static PyObject* _Body_getFriction (PyBodyObject* body,void* closure)
{
    return PyFloat_FromDouble (body->fFriction);
}

/**
 * Setter for Body.friction = x
 */
static int _Body_setFriction (PyBodyObject* body,PyObject* value,void* closure)
{
    if (PyNumber_Check (value))
    {
        PyObject *tmp = PyNumber_Float (value);

        if (tmp)
        {
            double friction = PyFloat_AsDouble (tmp);
            Py_DECREF (tmp);
            if (PyErr_Occurred ())
                return -1;
            if (friction < 0)
            {
                PyErr_SetString(PyExc_ValueError,
                    "friction must not be negative");
                return -1;
            }
            body->fFriction = friction;
            return 0;
        }
    }
    PyErr_SetString (PyExc_TypeError, "friction must be a float");
    return -1;
}

/**
 * Setter for Body.static = x
 */
static PyObject* _Body_getBStatic (PyBodyObject* body,void* closure)
{
    if (body->bStatic)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

/**
 * Getter for Body.static
 */
static int _Body_setBStatic (PyBodyObject* body,PyObject* value,void* closure)
{
    if (PyBool_Check (value))
    {
        body->bStatic = (value == Py_True) ? 1 : 0;
        return 0;

    }
    PyErr_SetString (PyExc_TypeError, "static must be a bool");
    return -1;
}

/**
 * Getter for Body.shape
 */
static PyObject* _Body_getShape (PyBodyObject* body,void* closure)
{
    if (!body->shape)
        Py_RETURN_NONE;

    Py_INCREF (body->shape);
    return body->shape;
}


/* Body methods */

/**
 * Body.get_points ()
 */
static PyObject *_Body_getPointList(PyObject *self, PyObject *args)
{
    PyBodyObject* body = (PyBodyObject*)self;
    PyObject* list;
    PyVector2* pVertex;
    PyVector2 golVertex;
    PyObject* tuple;

    if (!body->shape)
    {
        Py_RETURN_NONE;
    }

    /* TODO: shapes */
    list = PyList_New (4);

    pVertex = &(((PyRectShapeObject*)(body->shape))->bottomleft);
    golVertex = PyBodyObject_GetGlobalPos(body,pVertex);
    tuple = FromPhysicsVector2ToPoint(golVertex);
    PyList_SetItem(list,0,tuple);

    pVertex = &(((PyRectShapeObject*)(body->shape))->bottomright);
    golVertex = PyBodyObject_GetGlobalPos(body,pVertex);
    tuple = FromPhysicsVector2ToPoint(golVertex);
    PyList_SetItem(list,1,tuple);

    pVertex = &(((PyRectShapeObject*)(body->shape))->topright);
    golVertex = PyBodyObject_GetGlobalPos(body,pVertex);
    tuple = FromPhysicsVector2ToPoint(golVertex);
    PyList_SetItem(list,2,tuple);

    pVertex = &(((PyRectShapeObject*)(body->shape))->topleft);
    golVertex = PyBodyObject_GetGlobalPos(body,pVertex);
    tuple = FromPhysicsVector2ToPoint(golVertex);
    PyList_SetItem(list,3,tuple);

    return list;
}

/* Internally used Body functions */

void PyBodyObject_FreeUpdateVel(PyBodyObject* body, PyVector2 gravity,
    double dt)
{
    PyVector2 totalF;
    double k1,k2;
    
    if (body->bStatic)
        return;

    totalF = c_sum(body->vecForce,
        PyVector2_MultiplyWithReal(gravity, body->fMass));
    body->vecLinearVelocity = c_sum(body->vecLinearVelocity, 
        PyVector2_MultiplyWithReal(totalF, dt/body->fMass));
    k1 = PG_Clamp(1-dt*body->fLinearVelDamping,0.0,1.0);
    k2 = PG_Clamp(1-dt*body->fAngleVelDamping,0.0,1.0);
    body->vecLinearVelocity =
        PyVector2_MultiplyWithReal(body->vecLinearVelocity,k1);
    body->fAngleVelocity *= k2;
}

void PyBodyObject_FreeUpdatePos(PyBodyObject* body,double dt)
{
    if (body->bStatic)
        return;

    body->vecPosition = c_sum(body->vecPosition, 
        PyVector2_MultiplyWithReal(body->vecLinearVelocity, dt));
    body->fRotation += body->fAngleVelocity*dt;
}

void PyBodyObject_CorrectPos(PyBodyObject* body, double dt)
{
    if (body->bStatic)
        return;

    body->vecPosition = c_sum(body->vecPosition, 
        PyVector2_MultiplyWithReal(body->cBiasLV, dt));
    body->fRotation += body->cBiasW*dt;
}

PyVector2 PyBodyObject_GetGlobalPos(PyBodyObject* body, PyVector2* local_p)
{
    PyVector2 ans;
	
    ans = *local_p;
    PyVector2_Rotate(&ans, body->fRotation);
    ans = c_sum(ans, body->vecPosition);

    return ans;
}

PyVector2 PyBodyObject_GetRelativePosFromGlobal(PyBodyObject* body,
    PyVector2* global_p)
{
	PyVector2 ans = c_diff(*global_p,body->vecPosition);

	PyVector2_Rotate(&ans, -body->fRotation);
	return ans;
}

PyVector2 PyBodyObject_GetRelativePos(PyBodyObject* bodyA, PyBodyObject* bodyB,
    PyVector2* p_in_B)
{
    PyVector2 trans, p_in_A;
    double rotate;
	
    trans = c_diff(bodyB->vecPosition, bodyA->vecPosition);
    PyVector2_Rotate(&trans, -bodyA->fRotation);
    rotate = bodyA->fRotation - bodyB->fRotation;
    p_in_A = *p_in_B;
    PyVector2_Rotate(&p_in_A, -rotate);
    p_in_A = c_sum(p_in_A, trans);
	
    return p_in_A;
}

PyVector2 PyBodyObject_GetLocalPointVelocity(PyBodyObject* body,
    PyVector2 localPoint)
{
    PyVector2 vel = PyVector2_fCross(body->fAngleVelocity,localPoint);
    return c_sum(vel,body->vecLinearVelocity);
}

/* C API */
static PyObject* PyBody_New(PyObject *shape)
{
    PyBodyObject* body;
    
    if (!PyShape_Check (shape))
    {
        PyErr_SetString (PyExc_TypeError, "shape must be Shape");
        return NULL;
    }

    body = (PyBodyObject*) _BodyNew(&PyBody_Type, NULL, NULL);
    body->shape = shape;

    // I = M(a^2 + b^2)/12
    // TODO:
    // This should be automatically be done by the shape.
    if (((PyShapeObject*)shape)->type == ST_RECT)
    {
        PyRectShapeObject* rsh = (PyRectShapeObject*) shape;
        double width = ABS (rsh->bottomright.real - rsh->bottomleft.real);
        double height = ABS (rsh->bottomright.imag - rsh->topright.imag);
        ((PyShapeObject*)shape)->rInertia = body->fMass *
            (width * width + height * height) / 12;
    }
    return (PyObject*)body;
    
}

static PyVector2 PyBody_GetGlobalPos (PyObject *body, PyVector2 point)
{
    return PyBodyObject_GetGlobalPos ((PyBodyObject*) body, &point);
}

void PyBodyObject_ExportCAPI (void **c_api)
{
    c_api[PHYSICS_BODY_FIRSTSLOT] = &PyBody_Type;
    c_api[PHYSICS_BODY_FIRSTSLOT + 1] = &PyBody_New;
    c_api[PHYSICS_BODY_FIRSTSLOT + 2] = &PyBody_GetGlobalPos;
}
