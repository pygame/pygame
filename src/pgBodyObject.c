#include "pgBodyObject.h"
#include "pgWorldObject.h"
#include "pgVector2.h"
#include "pgShapeObject.h"
#include "pgHelpFunctions.h"
#include <structmember.h>

extern PyTypeObject pgBodyType;

void PG_Bind_RectShape(pgBodyObject* body, double width, double height, double seta)
{
	if(body->shape == NULL)
		body->shape = PG_RectShapeNew(body, width, height, seta);
	else
	{
		Py_DECREF(body->shape);
		body->shape = PG_RectShapeNew(body, width, height, seta);
	}
}

void PG_FreeUpdateBodyVel(pgWorldObject* world,pgBodyObject* body, double dt)
{
	pgVector2 totalF;
	if(body->bStatic) return;

	totalF = c_sum(body->vecForce, c_mul_complex_with_real(world->vecGravity,
		body->fMass));
	body->vecLinearVelocity = c_sum(body->vecLinearVelocity, 
		c_mul_complex_with_real(totalF, dt/body->fMass));
}

//void PG_FreeUpdateBodyPos(pgWorldObject* world,pgBodyObject* body,double dt)
//{
//	pgVector2 v;
//	double w;
//
//	if(body->bStatic) return;
//	
//	v = c_sum(body->vecLinearVelocity, body->cBiasLV);
//	w = body->fAngleVelocity + body->cBiasW;
//	body->vecPosition = c_sum(body->vecPosition, 
//		c_mul_complex_with_real(v, dt));
//	body->fRotation += w*dt;
//}

void PG_FreeUpdateBodyPos(pgBodyObject* body,double dt)
{
	if(body->bStatic) return;

	body->vecPosition = c_sum(body->vecPosition, 
		c_mul_complex_with_real(body->vecLinearVelocity, dt));
	body->fRotation += body->fAngleVelocity*dt;
}

void PG_CorrectBodyPos(pgBodyObject* body, double dt)
{
	if(body->bStatic) return;

	body->vecPosition = c_sum(body->vecPosition, 
		c_mul_complex_with_real(body->cBiasLV, dt));
	body->fRotation += body->cBiasW*dt;
}

//void PG_FreeUpdateBodyPos(pgWorldObject* world,pgBodyObject* body,double dt)
//{
//	pgVector2 v;
//	double w;
//
//	if(body->bStatic) return;
//
//	v = c_sum(body->vecLinearVelocity, body->cBiasLV);
//	w = body->fAngleVelocity + body->cBiasW;
//	body->vecPosition = c_sum(body->vecPosition, 
//		c_mul_complex_with_real(v, dt));
//	body->fRotation += w*dt;
//}

void PG_BodyInit(pgBodyObject* body)
{
	body->fAngleVelocity = 0.0;
	body->fFriction = 0.0;
	body->fMass = 1.0;
	body->fRestitution = 1.0;
	body->fRotation = 0.0;
	body->fTorque = 0.0;
	body->shape = NULL;
	body->bStatic = 0;
	PG_Set_Vector2(body->vecForce,0.0,0.0);
	PG_Set_Vector2(body->vecImpulse,0.0,0.0);
	PG_Set_Vector2(body->vecLinearVelocity,0.0,0.0);
	PG_Set_Vector2(body->vecPosition,0.0,0.0);
}

PyObject* _PG_BodyNew(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	//TODO: parse args later on
	pgBodyObject* op;
	if(PyType_Ready(type)==-1) return NULL;
	//op = (pgBodyObject*)type->tp_alloc(type, 0);
	op = PyObject_New(pgBodyObject,type);
	Py_INCREF(op);
	PG_BodyInit(op);
	return (PyObject*)op;
}

void PG_BodyDestroy(pgBodyObject* body)
{
	/*
	* DECREF anything related to the Body, such as the lists and
	* release any other memory hold by it.
	*/

	//delete shape
	Py_DECREF(body->shape);
	body->ob_type->tp_free((PyObject*)body);
}



pgBodyObject* PG_BodyNew()
{
	return (pgBodyObject*) _PG_BodyNew(&pgBodyType, NULL, NULL);
}

pgVector2 PG_GetGlobalPos(pgBodyObject* body, pgVector2* local_p)
{
	pgVector2 ans;
	
	ans = *local_p;
	c_rotate(&ans, body->fRotation);
	ans = c_sum(ans, body->vecPosition);

	return ans;
}


pgVector2 PG_GetRelativePos(pgBodyObject* bodyA, pgBodyObject* bodyB, pgVector2* p_in_B)
{
	pgVector2 trans, p_in_A;
	double rotate;
	
	trans = c_diff(bodyB->vecPosition, bodyA->vecPosition);
	c_rotate(&trans, -bodyA->fRotation);
	rotate = bodyA->fRotation - bodyB->fRotation;
	p_in_A = *p_in_B;
	c_rotate(&p_in_A, -rotate);
	p_in_A = c_sum(p_in_A, trans);
	
	return p_in_A;
}

pgVector2 PG_GetLocalPointVelocity(pgBodyObject* body,pgVector2 localPoint)
{
	pgVector2 vel = c_fcross(body->fAngleVelocity,localPoint);
	return c_sum(vel,body->vecLinearVelocity);
}

//============================================================
//getter and setter functions

//velocity
static PyObject* _pgBody_getVelocity(pgBodyObject* body,void* closure)
{
    return Py_BuildValue ("(ff)", body->vecLinearVelocity.real,
        body->vecLinearVelocity.imag);
}

static int _pgBody_setVelocity(pgBodyObject* body,PyObject* value,void* closure)
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

//position
static PyObject* _pgBody_getPosition(pgBodyObject* body,void* closure)
{
    return Py_BuildValue ("(ff)", body->vecPosition.real,
        body->vecPosition.imag);
}

static int _pgBody_setPosition(pgBodyObject* body,PyObject* value,void* closure)
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

//force
static PyObject* _pgBody_getForce(pgBodyObject* body,void* closure)
{
    return Py_BuildValue ("(ff)", body->vecForce.real, body->vecForce.imag);
}


static int _pgBody_setForce(pgBodyObject* body,PyObject* value,void* closure)
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
        return -1;
    item = PySequence_GetItem (value, 1);
    if (!DoubleFromObj (item, &imag))
        return -1;
    
    body->vecForce.real = real;
    body->vecForce.imag = imag;
    return 0;
}

/**
 * Getter for retrieving the mass of the passed body.
 */
static PyObject* _pgBody_getMass (pgBodyObject* body,void* closure)
{
    return PyFloat_FromDouble (body->fMass);
}

/**
 * Sets the mass of the passed body.
 */
static int _pgBody_setMass(pgBodyObject* body,PyObject* value,void* closure)
{
    if (PyNumber_Check (value))
    {
        PyObject *tmp = PyNumber_Float (value);

        if (tmp)
        {
            double mass = PyFloat_AsDouble (tmp);
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
    return -1;
}

/**
 * Getter for retrieving the rotation of the passed body.
 */
static PyObject* _pgBody_getRotation (pgBodyObject* body,void* closure)
{
    return PyFloat_FromDouble (body->fRotation);
}

/**
 * Sets the rotation of the passed body.
 */
static int _pgBody_setRotation(pgBodyObject* body,PyObject* value,void* closure)
{
    if (PyNumber_Check (value))
    {
        PyObject *tmp = PyNumber_Float (value);

        if (tmp)
        {
            double rotation = PyFloat_AsDouble (tmp);
            if (PyErr_Occurred ())
                return -1;
            body->fRotation = rotation;
            return 0;
        }
    }
    PyErr_SetString (PyExc_TypeError, "rotation must be a float");
    return -1;
}

/**
 * Getter for retrieving the torque of the passed body.
 */
static PyObject* _pgBody_getTorque (pgBodyObject* body,void* closure)
{
    return PyFloat_FromDouble (body->fTorque);
}

/**
 * Sets the torque of the passed body.
 */
static int _pgBody_setTorque (pgBodyObject* body,PyObject* value,void* closure)
{
    if (PyNumber_Check (value))
    {
        PyObject *tmp = PyNumber_Float (value);

        if (tmp)
        {
            double torque = PyFloat_AsDouble (tmp);
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
 * Getter for retrieving the restitution of the passed body.
 */
static PyObject* _pgBody_getRestitution (pgBodyObject* body,void* closure)
{
    return PyFloat_FromDouble (body->fRestitution);
}

/**
 * Sets the restitution of the passed body.
 */
static int _pgBody_setRestitution (pgBodyObject* body,PyObject* value,void* closure)
{
    if (PyNumber_Check (value))
    {
        PyObject *tmp = PyNumber_Float (value);

        if (tmp)
        {
            double rest = PyFloat_AsDouble (tmp);
            if (PyErr_Occurred ())
                return -1;
            body->fRestitution = rest;
            return 0;
        }
    }
    PyErr_SetString (PyExc_TypeError, "torque must be a float");
    return -1;
}

/**
 * Getter for retrieving the friction of the passed body.
 */
static PyObject* _pgBody_getFriction (pgBodyObject* body,void* closure)
{
    return PyFloat_FromDouble (body->fFriction);
}

/**
 * Sets the friction of the passed body.
 */
static int _pgBody_setFriction (pgBodyObject* body,PyObject* value,void* closure)
{
    if (PyNumber_Check (value))
    {
        PyObject *tmp = PyNumber_Float (value);

        if (tmp)
        {
            double friction = PyFloat_AsDouble (tmp);
            if (PyErr_Occurred ())
                return -1;
            body->fFriction = friction;
            return 0;
        }
    }
    PyErr_SetString (PyExc_TypeError, "torque must be a float");
    return -1;
}

/**
* Getter for retrieving the bStatic of the passed body.
*/
static PyObject* _pgBody_getBStatic (pgBodyObject* body,void* closure)
{
	return PyInt_FromLong (body->bStatic);
}

/**
* Sets the bStatic of the passed body.
*/
static int _pgBody_setBStatic (pgBodyObject* body,PyObject* value,void* closure)
{
	if (PyInt_Check (value))
	{
		body->bStatic = PyInt_AsLong (value);
		return 0;

	}
	PyErr_SetString (PyExc_TypeError, "torque must be a float");
	return -1;
}

static PyObject* _pgBody_bindRectShape(PyObject* body,PyObject* args)
{
	double width,height,seta;
	if (!PyArg_ParseTuple(args,"ddd",&width,&height,&seta))
	{
		PyErr_SetString(PyExc_ValueError,"parameters are wrong");
		return NULL;
	}
	else
	{
		PG_Bind_RectShape((pgBodyObject*)body,width,height,seta);
		if (((pgBodyObject*)body)->shape == NULL)
		{
			PyErr_SetString(PyExc_ValueError,"shape binding is failed");
			return NULL;
		}
		else
		{
			Py_RETURN_NONE;
			//return ((pgBodyObject*)body)->shape;
		}
	}
}



static PyObject * _pg_getPointListFromBody(PyObject *self, PyObject *args)
{
	pgBodyObject* body = (pgBodyObject*)self;
	int i;
	PyObject* list;

	/*if (!PyArg_ParseTuple(args,"O",&body))
	{
		PyErr_SetString(PyExc_ValueError,"arg is not body type");
		return NULL;
	}
	else*/
	{
		if (body->shape == NULL)
		{
			PyErr_SetString(PyExc_ValueError,"Shape is NULL");
			return NULL;
		}
		list = PyList_New(4);
		for (i = 0;i < 4;i++)
		{
			pgVector2* pVertex = &(((pgRectShape*)(body->shape))->point[i]);
			pgVector2 golVertex = PG_GetGlobalPos(body,pVertex);
			PyObject* tuple = FromPhysicsVector2ToPoint(golVertex);
			PyList_SetItem(list,i,tuple);
		}
		return (PyObject*)list;
	}
}

//===============================================================

static PyMethodDef _pgBody_methods[] = {
	{"bind_rect_shape",_pgBody_bindRectShape,METH_VARARGS,""},
	{"get_point_list",_pg_getPointListFromBody,METH_VARARGS,""	},
    {NULL, NULL, 0, NULL}   /* Sentinel */
};

static PyGetSetDef _pgBody_getseters[] = {
    { "mass", (getter) _pgBody_getMass, (setter) _pgBody_setMass, "Mass",
      NULL },
    { "rotation", (getter) _pgBody_getRotation, (setter) _pgBody_setRotation,
      "Rotation", NULL },
    { "torque", (getter) _pgBody_getTorque, (setter) _pgBody_setTorque,
      "Torque", NULL },
    { "restitution", (getter) _pgBody_getRestitution,
      (setter) _pgBody_setRestitution, "Restitution", NULL },
    {"friction", (getter) _pgBody_getFriction, (setter) _pgBody_setFriction,
     "Friction", NULL },

    {"velocity",(getter)_pgBody_getVelocity,(setter)_pgBody_setVelocity,"velocity",NULL},
    {"position",(getter)_pgBody_getPosition,(setter)_pgBody_setPosition,"position",NULL},
    {"force",(getter)_pgBody_getForce,(setter)_pgBody_setForce,"force",NULL},
	{"static",(getter)_pgBody_getBStatic,(setter)_pgBody_setBStatic,"whether static",NULL},
    { NULL, NULL, NULL, NULL, NULL}
};

PyTypeObject pgBodyType =
{
	PyObject_HEAD_INIT(NULL)
	0,
	"physics.Body",            /* tp_name */
	sizeof(pgBodyObject),      /* tp_basicsize */
	0,                          /* tp_itemsize */
	(destructor)PG_BodyDestroy,/* tp_dealloc */
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
	_pgBody_methods,		   	/* tp_methods */
	0,                          /* tp_members */
	_pgBody_getseters,          /* tp_getset */
	0,                          /* tp_base */
	0,                          /* tp_dict */
	0,                          /* tp_descr_get */
	0,                          /* tp_descr_set */
	0,                          /* tp_dictoffset */
	0,                          /* tp_init */
	0,                          /* tp_alloc */
	_PG_BodyNew,                /* tp_new */
	0,                          /* tp_free */
	0,                          /* tp_is_gc */
	0,                          /* tp_bases */
	0,                          /* tp_mro */
	0,                          /* tp_cache */
	0,                          /* tp_subclasses */
	0,                          /* tp_weaklist */
	0                           /* tp_del */
};


