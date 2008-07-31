#include "pgWorldObject.h"
#include "pgBodyObject.h"
#include "pgJointObject.h"
#include "pgCollision.h"
#include "pgShapeObject.h"
#include <structmember.h>

#define MAX_ITERATION 5

extern PyTypeObject pgWorldType;

void _PG_FreeBodySimulation(pgWorldObject* world,double stepTime)
{
	Py_ssize_t i;
	Py_ssize_t size = PyList_Size((PyObject*)(world->bodyList));
	for (i = 0; i < size; ++i)
	{
		pgBodyObject* body = (pgBodyObject*)(PyList_GetItem(world->bodyList, i));		
		PG_FreeUpdateBodyVel(world, body, stepTime);
	}
}

void _PG_BodyCollisionDetection(pgWorldObject* world, double step)
{
	Py_ssize_t i, j, body_cnt, contact_cnt;
	pgBodyObject* refBody, *incBody;
	pgJointObject* contact;
	
	body_cnt = PyList_Size(world->bodyList);
	//clear contactList first
	contact_cnt = PyList_Size(world->contactList);
	if(PyList_SetSlice(world->contactList, 0, contact_cnt, NULL) < 0) return;
	assert(PyList_Size(world->contactList)==0);
	
	//for all pair of objects, do collision test	
	//update AABB
	for(i = 0; i < body_cnt; ++i)
	{
		refBody = (pgBodyObject*)(PyList_GetItem(world->bodyList, i));
		refBody->shape->UpdateAABB(refBody);
	}
	
	//collision test
	for(i = 0; i < body_cnt-1; ++i)
	{
		refBody = (pgBodyObject*)(PyList_GetItem(world->bodyList, i));
		for(j = i+1; j < body_cnt; ++j)
		{
			incBody = (pgBodyObject*)(PyList_GetItem(world->bodyList, j));
			if(refBody->bStatic && incBody->bStatic) continue;
			
			if(PG_IsOverlap(&(refBody->shape->box), &(incBody->shape->box), 1e-8))
			{
				
				PG_AppendContact(refBody, incBody, world->contactList);
			}
		}
	}

	contact_cnt = PyList_Size(world->contactList);
	if (contact_cnt)
	{
		printf("contact_cnt:%d\n",contact_cnt);
	}
	
	for(j = 0; j < MAX_ITERATION; ++j)
	{
		//clear bias
		for(i = 0; i < body_cnt; ++i)
		{
			refBody = (pgBodyObject*)(PyList_GetItem(world->bodyList, i));
			PG_Set_Vector2(refBody->cBiasLV, 0.f, 0.f);
			refBody->cBiasW = 0.f;
		}
		//clear impulse
		for(i = 0; i < contact_cnt; ++i)
		{
			contact = (pgJointObject*)(PyList_GetItem(world->contactList, i));
			PG_Set_Vector2(**(((pgContact*)contact)->ppAccMoment), 0, 0);
			PG_Set_Vector2(**(((pgContact*)contact)->ppSplitAccMoment), 0, 0);
		}

		//collision reaction
		contact_cnt = PyList_Size(world->contactList);
		for(i = 0; i < contact_cnt; ++i)
		{
			contact = (pgJointObject*)(PyList_GetItem(world->contactList, i));
			PG_ApplyContact((PyObject*)contact, step);
		}
		//update V	
		for(i = 0; i < contact_cnt; ++i)
		{
			contact = (pgJointObject*)(PyList_GetItem(world->contactList, i));
			contact->SolveConstraintVelocity(contact, step);
		}
	}
}

void _PG_JointSolve(pgWorldObject* world,double stepTime)
{
	Py_ssize_t size = PyList_Size((PyObject*)(world->jointList));
	Py_ssize_t i;
	for (i = 0; i < size; ++i)
	{
		pgJointObject* joint = (pgJointObject*)(PyList_GetItem(world->jointList,i));
		//what happened here?
		if (joint->SolveConstraintVelocity)
		{
			joint->SolveConstraintVelocity(joint,stepTime);
		}
		/*if (joint->SolveConstraintPosition)
		{
			joint->SolveConstraintPosition(joint,stepTime);
		}*/
	}
}

void _PG_BodyPositionUpdate(pgWorldObject* world,double stepTime)
{
	Py_ssize_t size = PyList_Size(world->bodyList);
	Py_ssize_t i;
	for (i = 0; i < size; ++i)
	{
		pgBodyObject* body = (pgBodyObject*)(PyList_GetItem(world->bodyList,i));
		PG_FreeUpdateBodyPos(body,stepTime);	
	}
}


//void PG_Update(pgWorldObject* world,double stepTime)
//{
//	int i;
//
//	_PG_FreeBodySimulation(world, stepTime);
//	_PG_BodyCollisionDetection(world, stepTime);
//	for(i = 0; i < MAX_ITERATION; ++i)
//		_PG_JointSolve(world, stepTime);
//	_PG_BodyPositionUpdate(world, stepTime);
//}

void _PG_BodyPositionCorrection(pgWorldObject* world,double stepTime)
{
	Py_ssize_t size = PyList_Size((PyObject*)(world->bodyList));
	Py_ssize_t i;
	for (i = 0; i < size; ++i)
	{
		pgBodyObject* body = (pgBodyObject*)(PyList_GetItem((PyObject*)(world->bodyList),i));
		PG_CorrectBodyPos(body,stepTime);	
	}
}

void PG_Update(pgWorldObject* world,double stepTime)
{
	int i;
	_PG_FreeBodySimulation(world, stepTime);
	_PG_BodyPositionUpdate(world, stepTime);
	for(i = 0; i < MAX_ITERATION; ++i)
	{
		_PG_BodyCollisionDetection(world, stepTime);
		_PG_BodyPositionCorrection(world,stepTime);
		_PG_JointSolve(world, stepTime);
		
	}
	//_PG_BodyPositionUpdate(world, stepTime);
	
}


int PG_AddBodyToWorld(pgWorldObject* world,pgBodyObject* body)
{
	return PyList_Append(world->bodyList,(PyObject*)body);
}



int PG_RemoveBodyFromWorld(pgWorldObject* world,pgBodyObject* body)
{
	return 0;
}

int PG_AddJointToWorld(pgWorldObject* world,pgJointObject* joint)
{
	return PyList_Append(world->jointList,(PyObject*)joint);
}

int PG_RemoveJointFromWorld(pgWorldObject* world,pgJointObject* joint)
{
    return 0;
}

void PG_WorldInit(pgWorldObject* world)
{
	world->bodyList = PyList_New(0);
	world->jointList = PyList_New(0);
	world->contactList = PyList_New(0);
	world->fDamping = 0.0;
	world->fStepTime = 0.1;
	PG_Set_Vector2(world->vecGravity,0.0,-10);
	world->fTotalTime = 0.0;

}

pgWorldObject* _PG_WorldNewInternal(PyTypeObject *type)
{
	pgWorldObject* op = (pgWorldObject*)type->tp_alloc(type, 0);
	PG_WorldInit(op);
	return op;
}

PyObject* _PG_WorldNew(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	/* In case we have arguments in the python code, parse them later
	* on.
	*/
	if(PyType_Ready(type) == -1) return NULL;
	return (PyObject*) _PG_WorldNewInternal(type);
}

pgWorldObject* PG_WorldNew()
{
	return (pgWorldObject*) _PG_WorldNew(&pgWorldType, NULL, NULL);
}

void PG_WorldDestroy(pgWorldObject* world)
{
	/*
	* DECREF anything related to the world, such as the lists and
	* release any other memory hold by it.
	*/
	Py_XDECREF(world->bodyList);
	Py_XDECREF(world->jointList);
	Py_XDECREF(world->contactList);

	world->ob_type->tp_free((PyObject*)world);
}

//static PyObject* _world_test_noargs(pgWorldObject *world)
//{
//	/* Do some things here */
//	//Py_RETURN_NONE;
//	PyObject* result;
//	result = Py_BuildValue("s","world test");
//	return result;
//}
//
//static PyObject* _world_test_args(pgWorldObject *world, PyObject *args)
//{
//	/* Parse arguments and do some things here */
//	Py_RETURN_NONE;
//}

static PyObject* _world_update(pgWorldObject* world,PyObject* pyfloat)
{
	double dt = 0.1;
	//double dt = PyFloat_AsDouble(pyfloat);
	PG_Update(world,dt);
	Py_RETURN_NONE;
}

static PyObject* _world_add_body(pgWorldObject* world,PyObject* args)
{
	pgBodyObject* body;
	if (!PyArg_ParseTuple(args,"O",&body))
	{
		PyErr_SetString(PyExc_ValueError,"arg is not body type");
		return NULL;
	}
	if(PG_AddBodyToWorld(world,body))
	{
		Py_RETURN_FALSE;
	}
	else
	{
		Py_RETURN_TRUE;
	}
}

static PyObject* _world_add_joint(pgWorldObject* world,PyObject* args)
{
	pgJointObject* joint;
	if (!PyArg_ParseTuple(args,"O",&joint))
	{
		PyErr_SetString(PyExc_ValueError,"arg is not joint type");
		return NULL;
	}
	if(PG_AddJointToWorld(world,joint))
	{
		Py_RETURN_FALSE;
	}
	else
	{
		Py_RETURN_TRUE;
	}
}


static PyObject* _pgWorld_getGravity(pgWorldObject* world,void* closure)
{
	return PyComplex_FromCComplex(world->vecGravity);
}

static int _pgWorld_setGravity(pgWorldObject* world,PyObject* value,void* closure)
{
	if (value == NULL || (!PyComplex_Check(value))) {
		PyErr_SetString(PyExc_TypeError, "Cannot set the gravity attribute");
		return -1;
	}
	else
	{
		world->vecGravity = PyComplex_AsCComplex(value);
		return 0;
	}
}

static PyObject* _pgWorld_getBodyList(pgWorldObject* world,void* closure)
{
	Py_INCREF (world->bodyList);
	return world->bodyList;
}

static PyObject* _pgWorld_getJointList(pgWorldObject* world,void* closure)
{
	Py_INCREF (world->jointList);
	return world->jointList;
}



/**
* Here we allow the Python object to do stuff like
*
*  myworld.test_noargs ()
*  myworld.test_args (arg1, arg2, ...)
*/

static PyGetSetDef _pgWorld_getseters[] = {
	{
		"gravity",(getter)_pgWorld_getGravity,(setter)_pgWorld_setGravity,"gravity",NULL,
	},
	{
            "body_list",(getter)_pgWorld_getBodyList,NULL,NULL,NULL
	},
	{
            "joint_list",(getter)_pgWorld_getJointList,NULL,NULL,NULL
	},
	{
            NULL, NULL, NULL, NULL, NULL
	}
};


static PyMethodDef _pgWorld_methods[] =
{
	//{ "test_noargs", (PyCFunction) _world_test_noargs, METH_NOARGS, "" },
	//{ "test_args", (PyCFunction) _world_test_args, METH_VARARGS, "" },
	{ "update", (PyCFunction) _world_update, METH_VARARGS, "" },
	{"add_body",(PyCFunction) _world_add_body, METH_VARARGS, ""},
	{"add_joint",(PyCFunction) _world_add_joint, METH_VARARGS, ""},
	{ NULL, NULL, 0, NULL } /* The NULL sentinel is important! */
};

static PyMemberDef _pgWorld_members[] = 
{
	{"damping",T_DOUBLE,offsetof(pgWorldObject,fDamping),0,""},
	{
            NULL, 0, 0, 0, NULL
	}
}; 




PyTypeObject pgWorldType =
{
	PyObject_HEAD_INIT(NULL)
	0,
	"physics.World",            /* tp_name */
	sizeof(pgWorldObject),      /* tp_basicsize */
	0,                          /* tp_itemsize */
	(destructor) PG_WorldDestroy,/* tp_dealloc */
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
	_pgWorld_methods,           /* tp_methods */
	_pgWorld_members,           /* tp_members */
	_pgWorld_getseters,         /* tp_getset */
	0,                          /* tp_base */
	0,                          /* tp_dict */
	0,                          /* tp_descr_get */
	0,                          /* tp_descr_set */
	0,                          /* tp_dictoffset */
	0,                          /* tp_init */
	0,                          /* tp_alloc */
	_PG_WorldNew,               /* tp_new */
	0,                          /* tp_free */
	0,                          /* tp_is_gc */
	0,                          /* tp_bases */
	0,                          /* tp_mro */
	0,                          /* tp_cache */
	0,                          /* tp_subclasses */
	0,                          /* tp_weaklist */
	0                           /* tp_del */
};

