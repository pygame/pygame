#include "pgWorldObject.h"
#include "pgBodyObject.h"
#include "pgJointObject.h"

#define MAX_SOLVE_INTERAT 20

static PyTypeObject pgWorldType;

void _PG_FreeBodySimulation(pgWorldObject* world,double stepTime)
{
	Py_ssize_t size = PyList_Size((PyObject*)(world->bodyList));
	Py_ssize_t i;
	for (i = 0;i < size;i++)
	{
		pgBodyObject* body = (pgBodyObject*)(PyList_GetItem((PyObject*)(world->bodyList),i));
		PG_FreeUpdateBodyVel(world,body,stepTime);
	}
}

void _PG_BodyCollisionDetection(pgWorldObject* world)
{

}

void _PG_JointSolve(pgWorldObject* world,double stepTime)
{
	Py_ssize_t size = PyList_Size((PyObject*)(world->jointList));
	Py_ssize_t i;
	for (i = 0;i < size;i++)
	{
		pgJointObject* joint = (pgJointObject*)(PyList_GetItem((PyObject*)(world->jointList),i));
		if (joint->SolveConstraintPosition)
		{
			joint->SolveConstraintPosition(joint,stepTime);
		}
		if (joint->SolveConstraintVelocity)
		{
			joint->SolveConstraintVelocity(joint,stepTime);
		}
	}
}

void _PG_BodyPositionUpdate(pgWorldObject* world,double stepTime)
{
	Py_ssize_t size = PyList_Size((PyObject*)(world->bodyList));
	Py_ssize_t i;
	for (i = 0;i < size;i++)
	{
		pgBodyObject* body = (pgBodyObject*)(PyList_GetItem((PyObject*)(world->bodyList),i));
		PG_FreeUpdateBodyPos(world,body,stepTime);
	}
}


void PG_Update(pgWorldObject* world,double stepTime)
{
	int i;

	_PG_FreeBodySimulation(world, stepTime);
	_PG_BodyCollisionDetection(world);
	for (i = 0;i < MAX_SOLVE_INTERAT;i++)
	{
		_PG_JointSolve(world,stepTime);
	}
	
	_PG_BodyPositionUpdate(world, stepTime);
}


void PG_AddBodyToWorld(pgWorldObject* world,pgBodyObject* body)
{
	PyList_Append((PyObject*)world->bodyList,(PyObject*)body);
}

void PG_RemoveBodyFromWorld(pgWorldObject* world,pgBodyObject* body)
{
	
}

void PG_AddJointToWorld(pgWorldObject* world,pgJointObject* joint)
{
	PyList_Append((PyObject*)world->jointList,(PyObject*)joint);
}

void PG_RemoveJointFromWorld(pgWorldObject* world,pgJointObject* joint)
{

}






void PG_WorldInit(pgWorldObject* world)
{
	world->bodyList = (PyListObject*)PyList_New(0);
	world->jointList = (PyListObject*)PyList_New(0);
	world->fDamping = 0.0;
	world->fStepTime = 0.1;
	PG_Set_Vector2(world->vecGravity,0.0,-50);
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
	if(PyType_Ready(type)==-1) return NULL;
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
	//TODO: i don't know how to release jointObject here,
	//		since it's sub class's constructor function is diff from each other
	//Py_XDECREF(world->jointList);

	world->ob_type->tp_free((PyObject*)world);
}

static PyObject* _world_test_noargs(pgWorldObject *world)
{
	/* Do some things here */
	Py_RETURN_NONE;
}

static PyObject* _world_test_args(pgWorldObject *world, PyObject *args)
{
	/* Parse arguments and do some things here */
	Py_RETURN_NONE;
}

/**
* Here we allow the Python object to do stuff like
*
*  myworld.test_noargs ()
*  myworld.test_args (arg1, arg2, ...)
*/
static PyMethodDef _pgWorld_methods[] =
{
	{ "test_noargs", (PyCFunction) _world_test_noargs, METH_NOARGS, "" },
	{ "test_args", (PyCFunction) _world_test_args, METH_VARARGS, "" },
	{ NULL, NULL, 0, NULL } /* The NULL sentinel is important! */
};

static PyTypeObject pgWorldType =
{
	PyObject_HEAD_INIT(NULL)
	0,
	"physics.world",            /* tp_name */
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
	0,                          /* tp_members */
	0,                          /* tp_getset */
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

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initphysics(void) 
{
	PyObject* m;

	if (PyType_Ready(&pgWorldType) < 0)
		return;

	m = Py_InitModule3("physics", _pgWorld_methods,
		"Example module that creates an extension type.");

	if (m == NULL)
		return;

	Py_INCREF(&pgWorldType);
	PyModule_AddObject(m, "world", (PyObject *)&pgWorldType);
}
