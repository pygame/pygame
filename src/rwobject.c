/*
    pygame - Python Game Library
    Copyright (C) 2000  Pete Shinners

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

    Pete Shinners
    pete@shinners.org
*/

/*
 *  SDL_RWops support for python objects
 */
#define PYGAMEAPI_RWOBJECT_INTERNAL
#include "pygame.h"


typedef struct
{
	PyObject* read;
	PyObject* write;
	PyObject* seek;
	PyObject* tell;
	PyObject* close;
}RWHelper;


static int rw_seek(SDL_RWops* context, int offset, int whence);
static int rw_read(SDL_RWops* context, void* ptr, int size, int maxnum);
static int rw_write(SDL_RWops* context, const void* ptr, int size, int maxnum);
static int rw_close(SDL_RWops* context);



static SDL_RWops* RWopsFromPython(PyObject* obj)
{
	SDL_RWops* rw;
	RWHelper* helper;

	if(!obj)
		return RAISE(PyExc_TypeError, "Invalid filetype object");

	if(PyString_Check(obj))
		rw = SDL_RWFromFile(PyString_AsString(obj), "rb");
	else if(PyFile_Check(obj))
		rw = SDL_RWFromFP(PyFile_AsFile(obj), 1);
	else
	{
		helper = PyMem_New(RWHelper, 1);
		helper->read = helper->write = helper->seek =
					helper->tell = helper->close = NULL;

		/*find our functions*/
		if(PyObject_HasAttrString(obj, "read"))
		{
			helper->read = PyObject_GetAttrString(obj, "read");
			if(helper->read && !PyCallable_Check(helper->read))
				helper->read = NULL;
		}
		if(PyObject_HasAttrString(obj, "write"))
		{
			helper->write = PyObject_GetAttrString(obj, "write");
			if(helper->write && !PyCallable_Check(helper->write))
				helper->write = NULL;
		}
		if(PyObject_HasAttrString(obj, "seek"))
		{
			helper->seek = PyObject_GetAttrString(obj, "seek");
			if(helper->seek && !PyCallable_Check(helper->seek))
				helper->seek = NULL;
		}
		if(PyObject_HasAttrString(obj, "tell"))
		{
			helper->tell = PyObject_GetAttrString(obj, "tell");
			if(helper->tell && !PyCallable_Check(helper->tell))
				helper->tell = NULL;
		}
		if(PyObject_HasAttrString(obj, "close"))
		{
			helper->close = PyObject_GetAttrString(obj, "close");
			if(helper->close && !PyCallable_Check(helper->close))
				helper->close = NULL;
		}

		rw = SDL_AllocRW();
		rw->hidden.unknown.data1 = (void*)helper;
		rw->seek = rw_seek;
		rw->read = rw_read;
		rw->write = rw_write;
		rw->close = rw_close;
	}

	return rw;
}



static int rw_seek(SDL_RWops* context, int offset, int whence)
{
	RWHelper* helper = (RWHelper*)context->hidden.unknown.data1;
	PyObject* result;
	int retval;

	if(!helper->seek || !helper->tell)
		return -1;

	if(!(offset == 0 && whence == SEEK_CUR)) /*being called only for 'tell'*/
	{
		result = PyObject_CallFunction(helper->seek, "ii", offset, whence);
		if(!result)
			return -1;
		Py_DECREF(result);
	}

	result = PyObject_CallFunction(helper->tell, NULL);
	if(!result)
		return -1;

	retval = PyInt_AsLong(result);
	Py_DECREF(result);

	return retval;
}


static int rw_read(SDL_RWops* context, void* ptr, int size, int maxnum)
{
	RWHelper* helper = (RWHelper*)context->hidden.unknown.data1;
	PyObject* result;
	int retval;

	if(!helper->read)
		return -1;

	result = PyObject_CallFunction(helper->read, "i", size * maxnum);
	if(!result)
		return -1;
		
	if(!PyString_Check(result))
	{
		Py_DECREF(result);
		return -1;
	}
		
	retval = PyString_GET_SIZE(result);
	memcpy(ptr, PyString_AsString(result), retval);
	retval /= size;

	Py_DECREF(result);
	return retval;
}


static int rw_write(SDL_RWops* context, const void* ptr, int size, int num)
{
	RWHelper* helper = (RWHelper*)context->hidden.unknown.data1;
	PyObject* result;

	if(!helper->write)
		return -1;

	result = PyObject_CallFunction(helper->write, "s#", ptr, size * num);
	if(!result)
		return -1;

	Py_DECREF(result);
	return num;
}


static int rw_close(SDL_RWops* context)
{
	RWHelper* helper = (RWHelper*)context->hidden.unknown.data1;
	PyObject* result;
	int retval = 0;

	if(helper->close)
	{
		result = PyObject_CallFunction(helper->close, NULL);
		if(result)
			retval = -1;
		Py_XDECREF(result);
	}

	Py_XDECREF(helper->seek);
	Py_XDECREF(helper->tell);
	Py_XDECREF(helper->write);
	Py_XDECREF(helper->read);
	Py_XDECREF(helper->close);
	PyMem_Del(helper);
	SDL_FreeRW(context);
	return retval;
}




static PyMethodDef rwobject__builtins__[] =
{
	{NULL, NULL}
};




void initrwobject()
{
	PyObject *module, *dict, *apiobj;
	static void* c_api[PYGAMEAPI_RWOBJECT_NUMSLOTS];

	/* Create the module and add the functions */
	module = Py_InitModule3("rwobject", rwobject__builtins__, "SDL_RWops support");
	dict = PyModule_GetDict(module);

	/* export the c api */
	c_api[0] = RWopsFromPython;
	apiobj = PyCObject_FromVoidPtr(c_api, NULL);
	PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
}
