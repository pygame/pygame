/*
  pygame - Python Game Library
  Copyright (C) 2010 Marcus von Appen

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
#define PYGAME_OPENALBASE_INTERNAL

#include "pymacros.h"
#include "openalmod.h"
#include "pgbase.h"
#include "pgopenal.h"
/*#include "openalbase_doc.h"*/

static PyObject* _openal_init (PyObject *self);
static PyObject* _openal_quit (PyObject *self);
static PyObject* _openal_geterror (PyObject *self);
static PyObject* _openal_isextensionpresent (PyObject *self, PyObject *args);

static PyMethodDef _openal_methods[] = {
    { "init", (PyCFunction)_openal_init, METH_NOARGS, ""/*DOC_BASE_INIT*/ },
    { "quit", (PyCFunction)_openal_quit, METH_NOARGS, ""/*DOC_BASE_QUIT*/ },
    { "get_error", (PyCFunction)_openal_geterror, METH_NOARGS, ""/*DOC_BASE_GETERROR*/ },
    { "is_extension_present", _openal_isextensionpresent, METH_VARARGS, "" },
    { NULL, NULL, 0, NULL },
};

static PyObject*
_openal_init (PyObject *self)
{
    Py_RETURN_NONE;
}

static PyObject*
_openal_quit (PyObject *self)
{
    Py_RETURN_NONE;
}

static PyObject*
_openal_geterror (PyObject *self)
{
    ALenum error = alGetError ();
    switch (error)
    {
    case AL_INVALID_ENUM:
        return Text_FromUTF8 ("invalid enumeration value");
    case AL_INVALID_VALUE:
        return Text_FromUTF8 ("invalid value");
    case AL_INVALID_OPERATION:
        return Text_FromUTF8 ("invalid operation request");
    case AL_OUT_OF_MEMORY:
        return Text_FromUTF8 ("insufficient memory");
    default:
        Py_RETURN_NONE;
    }
}

static PyObject*
_openal_isextensionpresent (PyObject *self, PyObject *args)
{
    char *extname = NULL;
    PyObject *device = NULL;
    ALCboolean present;
    
    if(!PyArg_ParseTuple (args, "s|O", &extname, &device))
        return NULL;
    if (device && !PyDevice_Check (device))
    {
        PyErr_SetString (PyExc_TypeError, "device must be a Device");
        return NULL;
    }
    
    if (device)
        present = alcIsExtensionPresent (PyDevice_AsDevice (device),
            (const ALchar*) extname);
    else
        present = alIsExtensionPresent ((const ALchar*) extname);
    if (SetALErrorException (alGetError ()))
        return NULL;
    if (present == ALC_FALSE)
        Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}

/* C API */
int
SetALErrorException (ALenum error)
{
    switch (error)
    {
    case AL_INVALID_ENUM:
        PyErr_SetString (PyExc_PyGameError, "invalid enumeration value");
        return 1;
    case AL_INVALID_VALUE:
        PyErr_SetString (PyExc_PyGameError, "invalid value");
        return 1;
    case AL_INVALID_OPERATION:
        PyErr_SetString (PyExc_PyGameError, "invalid operation request");
        return 1;
    case AL_OUT_OF_MEMORY:
        PyErr_SetString (PyExc_PyGameError, "insufficient memory");
        return 1;
    default:
        return 0;
    }
}

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_base (void)
#else
PyMODINIT_FUNC initbase (void)
#endif
{
    PyObject *mod;

#ifdef IS_PYTHON_3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "base",
        ""/*DOC_BASE*/,
        -1,
        _openal_methods,
        NULL, NULL, NULL, NULL
    };
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("base", _openal_methods, ""/*DOC_BASE*/);
#endif
    if (!mod)
        goto fail;
        
    if (import_pygame2_base () < 0)
        goto fail;
    
    PyDevice_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready (&PyDevice_Type) < 0)
        goto fail;
    
    ADD_OBJ_OR_FAIL (mod, "Device", PyDevice_Type, fail);
    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
