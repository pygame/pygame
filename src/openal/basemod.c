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

#include <AL.h>
#include "pgbase.h"
#include "pgopenal.h"
/*#include "openalbase_doc.h"*/

static PyObject* _openal_init (PyObject *self);
static PyObject* _openal_quit (PyObject *self);

static PyMethodDef _openal_methods[] = {
    { "init", (PyCFunction)_openal_init, METH_NOARGS, ""/*DOC_BASE_INIT*/ },
    { "quit", (PyCFunction)_openal_quit, METH_NOARGS, ""/*DOC_BASE_QUIT*/ },
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
        _image_methods,
        NULL, NULL, NULL, NULL
    };
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("base", _openal_methods, ""/*DOC_BASE*/);
#endif
    if (!mod)
        goto fail;
        
    PyModule_AddObject (mod, "Device", (PyObject *) &PyDevice_Type);
    
    if (import_pygame2_base () < 0)
        goto fail;
    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
