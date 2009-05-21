/*
  pygame - Python Game Library
  Copyright (C) 2009 Vicent Marti

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

#define PYGAME_FREETYPE_INTERNAL

#include "ft_mod.h"
#include "pgfreetype.h"
#include "pgsdl.h"
#include "freetypebase_doc.h"

static FreeTypeInstance *__freetype = NULL;

static int _init(void);
static void _quit(void);

static PyObject *_ft_quit(PyObject *self);
static PyObject *_ft_init(PyObject *self);
static PyObject *_ft_get_version(PyObject *self);
static PyObject *_ft_was_init(PyObject *self);

/***************************************************************
 *
 * Fonts Module method table
 *
 **************************************************************/
static PyMethodDef _ft_methods[] = 
{
    { 
        "init", 
        (PyCFunction) _ft_init, 
        METH_NOARGS, 
        DOC_BASE_INIT 
    },
    { 
        "quit", 
        (PyCFunction) _ft_quit, 
        METH_NOARGS, 
        DOC_BASE_QUIT 
    },
    { 
        "was_init", 
        (PyCFunction) _ft_was_init, 
        METH_NOARGS, 
        DOC_BASE_WAS_INIT 
    },
    { 
        "get_version", 
        (PyCFunction) _ft_get_version, 
        METH_NOARGS,
        DOC_BASE_GET_VERSION 
    },
    { NULL, NULL, 0, NULL },
};

/*
 * Get a pointer to the active FT library
 *
 * TODO: Someday this should automatically handle returning
 * libraries based on the active thread to prevent multi-
 * threading issues.
 */
FreeTypeInstance *
_get_freetype(void)
{
    return __freetype;
}

/*
 * Deinitialize the FreeType library.
 */
static void
_quit(void)
{
    if (__freetype)
    {
        PGFT_Quit(__freetype);
        __freetype = NULL;
    }
}

/*
 * Initialize the FreeType library.
 */
static int
_init(void)
{
    FT_Error error;

    if (__freetype)
        return 0;

    error = PGFT_Init(&__freetype);

    return (error);
}

/***************************************************************
 *
 * Bindings for initialization/cleanup functions
 *
 * Explicit init/quit functions are required to work around
 * some issues regarding module caching and multi-threaded apps.
 * It's always good to let the user choose when to initialize
 * the module.
 *
 * TODO: These bindings can be removed once proper threading
 * support is in place.
 *
 ***************************************************************/
static PyObject *
_ft_quit(PyObject *self)
{
    _quit();
    Py_RETURN_NONE;
}

static PyObject *
_ft_init(PyObject *self)
{
    if (_init() != 0)
    {
        /* TODO: More accurate error message */
        PyErr_SetString(PyExc_PyGameError, 
                "Failed to initialize the FreeType2 library");
        return NULL;
    }

    Py_RETURN_NONE;
}


static PyObject *
_ft_get_version(PyObject *self)
{
    /* Return the linked FreeType2 version */
    return Py_BuildValue("(iii)", FREETYPE_MAJOR, FREETYPE_MINOR, FREETYPE_PATCH);
}

static PyObject *
_ft_was_init(PyObject *self)
{
    return PyBool_FromLong((__freetype != 0));
}

PyMODINIT_FUNC
#ifdef IS_PYTHON_3
    PyInit_base(void)
#else
    initbase(void)
#endif
{
    PyObject *mod = NULL;
    PyObject *c_api_obj;
    static void *c_api[PYGAME_FREETYPE_SLOTS];

#ifdef IS_PYTHON_3

    /* Python 3 module initialization needs a PyModuleDef struct */
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "base",
        DOC_BASE,
        -1,
        _ft_methods,
        NULL, NULL, NULL, NULL
    };

    mod = PyModule_Create(&_module);

#else

    /* Standard init for 2.x module */
    mod = Py_InitModule3 ("base", _ft_methods, DOC_BASE);

#endif

    if (!mod)
        goto fail;


    /* Import Pygame2 Base API to access PyFont_Type */
    if (import_pygame2_base() < 0)
        goto fail;

    PyFreeTypeFont_Type.tp_base = &PyFont_Type;
    if (PyType_Ready(&PyFreeTypeFont_Type) < 0)
        goto fail;

    Py_INCREF(&PyFreeTypeFont_Type);
    PyModule_AddObject(mod, "Font", (PyObject *)&PyFreeTypeFont_Type); 

    /* 
     * Export C API.
     */

    ftfont_export_capi(c_api);

    c_api_obj = PyCObject_FromVoidPtr((void *) c_api, NULL);

    if (c_api_obj)
        PyModule_AddObject(mod, PYGAME_FREETYPE_ENTRY, c_api_obj);    

/*    RegisterQuitCallback(_quit); */

    MODINIT_RETURN(mod);

fail:
    Py_XDECREF(mod);
    MODINIT_RETURN (NULL);
}
