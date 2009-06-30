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
#include "ft_wrap.h"
#include "pgfreetype.h"
#include "freetypebase_doc.h"

static int _ft_traverse (PyObject *mod, visitproc visit, void *arg);
static int _ft_clear (PyObject *mod);

static PyObject *_ft_quit(PyObject *self);
static PyObject *_ft_init(PyObject *self);
static PyObject *_ft_get_version(PyObject *self);
static PyObject *_ft_get_error(PyObject *self);
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
        "get_error", 
        (PyCFunction) _ft_get_error, 
        METH_NOARGS,
        DOC_BASE_GET_ERROR 
    },
    { 
        "get_version", 
        (PyCFunction) _ft_get_version, 
        METH_NOARGS,
        DOC_BASE_GET_VERSION 
    },
    { NULL, NULL, 0, NULL },
};

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
    if (FREETYPE_MOD_STATE (self)->freetype)
    {
        PGFT_Quit(FREETYPE_MOD_STATE (self)->freetype);
        FREETYPE_MOD_STATE (self)->freetype = NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_ft_init(PyObject *self)
{
    FT_Error error;

    if (FREETYPE_MOD_STATE (self)->freetype)
        Py_RETURN_NONE;

    error = PGFT_Init(&(FREETYPE_MOD_STATE (self)->freetype));
    if (error != 0)
    {
        /* TODO: More accurate error message */
        PyErr_SetString(PyExc_PyGameError, 
                "Failed to initialize the FreeType2 library");
        return NULL;
    }

    Py_RETURN_NONE;
}


static PyObject *
_ft_get_error(PyObject *self)
{
    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (ft->_error_msg[0])
    {
        return Text_FromUTF8(ft->_error_msg);
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
    return PyBool_FromLong((FREETYPE_MOD_STATE (self)->freetype != NULL));
}


static int
_ft_traverse (PyObject *mod, visitproc visit, void *arg)
{
    return 0;
}

static int
_ft_clear (PyObject *mod)
{
    if (FREETYPE_MOD_STATE(mod)->freetype)
    {
        PGFT_Quit(FREETYPE_MOD_STATE(mod)->freetype);
        FREETYPE_MOD_STATE(mod)->freetype = NULL;
    }
    return 0;
}

#ifdef IS_PYTHON_3
struct PyModuleDef _freetypemodule = {
    PyModuleDef_HEAD_INIT,
    "base",
    DOC_BASE, 
    sizeof (_FreeTypeState),
    _ft_methods,
    NULL,
    _ft_traverse,
    _ft_clear,
    NULL
};
#else
_FreeTypeState _modstate;
#endif

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
    mod = PyModule_Create(&_freetypemodule);
#else
    /* Standard init for 2.x module */
    mod = Py_InitModule3 ("base", _ft_methods, DOC_BASE);
#endif

    if (!mod)
        goto fail;
    FREETYPE_MOD_STATE(mod)->freetype = NULL;

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

#ifdef HAVE_PYGAME_SDL_VIDEO
    if (import_pygame2_sdl_base () < 0)
        goto fail;
    if (import_pygame2_sdl_video () < 0)
        goto fail;
#endif /* HAVE_PYGAME_SDL_VIDEO */

    MODINIT_RETURN(mod);

fail:
    Py_XDECREF(mod);
    MODINIT_RETURN (NULL);
}
