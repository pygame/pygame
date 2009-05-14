/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners
  Copyright (C) 2008 Marcus von Appen
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

#include "ft_mod.h"
#include "pgsdl.h"
#include "freetypebase_doc.h"

static FT_Library g_freetype_lib = NULL;

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
 * Deinitialize the FreeType library.
 */
static void
_quit(void)
{
    if (g_freetype_lib)
    {
        FT_Done_FreeType(g_freetype_lib);
        g_freetype_lib = NULL;
    }
}

/*
 * Initialize the FreeType library.
 * Return 1 if initialization was successful, 0 otherwise.
 */
static int
_init(void)
{
    FT_Error error;

    if (g_freetype_lib)
        return 1;

    error = FT_Init_FreeType(&g_freetype_lib);

    return (error == 0);
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
    if (_init())
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
    return PyBool_FromLong((g_freetype_lib != 0));
}

PyMODINIT_FUNC
#ifdef IS_PYTHON_3
    PyInit_base(void)
#else
    initbase(void)
#endif
{
    PyObject *mod = NULL;

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


    /* 
     * Insert our base Font class into the main module 
     * TODO: We need a font class hawhaw
     */

    /*  

    if (PyType_Ready(&PySDLFont_TTF_Type) < 0)
        goto fail;

    Py_INCREF (&PySDLFont_TTF_Type);
    PyModule_AddObject (mod, "Font", (PyObject *) &PySDLFont_TTF_Type); 
    
    */

    /* 
     * Export C API.
     *
     * FIXME: What does this exactly do? 
     * Why do we need to export the base C API to python? 
     */

    /*
       font_export_capi (c_api);

       c_api_obj = PyCObject_FromVoidPtr ((void *) c_api, NULL);
       if (c_api_obj)
       PyModule_AddObject (mod, PYGAME_SDLTTF_ENTRY, c_api_obj);    
       */

    /* Import PyGame2 modules */
    if (import_pygame2_base() < 0)
        goto fail;
    if (import_pygame2_sdl_base() < 0)
        goto fail;
    if (import_pygame2_sdl_rwops() < 0)
        goto fail;
    if (import_pygame2_sdl_video() < 0)
        goto fail;

    RegisterQuitCallback(_quit);
    MODINIT_RETURN(mod);

fail:
    Py_XDECREF(mod);
    MODINIT_RETURN (NULL);
}
