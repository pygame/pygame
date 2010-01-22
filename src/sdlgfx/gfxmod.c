/*
  pygame - Python Game Library
  Copyright (C) 2008 Marcus von Appen

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
#define PYGAME_SDLGFX_INTERNAL

#include <SDL_gfxPrimitives.h>
#include "pymacros.h"
#include "gfxmod.h"
#include "pggfx.h"
#include "pgsdl.h"
#include "sdlgfxbase_doc.h"

static PyObject* _gfx_getcompiledversion (PyObject *self);

static PyMethodDef _gfx_methods[] = {
    { "get_compiled_version", (PyCFunction) _gfx_getcompiledversion,
      METH_NOARGS, DOC_BASE_GET_COMPILED_VERSION },
    { NULL, NULL, 0, NULL },
};

static PyObject*
_gfx_getcompiledversion (PyObject *self)
{
    return Py_BuildValue ("(iii)", SDL_GFXPRIMITIVES_MAJOR,
        SDL_GFXPRIMITIVES_MINOR, SDL_GFXPRIMITIVES_MICRO);
}

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_base (void)
#else
PyMODINIT_FUNC initbase (void)
#endif
{
    PyObject *mod = NULL;
    PyObject *c_api_obj;
    static void *c_api[PYGAME_SDLGFX_SLOTS];

#ifdef IS_PYTHON_3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "base",
        DOC_BASE,
        -1,
        _gfx_methods,
        NULL, NULL, NULL, NULL
    };
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("base", _gfx_methods, DOC_BASE);
#endif
    if (!mod)
        goto fail;

    if (PyType_Ready (&PyFPSmanager_Type) < 0)
        goto fail;
    ADD_OBJ_OR_FAIL (mod, "FPSmanager", PyFPSmanager_Type, fail);

    /* Export C API */
    fps_export_capi (c_api);

    c_api_obj = PyCObject_FromVoidPtr ((void *) c_api, NULL);
    if (c_api_obj)
    {
        if (PyModule_AddObject (mod, PYGAME_SDLGFX_ENTRY, c_api_obj) == -1)
        {
            Py_DECREF (c_api_obj);
            goto fail;
        }
    }   
    if (import_pygame2_base () < 0)
        goto fail;
    if (import_pygame2_sdl_base () < 0)
        goto fail;
    MODINIT_RETURN(mod);

fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
