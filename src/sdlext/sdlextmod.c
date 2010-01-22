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
#define PYGAME_SDLEXT_INTERNAL

#include "pymacros.h"
#include "sdlextmod.h"
#include "pgsdlext.h"
#include "pgsdl.h"
#include "sdlextbase_doc.h"

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_base (void)
#else
PyMODINIT_FUNC initbase (void)
#endif
{
    PyObject *mod = NULL;
    PyObject *c_api_obj;
    static void *c_api[PYGAME_SDLEXT_SLOTS];
    
#ifdef IS_PYTHON_3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "base",
        DOC_BASE,
        -1,
        NULL,
        NULL, NULL, NULL, NULL
    };
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("base", NULL, DOC_BASE);
#endif
    if (!mod)
        goto fail;

    if (PyType_Ready (&PyPixelArray_Type) < 0)
        goto fail;
    ADD_OBJ_OR_FAIL (mod, "PixelArray", PyPixelArray_Type, fail);
    
    /* Export C API */
    pixelarray_export_capi (c_api);
   
    c_api_obj = PyCObject_FromVoidPtr ((void *) c_api, NULL);
    if (c_api_obj)
    {
        if (PyModule_AddObject (mod, PYGAME_SDLEXT_ENTRY, c_api_obj) == -1)
        {
            Py_DECREF (c_api_obj);
            goto fail;
        }
    }

    if (import_pygame2_base () < 0)
        goto fail;
    if (import_pygame2_sdl_base () < 0)
        goto fail;
    if (import_pygame2_sdl_video () < 0)
        goto fail;

    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
