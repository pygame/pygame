/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners, 2008 Marcus von Appen

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
#define PYGAME_SDLTTFCONSTANTS_INTERNAL

#include "ttfmod.h"
#include "pgttf.h"

/* macros used to create each constant */
#define DEC_CONST(x)  PyModule_AddIntConstant(module, #x, (long) TTF_##x)
#define DEC_CONSTS(x)  PyModule_AddIntConstant(module, #x, (long) x)

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_constants (void)
#else
PyMODINIT_FUNC initconstants (void)
#endif
{
    PyObject *module;

#ifdef IS_PYTHON_3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "constants",
        "Pygame SDL TTF constants",
        -1,
        NULL,
         NULL, NULL, NULL, NULL
   };
    module = PyModule_Create (&_module);
#else
    module = Py_InitModule3 ("constants", NULL, "Pygame SDL TTF constants");
#endif
    if (!module)
        goto fail;

    DEC_CONST (STYLE_NORMAL);
    DEC_CONST (STYLE_BOLD);
    DEC_CONST (STYLE_ITALIC);
    DEC_CONST (STYLE_UNDERLINE);

    DEC_CONSTS (RENDER_SOLID);
    DEC_CONSTS (RENDER_SHADED);
    DEC_CONSTS (RENDER_BLENDED);
    
    MODINIT_RETURN(module);
fail:
    Py_XDECREF (module);
    MODINIT_RETURN (NULL);
}
