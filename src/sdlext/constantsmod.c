/*
  pygame - Python Game Library
  Copyright (C) 2006-2008 Rene Dudfield, Marcus von Appen

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
#define PYGAME_SDLEXTCONSTANTS_INTERNAL

#include "sdlextmod.h"
#include "pgsdlext.h"
#include "scrap.h"
#include "filters.h"

/* macros used to create each constant */
#define DEC_CONSTS(x)  PyModule_AddIntConstant(module, #x, (long) #x)
#define ADD_STRING_CONST(x) PyModule_AddStringConstant(module, #x, x)

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
        "Pygame SDL extension constants",
        -1,
        NULL,
        NULL, NULL, NULL, NULL
    };
    module = PyModule_Create (&_module);
#else
    module = Py_InitModule3 ("constants", NULL,
        "Pygame SDL extension constants");
#endif
    if (!module)
        goto fail;
    
    DEC_CONSTS (SCRAP_SELECTION);
    DEC_CONSTS (SCRAP_CLIPBOARD);
    ADD_STRING_CONST (SCRAP_FORMAT_TEXT);
    ADD_STRING_CONST (SCRAP_FORMAT_BMP);
    ADD_STRING_CONST (SCRAP_FORMAT_PPM);
    ADD_STRING_CONST (SCRAP_FORMAT_PBM);

    DEC_CONSTS (FILTER_C);
    DEC_CONSTS (FILTER_MMX);
    DEC_CONSTS (FILTER_SSE);
    
    MODINIT_RETURN(module);
fail:
    Py_XDECREF (module);
    MODINIT_RETURN (NULL);
}
