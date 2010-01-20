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
#define PYGAME_SDLGFXCONSTANTS_INTERNAL

#include "gfxmod.h"
#include "pggfx.h"
#include <SDL_rotozoom.h>

/* macros used to create each constant */
#define DEC_CONSTN(x)  PyModule_AddIntConstant(module, #x, (long) x)

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
        "Pygame SDL_gfx constants",
        -1,
        NULL,
        NULL, NULL, NULL, NULL
    };
    module = PyModule_Create (&_module);
#else
    module = Py_InitModule3 ("constants", NULL, "Pygame SDL_gfx constants");
#endif
    if (!module)
        goto fail;
    
    DEC_CONSTN(FPS_UPPER_LIMIT);
    DEC_CONSTN(FPS_LOWER_LIMIT);
    DEC_CONSTN(FPS_DEFAULT);

    DEC_CONSTN(SMOOTHING_OFF);
    DEC_CONSTN(SMOOTHING_ON);
    
    MODINIT_RETURN(module);
fail:
    Py_XDECREF (module);
    MODINIT_RETURN (NULL);
}
