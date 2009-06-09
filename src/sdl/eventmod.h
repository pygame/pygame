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
#ifndef _PYGAME_SDLEVENT_H_
#define _PYGAME_SDLEVENT_H_

#include "pgcompat.h"
#include <SDL.h>

#define PYGAME_SDLEVENT_INTERNAL

typedef struct {
    PyObject *filterhook;
} _SDLEventState;

#ifdef IS_PYTHON_3
extern struct PyModuleDef _eventmodule;
#define SDLEVENT_MOD_STATE(mod) ((_SDLEventState*)PyModule_GetState(mod))
#define SDLEVENT_STATE SDLEVENT_MOD_STATE(PyState_FindModule(&_eventmodule))
#else
extern _SDLEventState _modstate;
#define SDLEVENT_MOD_STATE(mod) (&_modstate)
#define SDLEVENT_STATE SDLEVENT_MOD_STATE(NULL)
#endif

extern PyTypeObject PyEvent_Type;

#define PyEvent_Check(x) (PyObject_TypeCheck (x, &PyEvent_Type))
PyObject* PyEvent_NewInternal (SDL_Event *event, int release);
PyObject* PyEvent_New (SDL_Event *event);
int PyEvent_SDLEventFromEvent (PyObject *ev, SDL_Event *event);

void event_export_capi (void **capi);

#endif /* _PYGAME_SDLEVENT_H_ */
