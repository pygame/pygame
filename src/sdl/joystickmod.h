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
#ifndef _PYGAME_SDLJOYSTICK_H_
#define _PYGAME_SDLJOYSTICK_H_

#include "pgcompat.h"
#include <SDL.h>

#define PYGAME_SDLJOYSTICK_INTERNAL

#define MAX_JOYSTICKS 32
typedef struct {
    SDL_Joystick *joysticks[MAX_JOYSTICKS];
} _SDLJoystickState;

#ifdef IS_PYTHON_3
extern struct PyModuleDef _joystickmodule;
#define SDLJOYSTICK_MOD_STATE(mod) ((_SDLJoystickState*)PyModule_GetState(mod))
#define SDLJOYSTICK_STATE \
    SDLJOYSTICK_MOD_STATE(PyState_FindModule(&_joystickmodule))
#else
extern _SDLJoystickState _modstate;
#define SDLJOYSTICK_MOD_STATE(mod) (&_modstate)
#define SDLJOYSTICK_STATE SDLJOYSTICK_MOD_STATE(NULL)
#endif

extern PyTypeObject PyJoystick_Type;

#define PyJoystick_Check(x) (PyObject_TypeCheck (x, &PyJoystick_Type))
PyObject* PyJoystick_New (int index);

void joystick_export_capi (void **capi);

void joystickmod_add_joystick (int _index, SDL_Joystick *joystick);
void joystickmod_remove_joystick (int _index);
SDL_Joystick* joystickmod_get_joystick (int _index);

#endif /* _PYGAME_SDLJOYSTICK_H_ */
