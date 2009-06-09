/*
  Copyright (C) 2002-2007 Ulf Ekstrom except for the bitcount function.
  This wrapper code was originally written by Danny van Bruggen(?) for
  the SCAM library, it was then converted by Ulf Ekstrom to wrap the
  bitmask library, a spinoff from SCAM.

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
#ifndef _PYGAME_MASKMOD_H_
#define _PYGAME_MASKMOD_H_

#define PYGAME_MASK_INTERNAL

#include <math.h>
#include "pgcompat.h"
#include "pgdefines.h"
#include "bitmask.h"

extern PyTypeObject PyMask_Type;
#define PyMask_Check(x) (PyObject_TypeCheck (x, &PyMask_Type))
PyObject* PyMask_New (int w, int h);

void mask_export_capi (void **capi);

#endif /* _PYGAME_MASKMOD_H_ */
