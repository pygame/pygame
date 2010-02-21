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
#define PYGAME_SDLJOYSTICK_INTERNAL

#include "joystickmod.h"
#include "pgsdl.h"
#include "sdljoystick_doc.h"

static PyObject* _joystick_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);
static int _joystick_init (PyObject *joystick, PyObject *args, PyObject *kwds);
static void _joystick_dealloc (PyJoystick *self);

static PyObject* _joystick_getname (PyObject *self, void *closure);
static PyObject* _joystick_getindex (PyObject *self, void *closure);
static PyObject* _joystick_getnumaxes (PyObject *self, void *closure);
static PyObject* _joystick_getnumballs (PyObject *self, void *closure);
static PyObject* _joystick_getnumhats (PyObject *self, void *closure);
static PyObject* _joystick_getnumbuttons (PyObject *self, void *closure);
static PyObject* _joystick_getopened (PyObject *self, void *closure);

static PyObject* _joystick_getaxis (PyObject *self, PyObject *args);
static PyObject* _joystick_gethat (PyObject *self, PyObject *args);
static PyObject* _joystick_getbutton (PyObject *self, PyObject *args);
static PyObject* _joystick_getball (PyObject *self, PyObject *args);
static PyObject* _joystick_open (PyObject *self);
static PyObject* _joystick_close (PyObject *self);

/**
 */
static PyMethodDef _joystick_methods[] = {
    { "get_axis", _joystick_getaxis, METH_O, DOC_JOYSTICK_JOYSTICK_GET_AXIS },
    { "get_hat", _joystick_gethat, METH_O, DOC_JOYSTICK_JOYSTICK_GET_HAT },
    { "get_button", _joystick_getbutton, METH_O,
      DOC_JOYSTICK_JOYSTICK_GET_BUTTON },
    { "get_ball", _joystick_getball, METH_O, DOC_JOYSTICK_JOYSTICK_GET_BALL },
    { "open", (PyCFunction)_joystick_open, METH_NOARGS,
      DOC_JOYSTICK_JOYSTICK_OPEN },
    { "close", (PyCFunction)_joystick_close, METH_NOARGS,
      DOC_JOYSTICK_JOYSTICK_CLOSE },
    { NULL, NULL, 0, NULL }
};

/**
 */
static PyGetSetDef _joystick_getsets[] = {
    { "name", _joystick_getname, NULL, DOC_JOYSTICK_JOYSTICK_NAME, NULL },
    { "index", _joystick_getindex, NULL, DOC_JOYSTICK_JOYSTICK_INDEX, NULL },
    { "num_axes", _joystick_getnumaxes, NULL, DOC_JOYSTICK_JOYSTICK_NUM_AXES,
      NULL },
    { "num_balls", _joystick_getnumballs, NULL,
      DOC_JOYSTICK_JOYSTICK_NUM_BALLS, NULL },
    { "num_hats", _joystick_getnumhats, NULL, DOC_JOYSTICK_JOYSTICK_NUM_HATS,
      NULL },
    { "num_buttons", _joystick_getnumbuttons, NULL,
      DOC_JOYSTICK_JOYSTICK_NUM_BUTTONS, NULL },
    { "opened", _joystick_getopened, NULL, DOC_JOYSTICK_JOYSTICK_OPENED, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

/**
 */
PyTypeObject PyJoystick_Type =
{
    TYPE_HEAD(NULL, 0)
    "joystick.Joystick",              /* tp_name */
    sizeof (PyJoystick),   /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _joystick_dealloc, /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    0,                          /* tp_repr */
    0,                          /* tp_as_number */
    0,                          /* tp_as_sequence */
    0,                          /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    0,                          /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    DOC_JOYSTICK_JOYSTICK,
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _joystick_methods,          /* tp_methods */
    0,                          /* tp_members */
    _joystick_getsets,          /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _joystick_init,  /* tp_init */
    0,                          /* tp_alloc */
    _joystick_new,              /* tp_new */
    0,                          /* tp_free */
    0,                          /* tp_is_gc */
    0,                          /* tp_bases */
    0,                          /* tp_mro */
    0,                          /* tp_cache */
    0,                          /* tp_subclasses */
    0,                          /* tp_weaklist */
    0,                          /* tp_del */
#if PY_VERSION_HEX >= 0x02060000
    0                           /* tp_version_tag */
#endif
};

static void
_joystick_dealloc (PyJoystick *self)
{
    if (self->joystick && SDL_JoystickOpened (self->index))
    {
        SDL_JoystickClose (self->joystick);
        joystickmod_remove_joystick (self->index);
    }
    ((PyObject*)self)->ob_type->tp_free ((PyObject *) self);
}

static PyObject*
_joystick_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyJoystick *stick = (PyJoystick*) type->tp_alloc (type, 0);
    if (!stick)
        return NULL;
    stick->index = 0;
    stick->joystick = NULL;
    return (PyObject *) stick;
}

static int
_joystick_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    int _index;
    SDL_Joystick *joystick;
    
    ASSERT_JOYSTICK_INIT(-1);
    
    if (!PyArg_ParseTuple (args, "i", &_index))
        return -1;
    if (_index < 0 || _index > SDL_NumJoysticks ())
    {
        PyErr_SetString (PyExc_ValueError, "invalid joystick index");
        return -1;
    }
    joystick = SDL_JoystickOpen (_index);
    if (!joystick)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return -1;
    }

    ((PyJoystick*)self)->joystick = joystick;
    ((PyJoystick*)self)->index = _index;
    joystickmod_add_joystick (_index, joystick);
    
    return 0;
}

/* Getters/Setters */
static PyObject*
_joystick_getname (PyObject *self, void *closure)
{
    ASSERT_JOYSTICK_INIT(NULL);
    return Text_FromUTF8 (SDL_JoystickName (((PyJoystick*)self)->index));
}

static PyObject*
_joystick_getindex (PyObject *self, void *closure)
{
    ASSERT_JOYSTICK_INIT(NULL);
    return PyInt_FromLong (((PyJoystick*)self)->index);
}

static PyObject*
_joystick_getnumaxes (PyObject *self, void *closure)
{
    ASSERT_JOYSTICK_OPEN (self, NULL);
    return PyInt_FromLong (SDL_JoystickNumAxes (((PyJoystick*)self)->joystick));
}

static PyObject*
_joystick_getnumballs (PyObject *self, void *closure)
{
    ASSERT_JOYSTICK_OPEN (self, NULL);
    return PyInt_FromLong
        (SDL_JoystickNumBalls (((PyJoystick*)self)->joystick));
}

static PyObject*
_joystick_getnumhats (PyObject *self, void *closure)
{
    ASSERT_JOYSTICK_OPEN (self, NULL);
    return PyInt_FromLong
        (SDL_JoystickNumHats (((PyJoystick*)self)->joystick));
}

static PyObject*
_joystick_getnumbuttons (PyObject *self, void *closure)
{   
    ASSERT_JOYSTICK_OPEN (self, NULL);
    return PyInt_FromLong
        (SDL_JoystickNumButtons (((PyJoystick*)self)->joystick));
}

static PyObject*
_joystick_getopened (PyObject *self, void *closure)
{
    ASSERT_JOYSTICK_INIT(NULL);
    return PyBool_FromLong (SDL_JoystickOpened (((PyJoystick*)self)->index));
}

/* Methods */
static PyObject*
_joystick_getaxis (PyObject *self, PyObject *args)
{
    Sint16 value;
    int axis, maxaxes;
    SDL_Joystick *joystick = ((PyJoystick*)self)->joystick;
    
    ASSERT_JOYSTICK_OPEN (self, NULL);
    
    if (!IntFromObj (args, &axis))
        return NULL;

    maxaxes = SDL_JoystickNumAxes (joystick);
    if (axis < 0 || axis >= maxaxes)
    {
        PyErr_SetString (PyExc_ValueError, "axis must be a valid axis");
        return NULL;
    }
    value = SDL_JoystickGetAxis(joystick, axis);
    return PyInt_FromLong (value);
}

static PyObject*
_joystick_gethat (PyObject *self, PyObject *args)
{
    Uint8 value;
    int hat, maxhats;
    SDL_Joystick *joystick = ((PyJoystick*)self)->joystick;
    
    ASSERT_JOYSTICK_OPEN (self, NULL);
    
    if (!IntFromObj (args, &hat))
        return NULL;

    maxhats = SDL_JoystickNumHats (joystick);
    if (hat < 0 || hat >= maxhats)
    {
        PyErr_SetString (PyExc_ValueError, "hat must be a valid hat");
        return NULL;
    }
    value = SDL_JoystickGetHat(joystick, hat);
    return PyInt_FromLong (value);
}

static PyObject*
_joystick_getbutton (PyObject *self, PyObject *args)
{
    Uint8 value;
    int button, maxbuttons;
    SDL_Joystick *joystick = ((PyJoystick*)self)->joystick;
    
    ASSERT_JOYSTICK_OPEN (self, NULL);
    
    if (!IntFromObj (args, &button))
        return NULL;

    maxbuttons = SDL_JoystickNumHats (joystick);
    if (button < 0 || button >= maxbuttons)
    {
        PyErr_SetString (PyExc_ValueError, "button must be a valid button");
        return NULL;
    }
    value = SDL_JoystickGetButton(joystick, button);
    return PyBool_FromLong (value);
}

static PyObject*
_joystick_getball (PyObject *self, PyObject *args)
{
    int dx, dy;
    int ball, maxballs;
    SDL_Joystick *joystick = ((PyJoystick*)self)->joystick;
    
    ASSERT_JOYSTICK_OPEN (self, NULL);
    
    if (!IntFromObj (args, &ball))
        return NULL;

    maxballs = SDL_JoystickNumBalls (joystick);
    if (ball < 0 || ball >= maxballs)
    {
        PyErr_SetString (PyExc_ValueError, "ball must be a valid ball");
        return NULL;
    }
    if (SDL_JoystickGetBall (joystick, ball, &dx, &dy) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    return Py_BuildValue ("(ii)", dx, dy);
}

static PyObject*
_joystick_open (PyObject *self)
{
    SDL_Joystick *joystick;
    
    ASSERT_JOYSTICK_INIT(NULL);
    if (SDL_JoystickOpened (((PyJoystick*)self)->index))
        Py_RETURN_NONE; /* Already open */
    
    joystick = SDL_JoystickOpen (((PyJoystick*)self)->index);
    if (!joystick)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    ((PyJoystick*)self)->joystick = joystick;
    joystickmod_add_joystick (((PyJoystick*)self)->index, joystick);
    Py_RETURN_NONE;
}

static PyObject*
_joystick_close (PyObject *self)
{
    ASSERT_JOYSTICK_INIT(NULL);
    
    if (!SDL_JoystickOpened (((PyJoystick*)self)->index))
        Py_RETURN_NONE; /* Already closed */
    
    SDL_JoystickClose (((PyJoystick*)self)->joystick);
    ((PyJoystick*)self)->joystick = NULL;
    joystickmod_remove_joystick (((PyJoystick*)self)->index);

    Py_RETURN_NONE;
}

/* C API */
PyObject*
PyJoystick_New (int _index)
{
    PyJoystick *joystick;
    SDL_Joystick *joy;
    
    ASSERT_JOYSTICK_INIT(NULL);
    
   if (_index < 0 || _index > SDL_NumJoysticks ())
    {
        PyErr_SetString (PyExc_ValueError, "invalid joystick index");
        return NULL;
    }

   joystick = (PyJoystick*) PyJoystick_Type.tp_new (&PyJoystick_Type, NULL,
       NULL);
   if (!joystick)
       return NULL;
        
   joy = SDL_JoystickOpen (_index);
   if (!joy)
   {
       Py_DECREF (joystick);
       PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
       return NULL;
   }
   
   joystick->joystick = joy;
   joystick->index = _index;
   joystickmod_add_joystick (_index, joy);
   return (PyObject*) joystick;
}

void
joystick_export_capi (void **capi)
{
    capi[PYGAME_SDLJOYSTICK_FIRSTSLOT] = &PyJoystick_Type;
    capi[PYGAME_SDLJOYSTICK_FIRSTSLOT+1] = (void *)PyJoystick_New;
}
