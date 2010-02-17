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
#define PYGAME_SDLCURSOR_INTERNAL

#include "mousemod.h"
#include "pgsdl.h"

static PyObject* _cursor_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);
static int _cursor_init (PyObject *cursor, PyObject *args, PyObject *kwds);
static void _cursor_dealloc (PyCursor *self);

/**
 */
/*
static PyMethodDef _cursor_methods[] = {
    { NULL, NULL, 0, NULL }
};
*/
/**
 */
/*
static PyGetSetDef _cursor_getsets[] = {
    { NULL, NULL, NULL, NULL, NULL }
};
*/

/**
 */
PyTypeObject PyCursor_Type =
{
    TYPE_HEAD(NULL, 0)
    "mouse.Cursor",              /* tp_name */
    sizeof (PyCursor),   /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _cursor_dealloc, /* tp_dealloc */
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
    "",
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    0,                          /* tp_methods */
    0,                          /* tp_members */
    0,                          /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _cursor_init,    /* tp_init */
    0,                          /* tp_alloc */
    _cursor_new,                /* tp_new */
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
_cursor_dealloc (PyCursor *self)
{
    if (self->cursor)
        SDL_FreeCursor (self->cursor);

    ((PyObject*)self)->ob_type->tp_free ((PyObject *) self);
}

static PyObject*
_cursor_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyCursor *cursor = (PyCursor*) type->tp_alloc (type, 0);
    if (!cursor)
        return NULL;
    cursor->cursor = NULL;
    return (PyObject*) cursor;
}

static int
_cursor_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    Uint8 *data, *mask;
    Py_ssize_t datalen, masklen;
    int w, h, hotx = 0, hoty = 0;
    PyObject *databuf, *maskbuf;
    SDL_Cursor *cursor;

    /* Cursor (buf, mask, w, h, x, y) */
    if (!PyArg_ParseTuple (args, "OOii|ii", &databuf, &maskbuf, &w, &h, 
            &hotx, &hoty))
    {
        PyObject *size, *pt;
        PyErr_Clear ();

        /* Cursor (buf, mask, size, pt) */
        if (PyArg_ParseTuple (args, "OOO|O", &databuf, &maskbuf, &size,
                &pt))
            return -1;
        if (!SizeFromObj (size, (pgint32*)&w, (pgint32*)&h))
            return -1;
        if (!PointFromObj (pt, &hotx, &hoty))
            return -1;
    }

    if (w < 0 || h < 0)
    {
        PyErr_SetString (PyExc_ValueError,
            "width and height must not be negative");
        return -1;
    }
    
    if ((w % 8) != 0)
    {
        PyErr_SetString (PyExc_ValueError, "width must be a multiple of 8");
        return -1;
    }

    if (PyObject_AsReadBuffer (databuf, (const void**)&data, &datalen) == -1)
        return -1;
    if (datalen < w * h)
    {
        PyErr_SetString (PyExc_ValueError, "data buffer does not match size");
        return -1;
    }

    if (PyObject_AsReadBuffer (maskbuf, (const void**)&mask, &masklen) == -1)
        return -1;
    if (masklen < w * h)
    {
        PyErr_SetString (PyExc_ValueError, "mask buffer does not match size");
        return -1;
    }

    cursor = SDL_CreateCursor (data, mask, w, h, hotx, hoty);
    if (!cursor)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return -1;
    }
    
    ((PyCursor*)self)->cursor = cursor;
    return 0;
}

void
cursor_export_capi (void **capi)
{
    capi[PYGAME_SDLCURSOR_FIRSTSLOT] = &PyCursor_Type;
}
