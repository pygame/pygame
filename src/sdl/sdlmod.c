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
#define PYGAME_SDLBASE_INTERNAL

#include "pgsdl.h"
#include "sdlbase_doc.h"

static int _quitcount = 0;
static quit_func *_quitcallbacks = NULL;
static int _register_quitcallback (quit_func func);
static void _quit (void);
static int _check_sdl (void);

static PyObject* _sdl_init (PyObject *self, PyObject *args);
static PyObject* _sdl_quit (PyObject *self);
static PyObject* _sdl_initsubsystem (PyObject *self, PyObject *args);
static PyObject* _sdl_quitsubsystem (PyObject *self, PyObject *args);
static PyObject* _sdl_wasinit (PyObject *self, PyObject *args);
static PyObject* _sdl_geterror (PyObject *self);
static PyObject* _sdl_getcompiledversion (PyObject *self);
static PyObject* _sdl_getversion (PyObject *self);

static PyMethodDef _sdl_methods[] = {
    { "init", _sdl_init, METH_VARARGS, DOC_BASE_INIT },
    { "quit", (PyCFunction) _sdl_quit, METH_NOARGS, DOC_BASE_QUIT },
    { "was_init", _sdl_wasinit, METH_VARARGS, DOC_BASE_WAS_INIT },
    { "init_subsystem", _sdl_initsubsystem, METH_VARARGS,
      DOC_BASE_INIT_SUBSYSTEM },
    { "quit_subsystem", _sdl_quitsubsystem, METH_VARARGS,
      DOC_BASE_QUIT_SUBSYSTEM }, 
    { "get_error", (PyCFunction)_sdl_geterror, METH_NOARGS,
      DOC_BASE_GET_ERROR },
    { "get_compiled_version", (PyCFunction) _sdl_getcompiledversion,
      METH_NOARGS, DOC_BASE_GET_COMPILED_VERSION },
    { "get_version", (PyCFunction) _sdl_getversion, METH_NOARGS,
      DOC_BASE_GET_VERSION },
    { NULL, NULL, 0, NULL },
};

static int
_register_quitcallback (quit_func func)
{
    if (_quitcount == 0)
    {
        _quitcallbacks = malloc (sizeof (quit_func));
        if (!_quitcallbacks)
            return 0;
    }
    else
    {
        void *tmp = _quitcallbacks;
        tmp = realloc (_quitcallbacks, sizeof (quit_func) * (_quitcount + 1));
        if (!tmp)
            return 0;
        _quitcallbacks = tmp;
    }
    _quitcallbacks[_quitcount] = func;
    _quitcount++;
    return 1;
}

static void
_quit (void)
{
    int i;
    for (i = 0; i < _quitcount; i++)
        (*_quitcallbacks[i])();
    SDL_Quit ();
}

static int
_check_sdl (void)
{
    SDL_version compiled;
    const SDL_version* linked;
    SDL_VERSION (&compiled);
    linked = SDL_Linked_Version ();

    /*only check the major and minor version numbers.
      we will relax any differences in 'patch' version.*/

    if (compiled.major != linked->major || compiled.minor != linked->minor)
    {
        char err[1024];
        sprintf (err, "SDL compiled with version %d.%d.%d, linked to %d.%d.%d",
            compiled.major, compiled.minor, compiled.patch,
            linked->major, linked->minor, linked->patch);
        PyErr_SetString (PyExc_RuntimeError, err);
        return 0;
    }
    return 1;
}

static PyObject*
_sdl_init (PyObject *self, PyObject *args)
{
    Uint32 flags;
    
    if (!_check_sdl ())
        return NULL;

    if (!PyArg_ParseTuple (args, "l:init", &flags))
        return NULL;
    if (SDL_Init (flags) == -1)
        Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}

static PyObject*
_sdl_quit (PyObject *self)
{
    _quit ();
    Py_RETURN_TRUE;
}

static PyObject*
_sdl_initsubsystem (PyObject *self, PyObject *args)
{
    Uint32 flags;
    if (!PyArg_ParseTuple (args, "l:init_subsystem", &flags))
        return NULL;
    if (SDL_InitSubSystem (flags) == -1)
        Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}

static PyObject*
_sdl_quitsubsystem (PyObject *self, PyObject *args)
{
    Uint32 flags;
    if (!PyArg_ParseTuple (args, "l:quit_subsystem", &flags))
        return NULL;
    SDL_QuitSubSystem (flags);
    Py_RETURN_NONE;
}

static PyObject*
_sdl_wasinit (PyObject *self, PyObject *args)
{
    Uint32 flags, retval;

    if (!PyArg_ParseTuple (args, "l:was_init", &flags))
        return NULL;
    retval = SDL_WasInit (flags);
    return PyLong_FromLong ((long)retval);
}

static PyObject*
_sdl_geterror (PyObject *self)
{
    char *err = SDL_GetError ();
    if (!err)
        Py_RETURN_NONE;
    return Text_FromUTF8 (err);
}

static PyObject*
_sdl_getcompiledversion (PyObject *self)
{
    SDL_version compiled;
    SDL_VERSION (&compiled);
    return Py_BuildValue ("(iii)", compiled.major, compiled.minor,
        compiled.patch);
}

static PyObject*
_sdl_getversion (PyObject *self)
{
    const SDL_version *linked = SDL_Linked_Version ();
    return Py_BuildValue ("(iii)", linked->major, linked->minor, linked->patch);
}

/* C API */
static int
Uint8FromObj (PyObject *item, Uint8 *val)
{
    PyObject* intobj;
    long tmp;
    
    if (PyNumber_Check (item))
    {
        if (!(intobj = PyNumber_Int (item)))
            return 0;
        tmp = PyInt_AsLong (intobj);
        Py_DECREF (intobj);
        if (tmp == -1 && PyErr_Occurred ())
            return 0;
        if (tmp < 0)
        {
            PyErr_SetString (PyExc_ValueError, "value must not be negative");
            return 0;
        }
        *val = (Uint8)tmp;
        return 1;
    }
    return 0;
}

static int
Uint16FromObj (PyObject *item, Uint16 *val)
{
    PyObject* intobj;
    long tmp;
    
    if (PyNumber_Check (item))
    {
        if (!(intobj = PyNumber_Int (item)))
            return 0;
        tmp = PyInt_AsLong (intobj);
        Py_DECREF (intobj);
        if (tmp == -1 && PyErr_Occurred ())
            return 0;
        if (tmp < 0)
        {
            PyErr_SetString (PyExc_ValueError, "value must not be negative");
            return 0;
        }
        *val = (Uint16)tmp;
        return 1;
    }
    return 0;
}

static int
Sint16FromObj (PyObject *item, Sint16 *val)
{
    PyObject* intobj;
    long tmp;
    
    if (PyNumber_Check (item))
    {
        if (!(intobj = PyNumber_Int (item)))
            return 0;
        tmp = PyInt_AsLong (intobj);
        Py_DECREF (intobj);
        if (tmp == -1 && PyErr_Occurred ())
            return 0;
        *val = (Sint16)tmp;
        return 1;
    }
    return 0;
}

static int
Uint32FromObj (PyObject *item, Uint32 *val)
{
    PyObject* longobj;
    unsigned long tmp;
    
    if (PyNumber_Check (item))
    {
        if (!(longobj = PyNumber_Long (item)))
            return 0;
        tmp = PyLong_AsUnsignedLong (longobj);
        Py_DECREF (longobj);
        if (PyErr_Occurred ())
            return 0;
        *val = (Uint32)tmp;
        return 1;
    }
    return 0;
}

static int
Uint8FromSeqIndex (PyObject* obj, Py_ssize_t _index, Uint8* val)
{
    int result = 0;
    PyObject* item;
    item = PySequence_GetItem (obj, _index);
    if (item)
    {
        result = Uint8FromObj (item, val);
        Py_DECREF (item);
    }
    return result;
}

static int
Uint16FromSeqIndex (PyObject* obj, Py_ssize_t _index, Uint16* val)
{
    int result = 0;
    PyObject* item;
    item = PySequence_GetItem (obj, _index);
    if (item)
    {
        result = Uint16FromObj (item, val);
        Py_DECREF (item);
    }
    return result;
}

static int
Sint16FromSeqIndex (PyObject* obj, Py_ssize_t _index, Sint16* val)
{
    int result = 0;
    PyObject* item;
    item = PySequence_GetItem (obj, _index);
    if (item)
    {
        result = Sint16FromObj (item, val);
        Py_DECREF (item);
    }
    return result;
}

static int
Uint32FromSeqIndex (PyObject* obj, Py_ssize_t _index, Uint32* val)
{
    int result = 0;
    PyObject* item;
    item = PySequence_GetItem (obj, _index);
    if (item)
    {
        result = Uint32FromObj (item, val);
        Py_DECREF (item);
    }
    return result;
}

static int
IsValidRect (PyObject* rect)
{
    if (PyRect_Check (rect) || PyFRect_Check (rect))
        return 1;
    else if (PySequence_Check (rect) && PySequence_Size (rect) == 4)
    {
        Sint16 x, y;
        Uint16 w, h;
        if (!Sint16FromSeqIndex (rect, 0, &x))
            goto failed;
        if (!Sint16FromSeqIndex (rect, 1, &y))
            goto failed;
        if (!Uint16FromSeqIndex (rect, 2, &w))
            goto failed;
        if (!Uint16FromSeqIndex (rect, 3, &h))
            goto failed;
        return 1;
    }
failed:
    PyErr_Clear ();
    PyErr_SetString (PyExc_TypeError,
        "rect must be a Rect, FRect or 4-value sequence");
    return 0;
}

static int
SDLRect_FromRect (PyObject* rect, SDL_Rect *sdlrect)
{
    if (PyRect_Check (rect))
    {
        sdlrect->x = (Sint16) ((PyRect*)rect)->x;
        sdlrect->y = (Sint16) ((PyRect*)rect)->y;
        sdlrect->w = (Uint32) ((PyRect*)rect)->w;
        sdlrect->h = (Uint32) ((PyRect*)rect)->h;
        return 1;
    }
    else if (PyFRect_Check (rect))
    {
        sdlrect->x = (Sint16) round (((PyFRect*)rect)->x);
        sdlrect->y = (Sint16) round (((PyFRect*)rect)->y);
        sdlrect->w = (Uint32) round (((PyFRect*)rect)->w);
        sdlrect->h = (Uint32) round (((PyFRect*)rect)->h);
        return 1;
    }
    else if (PySequence_Check (rect) && PySequence_Size (rect) == 4)
    {
        if (!Sint16FromSeqIndex (rect, 0, &(sdlrect->x)))
            goto failed;
        if (!Sint16FromSeqIndex (rect, 1, &(sdlrect->y)))
            goto failed;
        if (!Uint16FromSeqIndex (rect, 2, &(sdlrect->w)))
            goto failed;
        if (!Uint16FromSeqIndex (rect, 3, &(sdlrect->h)))
            goto failed;
        return 1;
    }
failed:
    PyErr_Clear ();
    PyErr_SetString (PyExc_TypeError,
        "rect must be a Rect, FRect or 4-value sequence");
    return 0;
}

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_base (void)
#else
PyMODINIT_FUNC initbase (void)
#endif
{
    PyObject *mod;
    PyObject *c_api_obj;
    static void *c_api[PYGAME_SDLBASE_SLOTS];

#ifdef IS_PYTHON_3
    static struct PyModuleDef _sdlmodule = {
        PyModuleDef_HEAD_INIT,
        "base",
        DOC_BASE,
        -1,
        _sdl_methods,
        NULL, NULL, NULL, NULL
    };
    mod = PyModule_Create (&_sdlmodule);
#else
    mod = Py_InitModule3 ("base", _sdl_methods, DOC_BASE);
#endif
    if (!mod)
        goto fail;
    
    /* Export C API */
    c_api[PYGAME_SDLBASE_FIRSTSLOT] = _register_quitcallback;
    c_api[PYGAME_SDLBASE_FIRSTSLOT+1] = Uint8FromObj;
    c_api[PYGAME_SDLBASE_FIRSTSLOT+2] = Uint16FromObj;
    c_api[PYGAME_SDLBASE_FIRSTSLOT+3] = Sint16FromObj;
    c_api[PYGAME_SDLBASE_FIRSTSLOT+4] = Uint32FromObj;
    c_api[PYGAME_SDLBASE_FIRSTSLOT+5] = Uint8FromSeqIndex;
    c_api[PYGAME_SDLBASE_FIRSTSLOT+6] = Uint16FromSeqIndex;
    c_api[PYGAME_SDLBASE_FIRSTSLOT+7] = Sint16FromSeqIndex;
    c_api[PYGAME_SDLBASE_FIRSTSLOT+8] = Uint32FromSeqIndex;
    c_api[PYGAME_SDLBASE_FIRSTSLOT+9] = IsValidRect;
    c_api[PYGAME_SDLBASE_FIRSTSLOT+10] = SDLRect_FromRect;
   
    c_api_obj = PyCObject_FromVoidPtr ((void *) c_api, NULL);
    if (c_api_obj)
        PyModule_AddObject (mod, PYGAME_SDLBASE_ENTRY, c_api_obj);    

    Py_AtExit (_quit);
    if (import_pygame2_base () < 0)
        goto fail;

    MODINIT_RETURN (mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
