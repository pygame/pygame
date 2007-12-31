/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners

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

  Pete Shinners
  pete@shinners.org
*/
#define NO_PYGAME_C_API
#define PYGAMEAPI_BASE_INTERNAL
#include "pygame.h"
#include "pygamedocs.h"
#include <signal.h>


/* This file controls all the initialization of
 * the module and the various SDL subsystems
 */

/*platform specific init stuff*/

#ifdef MS_WIN32 /*python gives us MS_WIN32*/
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include<windows.h>
extern int SDL_RegisterApp (char*, Uint32, void*);
#endif

#if defined(macintosh)
#if(!defined(__MWERKS__) && !TARGET_API_MAC_CARBON)
QDGlobals qd;
#endif
#endif

static PyObject* quitfunctions = NULL;
static PyObject* PyExc_SDLError;
static void install_parachute (void);
static void uninstall_parachute (void);
static void atexit_quit (void);
static int PyGame_Video_AutoInit (void);
static void PyGame_Video_AutoQuit (void);

static int
CheckSDLVersions (void) /*compare compiled to linked*/
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

void
PyGame_RegisterQuit (void(*func)(void))
{
    PyObject* obj;
    if (!quitfunctions)
    {
        quitfunctions = PyList_New (0);
        if (!quitfunctions)
            return;
    }
    if (func)
    {
        obj = PyCObject_FromVoidPtr (func, NULL);
        PyList_Append (quitfunctions, obj);
        Py_DECREF (obj);
    }
}

static PyObject*
register_quit (PyObject* self, PyObject* arg)
{
    PyObject* quitfunc;

    if (!PyArg_ParseTuple (arg, "O", &quitfunc))
        return NULL;

    if (!quitfunctions)
    {
        quitfunctions = PyList_New (0);
        if (!quitfunctions)
            return NULL;
    }
    PyList_Append (quitfunctions, quitfunc);

    Py_RETURN_NONE;
}

static PyObject*
init (PyObject* self)
{
    PyObject *allmodules, *moduleslist, *dict, *func, *result, *mod;
    int loop, num;
    int success=0, fail=0;

    if (!CheckSDLVersions ())
        return NULL;


    /*nice to initialize timer, so startup time will reflec init() time*/
    SDL_Init (
#if defined(WITH_THREAD) && !defined(MS_WIN32) && defined(SDL_INIT_EVENTTHREAD)
        SDL_INIT_EVENTTHREAD |
#endif
        SDL_INIT_TIMER |
        SDL_INIT_NOPARACHUTE);


    /* initialize all pygame modules */
    allmodules = PyImport_GetModuleDict ();
    moduleslist = PyDict_Values (allmodules);
    if (!allmodules || !moduleslist)
        return Py_BuildValue ("(ii)", 0, 0);

    if (PyGame_Video_AutoInit ())
        ++success;
    else
        ++fail;

    num = PyList_Size (moduleslist);
    for (loop = 0; loop < num; ++loop)
    {
        mod = PyList_GET_ITEM (moduleslist, loop);
        if (!mod || !PyModule_Check (mod))
            continue;
        dict = PyModule_GetDict (mod);
        func = PyDict_GetItemString (dict, "__PYGAMEinit__");
        if(func && PyCallable_Check (func))
        {
            result = PyObject_CallObject (func, NULL);
            if (result && PyObject_IsTrue (result))
                ++success;
            else
            {
                PyErr_Clear ();
                ++fail;
            }
            Py_XDECREF (result);
        }
    }
    Py_DECREF (moduleslist);

    return Py_BuildValue ("(ii)", success, fail);
}

static void
atexit_quit (void)
{
    PyObject* quit;
    PyObject* privatefuncs;
    int num;

    if (!quitfunctions)
        return;

    privatefuncs = quitfunctions;
    quitfunctions = NULL;

    uninstall_parachute ();
    num = PyList_Size (privatefuncs);

    while (num--) /*quit in reverse order*/
    {
        quit = PyList_GET_ITEM (privatefuncs, num);
        if (PyCallable_Check (quit))
            PyObject_CallObject (quit, NULL);
        else if (PyCObject_Check (quit))
        {
            void* ptr = PyCObject_AsVoidPtr (quit);
            (*(void(*)(void)) ptr) ();
        }
    }
    Py_DECREF (privatefuncs);

    PyGame_Video_AutoQuit ();
    SDL_Quit ();
}

static PyObject*
get_sdl_version (PyObject* self)
{
    const SDL_version *v;
	
    v = SDL_Linked_Version ();
    return Py_BuildValue ("iii", v->major, v->minor, v->patch);
}

static PyObject*
get_sdl_byteorder (PyObject *self)
{
    return PyLong_FromLong (SDL_BYTEORDER);
}

static PyObject*
quit (PyObject* self)
{
    atexit_quit ();
    Py_RETURN_NONE;
}

/* internal C API utility functions */
static int
IntFromObj (PyObject* obj, int* val) {
    int tmp_val;
    tmp_val = PyInt_AsLong (obj);
    if (tmp_val == -1 && PyErr_Occurred ())
    {
        PyErr_Clear ();
        return 0;
    }
    *val = tmp_val;
    return 1;
}

static int
IntFromObjIndex (PyObject* obj, int _index, int* val)
{
    int result = 0;
    PyObject* item;
    item = PySequence_GetItem (obj, _index);
    if (item)
    {
        result = IntFromObj (item, val);
        Py_DECREF (item);
    }
    return result;
}

static int
TwoIntsFromObj (PyObject* obj, int* val1, int* val2)
{
    if (PyTuple_Check (obj) && PyTuple_Size (obj) == 1)
        return TwoIntsFromObj (PyTuple_GET_ITEM (obj, 0), val1, val2);

    if (!PySequence_Check (obj) || PySequence_Length (obj) != 2)
        return 0;

    if (!IntFromObjIndex (obj, 0, val1) || !IntFromObjIndex (obj, 1, val2))
        return 0;

    return 1;
}

static int FloatFromObj (PyObject* obj, float* val)
{
    PyObject* floatobj;

    if (PyNumber_Check (obj))
    {
        if (!(floatobj = PyNumber_Float (obj)))
            return 0;
        *val = (float) PyFloat_AsDouble (floatobj);
        Py_DECREF (floatobj);
        return 1;
    }
    return 0;
}

static int
FloatFromObjIndex (PyObject* obj, int _index, float* val)
{
    int result = 0;
    PyObject* item;
    item = PySequence_GetItem (obj, _index);
    if (item)
    {
        result = FloatFromObj (item, val);
        Py_DECREF (item);
    }
    return result;
}

static int
TwoFloatsFromObj (PyObject* obj, float* val1, float* val2)
{
    if (PyTuple_Check (obj) && PyTuple_Size (obj) == 1)
        return TwoFloatsFromObj (PyTuple_GET_ITEM (obj, 0), val1, val2);

    if (!PySequence_Check (obj) || PySequence_Length (obj) != 2)
        return 0;

    if (!FloatFromObjIndex (obj, 0, val1) || !FloatFromObjIndex (obj, 1, val2))
        return 0;

    return 1;
}

static int
UintFromObj (PyObject* obj, Uint32* val)
{
    PyObject* intobj;

    if (PyNumber_Check (obj))
    {
        if (!(intobj = PyNumber_Int (obj)))
            return 0;
        *val = (Uint32) PyInt_AsLong (intobj);
        Py_DECREF (intobj);
        return 1;
    }
    return 0;
}

static Uint32
UintFromObjIndex (PyObject* obj, int _index, Uint32* val)
{
    int result = 0;
    PyObject* item;
    item = PySequence_GetItem (obj, _index);
    if (item)
    {
        result = UintFromObj (item, val);
        Py_DECREF (item);
    }
    return result;
}

static int
RGBAFromObj (PyObject* obj, Uint8* RGBA)
{
    int length;
    Uint32 val;
    if (PyTuple_Check (obj) && PyTuple_Size (obj) == 1)
        return RGBAFromObj (PyTuple_GET_ITEM (obj, 0), RGBA);

    if (!PySequence_Check (obj))
        return 0;

    length = PySequence_Length (obj);
    if (length < 3 || length > 4)
        return 0;

    if (!UintFromObjIndex (obj, 0, &val) || val > 255)
        return 0;
    RGBA[0] = (Uint8) val;
    if (!UintFromObjIndex (obj, 1, &val) || val > 255)
        return 0;
    RGBA[1] = (Uint8) val;
    if (!UintFromObjIndex (obj, 2, &val) || val > 255)
        return 0;
    RGBA[2] = (Uint8) val;
    if (length == 4)
    {
        if (!UintFromObjIndex (obj, 3, &val) || val > 255)
            return 0;
        RGBA[3] = (Uint8) val;
    }
    else RGBA[3] = (Uint8) 255;

    return 1;
}

static PyObject*
get_error (PyObject* self)
{
    return PyString_FromString (SDL_GetError ());
}

/*video init needs to be here, because of it's
 *important init order priority
 */
static void
PyGame_Video_AutoQuit (void)
{
    if (SDL_WasInit (SDL_INIT_VIDEO))
        SDL_QuitSubSystem (SDL_INIT_VIDEO);
}

static int
PyGame_Video_AutoInit (void)
{
    if (!SDL_WasInit (SDL_INIT_VIDEO))
    {
        int status;
#if defined(__APPLE__) && defined(darwin)
        PyObject *module;
        PyObject *rval;
        module = PyImport_ImportModule ("pygame.macosx");
        if (!module)
            return -1;

        rval = PyObject_CallMethod (module, "init", "");
        Py_DECREF (module);
        if (!rval)
            return -1;

        status = PyObject_IsTrue (rval);
        Py_DECREF (rval);
        if (status != 1)
            return 0;
#endif
        status = SDL_InitSubSystem (SDL_INIT_VIDEO);
        if (status)
            return 0;
        SDL_EnableUNICODE (1);
        /*we special case the video quit to last now*/
        /*PyGame_RegisterQuit(PyGame_Video_AutoQuit);*/
    }
    return 1;
}

/*error signal handlers (replacing SDL parachute)*/
static void
pygame_parachute (int sig)
{
    char* signaltype;
    
    signal (sig, SIG_DFL);
    switch (sig)
    {
    case SIGSEGV:
        signaltype = "(pygame parachute) Segmentation Fault";
        break;
#ifdef SIGBUS
#if SIGBUS != SIGSEGV
    case SIGBUS:
        signaltype = "(pygame parachute) Bus Error";
        break;
#endif
#endif
#ifdef SIGFPE
    case SIGFPE:
        signaltype = "(pygame parachute) Floating Point Exception";
        break;
#endif
#ifdef SIGQUIT
    case SIGQUIT:
        signaltype = "(pygame parachute) Keyboard Abort";
        break;
#endif
    default:
        signaltype = "(pygame parachute) Unknown Signal";
        break;
    }

    atexit_quit ();
    Py_FatalError (signaltype);
}

static int fatal_signals[] =
{
    SIGSEGV,
#ifdef SIGBUS
    SIGBUS,
#endif
#ifdef SIGFPE
    SIGFPE,
#endif
#ifdef SIGQUIT
    SIGQUIT,
#endif
    0 /*end of list*/
};

static int parachute_installed = 0;
static void
install_parachute (void)
{
    int i;
    void (*ohandler)(int);

    if (parachute_installed)
        return;
    parachute_installed = 1;

    /* Set a handler for any fatal signal not already handled */
    for (i = 0; fatal_signals[i]; ++i)
    {
        ohandler = signal (fatal_signals[i], pygame_parachute);
        if (ohandler != SIG_DFL)
            signal (fatal_signals[i], ohandler);
    }
#ifdef SIGALRM
    {/* Set SIGALRM to be ignored -- necessary on Solaris */
        struct sigaction action, oaction;
        /* Set SIG_IGN action */
        memset (&action, 0, (sizeof action));
        action.sa_handler = SIG_IGN;
        sigaction (SIGALRM, &action, &oaction);
        /* Reset original action if it was already being handled */
        if (oaction.sa_handler != SIG_DFL)
            sigaction (SIGALRM, &oaction, NULL);
    }
#endif
    return;
}

static void
uninstall_parachute (void)
{
    int i;
    void (*ohandler)(int);

    if (!parachute_installed)
        return;
    parachute_installed = 0;

    /* Remove a handler for any fatal signal handled */
    for (i = 0; fatal_signals[i]; ++i)
    {
        ohandler = signal (fatal_signals[i], SIG_DFL);
        if (ohandler != pygame_parachute)
            signal (fatal_signals[i], ohandler);
    }
}

/* bind functions to python */

static PyObject*
do_segfault (PyObject* self)
{
    //force crash
    *((int*)1) = 45;
    memcpy ((char*)2, (char*)3, 10);
    Py_RETURN_NONE;
}

static PyMethodDef init__builtins__[] =
{
    { "init", (PyCFunction) init, METH_NOARGS, DOC_PYGAMEINIT },
    { "quit", (PyCFunction) quit, METH_NOARGS, DOC_PYGAMEQUIT },
    { "register_quit", register_quit, METH_VARARGS, DOC_PYGAMEREGISTERQUIT },
    { "get_error", (PyCFunction) get_error, METH_NOARGS, DOC_PYGAMEGETERROR },
    { "get_sdl_version", (PyCFunction) get_sdl_version, METH_NOARGS,
      DOC_PYGAMEGETSDLVERSION },
    { "get_sdl_byteorder", (PyCFunction) get_sdl_byteorder, METH_NOARGS,
      DOC_PYGAMEGETSDLBYTEORDER },

    { "segfault", (PyCFunction) do_segfault, METH_NOARGS, "crash" },
    { NULL, NULL, 0, NULL }
};

PYGAME_EXPORT
void initbase (void)
{
    PyObject *module, *dict, *apiobj;
    static void* c_api[PYGAMEAPI_BASE_NUMSLOTS];

    /* create the module */
    module = Py_InitModule3 ("base", init__builtins__, DOC_PYGAME);
    dict = PyModule_GetDict (module);

    /* create the exceptions */
    PyExc_SDLError = PyErr_NewException ("pygame.error", PyExc_RuntimeError,
                                         NULL);
    PyDict_SetItemString (dict, "error", PyExc_SDLError);
    Py_DECREF (PyExc_SDLError);

    /* export the c api */
    c_api[0] = PyExc_SDLError;
    c_api[1] = PyGame_RegisterQuit;
    c_api[2] = IntFromObj;
    c_api[3] = IntFromObjIndex;
    c_api[4] = TwoIntsFromObj;
    c_api[5] = FloatFromObj;
    c_api[6] = FloatFromObjIndex;
    c_api[7] = TwoFloatsFromObj;
    c_api[8] = UintFromObj;
    c_api[9] = UintFromObjIndex;
    c_api[10] = PyGame_Video_AutoQuit;
    c_api[11] = PyGame_Video_AutoInit;
    c_api[12] = RGBAFromObj;
    apiobj = PyCObject_FromVoidPtr (c_api, NULL);
    PyDict_SetItemString (dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF (apiobj);

    /*some intiialization*/
    Py_AtExit (atexit_quit);
    install_parachute ();
#ifdef MS_WIN32
    SDL_RegisterApp ("pygame", 0, GetModuleHandle (NULL));
#endif
#if defined(macintosh)
#if(!defined(__MWERKS__) && !TARGET_API_MAC_CARBON)
    SDL_InitQuickDraw (&qd);
#endif
#endif
}
