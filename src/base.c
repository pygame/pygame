/*
    pygame - Python Game Library
    Copyright (C) 2000  Pete Shinners

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

#define PYGAMEAPI_BASE_INTERNAL
#include "pygame.h"
#include <signal.h>


/* This file controls all the initialization of
 * the module and the various SDL subsystems
*/

#ifdef MS_WIN32 /*python gives us MS_WIN32*/
#define WIN32_LEAN_AND_MEAN
#include<windows.h>
extern int SDL_RegisterApp(char*, Uint32, void*);
#endif


extern int pygame_video_priority_init(void);

static PyObject* quitfunctions = NULL;
static PyObject* PyExc_SDLError;


static int PyGame_Video_AutoInit(void);




static int CheckSDLVersions(void) /*compare compiled to linked*/
{
	SDL_version compiled;
	const SDL_version* linked;
	SDL_VERSION(&compiled);
	linked = SDL_Linked_Version();

	if(compiled.major != linked->major ||
				compiled.minor != linked->minor ||
				compiled.patch != linked->patch)
	{
		char err[1024];
		sprintf(err, "SDL compiled with version %d.%d.%d, linked to %d.%d.%d",
					compiled.major, compiled.minor, compiled.patch,
					linked->major, linked->minor, linked->patch);
		PyErr_SetString(PyExc_RuntimeError, err);
		return 0;
	}
	return 1;
}





void PyGame_RegisterQuit(void(*func)(void))
{
	PyObject* obj;
	if(!quitfunctions)
	{
		quitfunctions = PyList_New(0);
		if(!quitfunctions) return;
	}
	if(func)
	{
		obj = PyCObject_FromVoidPtr(func, NULL);
		PyList_Append(quitfunctions, obj);
		/*Py_DECREF(obj);*/
	}
}

    /*DOC*/ static char doc_register_quit[] =
    /*DOC*/    "pygame.register_quit(callback) -> None\n"
    /*DOC*/    "routine to call when pyGame quits\n"
    /*DOC*/    "\n"
    /*DOC*/    "The given callback routine will be called when. pygame is\n"
    /*DOC*/    "quitting. Quit callbacks are served on a 'last in, first out'\n"
    /*DOC*/    "basis. Also be aware that your callback may be called more than\n"
    /*DOC*/    "once.\n"
    /*DOC*/ ;

static PyObject* register_quit(PyObject* self, PyObject* arg)
{
	PyObject* quitfunc;

	if(!PyArg_ParseTuple(arg, "O", &quitfunc))
		return NULL;

	if(!quitfunctions)
	{
		quitfunctions = PyList_New(0);
		if(!quitfunctions) return NULL;
	}
	if(quitfunc)
		PyList_Append(quitfunctions, quitfunc);

	Py_INCREF(Py_None);
	return Py_None;
}


    /*DOC*/ static char doc_init[] =
    /*DOC*/    "pygame.init() -> passed, failed\n"
    /*DOC*/    "autoinitialize all imported pygame modules\n"
    /*DOC*/    "\n"
    /*DOC*/    "Initialize all imported pygame modules. Including pygame modules\n"
    /*DOC*/    "that are not part of the base modules (like font and image).\n"
    /*DOC*/    "\n"
    /*DOC*/    "It does not raise exceptions, but instead silently counts which\n"
    /*DOC*/    "modules have failed to init. The return argument contains a count\n"
    /*DOC*/    "of the number of modules initialized, and the number of modules\n"
    /*DOC*/    "that failed to initialize.\n"
    /*DOC*/    "\n"
    /*DOC*/    "You can always initialize the modules you want by hand. The\n"
    /*DOC*/    "modules that need it have an init() and quit() routine built in,\n"
    /*DOC*/    "which you can call directly. They also have a get_init() routine\n"
    /*DOC*/    "which you can use to doublecheck the initialization. Note that\n"
    /*DOC*/    "the manual init() routines will raise an exception on error. Be\n"
    /*DOC*/    "aware that most platforms require the display module to be\n"
    /*DOC*/    "initialized before others. This init() will handle that for you,\n"
    /*DOC*/    "but if you initialize by hand, be aware of this constraint.\n"
    /*DOC*/    "\n"
    /*DOC*/    "As with the manual init() routines. It is safe to call this\n"
    /*DOC*/    "init() as often as you like. If you have imported pygame modules\n"
    /*DOC*/    "since the.\n"
    /*DOC*/ ;

static PyObject* init(PyObject* self,PyObject* args)
{
	PyObject *allmodules, *moduleslist, *dict, *func, *result, *mod;
	int loop, num;
	int success=0, fail=0;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;
	if(!CheckSDLVersions())
		return NULL;

	allmodules = PyImport_GetModuleDict();
	moduleslist = PyDict_Values(allmodules);
	if(!allmodules || !moduleslist) 
		return Py_BuildValue("(ii)", 0, 0);

	if(PyGame_Video_AutoInit())
		++success;
	else
		++fail;

	num = PyList_Size(moduleslist);
	for(loop = 0; loop < num; ++loop)
	{
		mod = PyList_GET_ITEM(moduleslist, loop);
		if(!mod || !PyModule_Check(mod))
			continue;
		dict = PyModule_GetDict(mod);
		func = PyDict_GetItemString(dict, "__PYGAMEinit__");
		if(func && PyCallable_Check(func))
		{
			result = PyObject_CallObject(func, NULL);
			if(result && PyObject_IsTrue(result))
				++success;
			else
			{
				PyErr_Clear();
				++fail;
			}
			Py_XDECREF(result);
		}
	}
	Py_DECREF(moduleslist);

	return Py_BuildValue("(ii)", success, fail);
}


static void atexit_quit(void)
{
	PyObject* quit;
	int num;

	SDL_QuitSubSystem(SDL_INIT_TIMER);

	if(!quitfunctions)
		return;

	num = PyList_Size(quitfunctions);
	while(num--) /*quit in reverse order*/
	{
		quit = PyList_GET_ITEM(quitfunctions, num);
		if(PyCallable_Check(quit))
			PyObject_CallObject(quit, NULL);
		else if(PyCObject_Check(quit))
		{
			void* ptr = PyCObject_AsVoidPtr(quit);
			(*(void(*)(void))ptr)();
		}
	}

	Py_DECREF(quitfunctions);
	quitfunctions = NULL;
}


    /*DOC*/ static char doc_quit[] =
    /*DOC*/    "pygame.quit() -> none\n"
    /*DOC*/    "uninitialize all pygame modules\n"
    /*DOC*/    "\n"
    /*DOC*/    "Uninitialize all pygame modules that have been initialized. Even\n"
    /*DOC*/    "if you initialized the module by hand, this quit() will\n"
    /*DOC*/    "uninitialize it for you.\n"
    /*DOC*/    "\n"
    /*DOC*/    "All the pygame modules are uninitialized automatically when your\n"
    /*DOC*/    "program exits, so you will usually not need this routine. If you\n"
    /*DOC*/    "program plans to keep running after it is done with pygame, then\n"
    /*DOC*/    "would be a good time to make this call.\n"
    /*DOC*/ ;

static PyObject* quit(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	atexit_quit();

	Py_INCREF(Py_None);
	return Py_None;
}





/* internal C API utility functions */
static int ShortFromObj(PyObject* obj, short* val)
{
	PyObject* intobj;

	if(PyNumber_Check(obj))
	{
		if(!(intobj = PyNumber_Int(obj)))
			return 0;
		*val = (short)PyInt_AsLong(intobj);
		Py_DECREF(intobj);
		return 1;
	}
	return 0;
}

static int ShortFromObjIndex(PyObject* obj, int index, short* val)
{
	int result = 0;
	PyObject* item;
	item = PySequence_GetItem(obj, index);
	if(item)
	{
		result = ShortFromObj(item, val);
		Py_DECREF(item);
	}
	return result;
}

static int TwoShortsFromObj(PyObject* obj, short* val1, short* val2)
{
	if(PyTuple_Check(obj) && PyTuple_Size(obj)==1)
		return TwoShortsFromObj(PyTuple_GET_ITEM(obj, 0), val1, val2);

	if(!PySequence_Check(obj) || PySequence_Length(obj) != 2)
		return 0;

	if(!ShortFromObjIndex(obj, 0, val1) || !ShortFromObjIndex(obj, 1, val2))
		return 0;

	return 1;
}

static int UintFromObj(PyObject* obj, Uint32* val)
{
	PyObject* intobj;

	if(PyNumber_Check(obj))
	{
		if(!(intobj = PyNumber_Int(obj)))
			return 0;
		*val = (Uint32)PyInt_AsLong(intobj);
		Py_DECREF(intobj);
		return 1;
	}
	return 0;
}

static Uint32 UintFromObjIndex(PyObject* obj, int index, Uint32* val)
{
	int result = 0;
	PyObject* item;
	item = PySequence_GetItem(obj, index);
	if(item)
	{
		result = UintFromObj(item, val);
		Py_DECREF(item);
	}
	return result;
}

static int RGBAFromObj(PyObject* obj, Uint8* RGBA)
{
	int length;
	Uint32 val;
	if(PyTuple_Check(obj) && PyTuple_Size(obj)==1)
		return RGBAFromObj(PyTuple_GET_ITEM(obj, 0), RGBA);

	if(!PySequence_Check(obj))
		return 0;

	length = PySequence_Length(obj);
	if(length < 3 || length > 4)
		return 0;

	if(!UintFromObjIndex(obj, 0, &val) || val > 255)
		return 0;
	RGBA[0] = (Uint8)val;
	if(!UintFromObjIndex(obj, 1, &val) || val > 255)
		return 0;
	RGBA[1] = (Uint8)val;
	if(!UintFromObjIndex(obj, 2, &val) || val > 255)
		return 0;
	RGBA[2] = (Uint8)val;
	if(length == 4)
	{
		if(!UintFromObjIndex(obj, 3, &val) || val > 255)
			return 0;
		RGBA[3] = (Uint8)val;
	}
	else RGBA[3] = (Uint8)255;

	return 1;
}



    /*DOC*/ static char doc_get_error[] =
    /*DOC*/    "pygame.get_error() -> errorstring\n"
    /*DOC*/    "get current error message\n"
    /*DOC*/    "\n"
    /*DOC*/    "SDL maintains an internal current error message. This message is\n"
    /*DOC*/    "usually given to you when an SDL related exception occurs, but\n"
    /*DOC*/    "sometimes you may want to call this directly yourself.\n"
    /*DOC*/ ;

static PyObject* get_error(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;
	return PyString_FromString(SDL_GetError());
}




/*video init needs to be here, because of it's
 *important init order priority
 */
static void PyGame_Video_AutoQuit(void)
{
	if(SDL_WasInit(SDL_INIT_VIDEO))
		SDL_QuitSubSystem(SDL_INIT_VIDEO);
}

static int PyGame_Video_AutoInit(void)
{
	if(!SDL_WasInit(SDL_INIT_VIDEO))
	{
		int status;

		Py_BEGIN_ALLOW_THREADS
		status = SDL_InitSubSystem(SDL_INIT_VIDEO);
		Py_END_ALLOW_THREADS
		
		if(status)
			return 0;
		SDL_EnableUNICODE(1);
		PyGame_RegisterQuit(PyGame_Video_AutoQuit);
	}
	return 1;
}



/*error signal handlers (replacing SDL parachute)*/
static void pygame_parachute(int sig)
{
	char* signaltype = "Unknown Signal";

	signal(sig, SIG_DFL);
	switch (sig)
	{
		case SIGSEGV:
			signaltype = "Segmentation Fault"; break;
#ifdef SIGBUS
#if SIGBUS != SIGSEGV
		case SIGBUS:
			signaltype = "Bus Error"; break;
#endif
#endif
#ifdef SIGFPE
		case SIGFPE:
			signaltype = "Floating Point Exception"; break;
#endif /* SIGFPE */
#ifdef SIGQUIT
		case SIGQUIT:
			signaltype = "Keyboard Quit"; break;
#endif /* SIGQUIT */
#ifdef SIGPIPE
		case SIGPIPE:
			signaltype = "Broken Pipe"; break;
#endif /* SIGPIPE */
		default:
			signaltype = "# %d", sig; break;
	}

#if 0
/*try to print traceback, ARGH*/
	tstate = PyThreadState_GET();
	if(tstate && PyTraceBack_Here(tstate->frame) != -1)
		PyObject_Print(tstate->exc_traceback, stderr, Py_PRINT_RAW);
#endif

	atexit_quit();
	Py_FatalError(signaltype);
}


static void install_parachute(void)
{
	int i;
	void (*ohandler)(int);
	int fatal_signals[] =
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
#ifdef SIGPIPE
		SIGPIPE,
#endif
		0 /*end of list*/
	};

	/* Set a handler for any fatal signal not already handled */
	for ( i=0; fatal_signals[i]; ++i )
	{
		ohandler = signal(fatal_signals[i], pygame_parachute);
		if ( ohandler != SIG_DFL )
			signal(fatal_signals[i], ohandler);
	}
#ifdef SIGALRM
	/* Set SIGALRM to be ignored -- necessary on Solaris */
	{
		struct sigaction action, oaction;

		/* Set SIG_IGN action */
		memset(&action, 0, (sizeof action));
		action.sa_handler = SIG_IGN;
		sigaction(SIGALRM, &action, &oaction);

		/* Reset original action if it was already being handled */
		if ( oaction.sa_handler != SIG_DFL ) {
			sigaction(SIGALRM, &oaction, NULL);
		}
	}
#endif
	return;
}






/* bind functions to python */


static PyMethodDef init__builtins__[] =
{
	{ "init", init, 1, doc_init }, 
	{ "quit", quit, 1, doc_quit },
	{ "register_quit", register_quit, 1, doc_register_quit },
	{ "get_error", get_error, 1, doc_get_error },
	{ NULL, NULL }
};


    /*DOC*/ static char doc_pygame_MODULE[] =
    /*DOC*/    "Contains the core routines that are used by the rest of the\n"
    /*DOC*/    "pygame modules. It's routines are merged directly into the pygame\n"
    /*DOC*/    "namespace. This mainly includes the auto-initialization init() and\n"
    /*DOC*/    "quit() routines.\n"
    /*DOC*/    "\n"
    /*DOC*/    "There is a small module named 'locals' that also gets merged into\n"
    /*DOC*/    "this namespace. This contains all the constants needed by pygame.\n"
    /*DOC*/    "Object constructors also get placed into this namespace, you can\n"
    /*DOC*/    "call functions like rect() and surface() to create objects of\n"
    /*DOC*/    "that type. As a convenience, you can import the members of\n"
    /*DOC*/    "pygame.locals directly into your module's namespace with 'from\n"
    /*DOC*/    "pygame.locals import *'. Most of the pygame examples do this if\n"
    /*DOC*/    "you'd like to take a look.\n"
    /*DOC*/ ;

void initbase(void)
{
	static int initialized_once = 0;
	PyObject *module, *dict, *apiobj;
	static void* c_api[PYGAMEAPI_BASE_NUMSLOTS];

    /* create the module */
	module = Py_InitModule3("base", init__builtins__, doc_pygame_MODULE);
	dict = PyModule_GetDict(module);

	/* create the exceptions */
	PyExc_SDLError = PyErr_NewException("pygame.error", PyExc_RuntimeError, NULL);
	PyDict_SetItemString(dict, "error", PyExc_SDLError);

	/* export the c api */
	c_api[0] = PyExc_SDLError;
	c_api[1] = PyGame_RegisterQuit;
	c_api[2] = ShortFromObj;
	c_api[3] = ShortFromObjIndex;
	c_api[4] = TwoShortsFromObj;
	c_api[5] = UintFromObj;
	c_api[6] = UintFromObjIndex;
	c_api[7] = PyGame_Video_AutoQuit;
	c_api[8] = PyGame_Video_AutoInit;
	c_api[9] = RGBAFromObj;
	apiobj = PyCObject_FromVoidPtr(c_api, NULL);
	PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);


/* let SDL do some basic initialization */
	if(!initialized_once)
	{
#ifdef MS_WIN32
		SDL_RegisterApp("pygame window", 0, GetModuleHandle(NULL));
#endif
		/*nice to initialize timer, so startup time will be correct before init call*/
		SDL_Init(SDL_INIT_TIMER|SDL_INIT_NOPARACHUTE);
		initialized_once = 1;
		Py_AtExit(atexit_quit);
		install_parachute();
	}

	/*touch PyGAME_C_API to keep compiler from warning*/
	PyGAME_C_API[0] = PyGAME_C_API[0];
}



#if 0 /*only for documentation*/
    /*DOC*/ static char doc_misc_MODULE[] =
    /*DOC*/    "Contains functions that weren't categorized correctly. Usually a\n"
    /*DOC*/    "problem with the documentation or documentation generation code\n"
    /*DOC*/    ":]\n"
    /*DOC*/ ;
#endif
