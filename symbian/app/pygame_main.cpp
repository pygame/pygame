/** Creates Python interpreter and launches pygame's main script */

#include <sdl.h>
#include <Python.h>

#if PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION == 2
#include <CSPyInterpreter.h>
#endif

#include "logmanutils.h"

extern "C" struct _inittab _PyGame_Inittab[];

#define NL "\n"

#ifndef PYGAME_MAIN_SCRIPT_PATH
#define PYGAME_MAIN_SCRIPT_PATH "\\data\\pygame\\pygame_main.py"
#define PYGAME_LAUNCHER_PATH    "\\data\\pygame\\launcher\\pygame_launcher.py"
#endif
static const char* gPygameMainScriptPath[2] = {
		PYGAME_MAIN_SCRIPT_PATH,
		PYGAME_LAUNCHER_PATH
		};

int main(int argc, char** argv)
{
	// Execute the main script
#if PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION == 2
	CSPyInterpreter* interp = CSPyInterpreter::NewInterpreterL();
#else
	SPy_DLC_Init();
	SPy_SetAllocator(SPy_DLC_Alloc, SPy_DLC_Realloc, SPy_DLC_Free, NULL);
#endif
	Py_Initialize();

	// Add built-in pygame modules
#ifdef HAVE_STATIC_MODULES
	PyImport_ExtendInittab(_PyGame_Inittab);
#endif

	LOGMAN_SENDLOG( "Entering interpreter");
#if PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION == 2
	TInt result = interp->RunScript(1, gPygameMainScriptPath);
#else

	LOGMAN_SENDLOGF("Number of args:%d", argc);
	if(argc > 1){
		gPygameMainScriptPath[1] = argv[1];
	}

	LOGMAN_SENDLOGF8( "Opening file:%s", gPygameMainScriptPath[0] );
	FILE *fp = fopen(gPygameMainScriptPath[0], "r");
	if (!fp) {
		LOGMAN_SENDLOG( "Failed to open main script" );
		return -1;
	}

	// This allows us to retrieve the path of the main script in Python from sys.argv[0]
	PySys_SetArgv(2, (char**)gPygameMainScriptPath);

	int result = PyRun_SimpleFile(fp, gPygameMainScriptPath[0]);
	fclose(fp);
#endif

	LOGMAN_SENDLOGF( "Interpreter result:%d", result )

#if PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION == 2
	PyEval_RestoreThread(PYTHON_TLS->thread_state);
	Py_Finalize();
	delete interp;
#else
	Py_Finalize();
	SPy_DLC_Fini();
#endif

	SDL_Quit();
	return result;
}
