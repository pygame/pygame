#include <sdl.h>
#include <Python.h>
#include <CSPyInterpreter.h>

#include "logmanutils.h"

extern int Frames;
extern int Ticks;
extern int Done;

void panic(char* aWhen)
{
	fprintf(stderr, "SDL error: %s: %s\n", aWhen, SDL_GetError());
	SDL_Delay(1000);
	SDL_Quit();
	exit(-1);
}

extern "C"
struct _inittab _PyGame_Inittab[];

#define NL "\n"

char* PYGAME_MAIN_SCRIPT_PATH[1] = {"\\data\\pygame\\pygame_main.py"};

int main(int argc, char** argv)
{
	// Execute the main script

	CSPyInterpreter* interp = CSPyInterpreter::NewInterpreterL();

	// Add built-in pygame modules
	PyImport_ExtendInittab(_PyGame_Inittab);
			
	LOGMAN_SENDLOG( "Entering interpreter");
	TInt result = interp->RunScript(1, PYGAME_MAIN_SCRIPT_PATH);
	LOGMAN_SENDLOGF( "Interpreter result:%d", result )
	
	PyEval_RestoreThread(PYTHON_TLS->thread_state);
	Py_Finalize();
	
	delete interp;

	SDL_Quit();
	return result;
}
