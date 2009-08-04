#ifndef _GCOMMAND_H_
#define _GCOMMAND_H_
#include <SDL.h>
#include <SDL_thread.h>
#include <Python.h>

/* Documentation: Command Queue Infrastructure
 *  This lower-level infrastructure code is meant to provide greater stability and 
 *  thread safety to the _movie module. Since we cannot manipulate the SDL event queue
 *  we have to use our own hand-rolled solution. It is just a singly linked list, 
 *  with references to the first and last item, allowing us to do a
 *  simple push/pop implementation. The items in the list are structs that have
 *  all the first members of the default Command struct, making it safe to 
 *  cast the pointers from the pseudo-Command structs to a pointer to 
 *  Command struct. Realistically, you can cast any pointer to any other kind of
 *  pointer(as long as they are the same size!), and C will let you. This is dangerous,
 *  and should only be done very, very carefully. This facility is only useful 
 *  when you need a OO approach, like we did here.
 *  
 *  When making new commands, use the FULL_COMMAND macro, and add a line to registerCommands 
 *  in _gmovie.c to add a new type value. This also enables future proofing as any changes to 
 *  the Command struct will be opaque to the user... mostly.
 * 
 *  -Tyler Laing, August 4th, 2009
 */

typedef struct Command
{
	int type;
	struct Command *next;
} Command;

#define FULL_COMMAND \
	int type;\
	Command *next;

typedef struct CommandQueue
{
	int size;
	SDL_mutex *q_mutex;
	Command *first;
	Command *last;
	int registry[1024];
	int reg_ix;
} CommandQueue;


void addCommand(CommandQueue *q, Command *comm);
Command *getCommand(CommandQueue *q);
int hasCommand(CommandQueue *q);
void flushCommands(CommandQueue *q);

int registerCommand(CommandQueue *q);

#endif /*_GCOMMAND_H_*/
