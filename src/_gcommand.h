#ifndef _GCOMMAND_H_
#define _GCOMMAND_H_
#include <SDL.h>
#include <SDL_thread.h>
#include <Python.h>

typedef struct Command
{
	int type;
	struct Command *next;
	size_t size;
} Command;

typedef struct CommandQueue
{
	int size;
	SDL_mutex *q_mutex;
	Command *first;
	Command *last;
} CommandQueue;


void addCommand(CommandQueue *q, Command *comm);
Command *getCommand(CommandQueue *q);
int hasCommand(CommandQueue *q);
void flushCommands(CommandQueue *q);

#endif /*_GCOMMAND_H_*/
