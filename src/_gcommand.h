#ifndef _GCOMMAND_H_
#define _GCOMMAND_H_
#include <SDL.h>
#include <SDL_thread.h>
#include <Python.h>

#define COMMON_COMMAND \
	int type;\
	size_t size; 


typedef struct Command
{
	COMMON_COMMAND
	struct Command *next;
} Command;

#define FULL COMMAND \
	int type;\
	size_t size;\
	struct Command *next;

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
