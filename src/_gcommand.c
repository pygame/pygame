#include "_gcommand.h"


void addCommand(CommandQueue *q, Command *comm)
{
	SDL_LockMutex(q->q_mutex);
	if(!q->first)
	{
		q->first=comm;
		q->size++;
		SDL_UnlockMutex(q->q_mutex);
		return;
	}
	if(!q->last)
	{
		q->last=comm;
		q->first->next=comm;
		q->size++;
		SDL_UnlockMutex(q->q_mutex);
		return;
	}
	q->last->next=comm;
	q->last=comm;
	q->size++;
	SDL_UnlockMutex(q->q_mutex);
	return;
}

Command *getCommand(CommandQueue *q)
{
	SDL_LockMutex(q->q_mutex);
	Command *comm;
	if(!q->last && q->first)
	{
		comm=q->first;
		q->size--;
		SDL_UnlockMutex(q->q_mutex);
		return comm;
	}
	else if (!q->last && !q->first)
	{
		SDL_UnlockMutex(q->q_mutex);
		return NULL;
	}
	comm=q->first;
	q->first=q->first->next;
	q->size--;
	SDL_UnlockMutex(q->q_mutex);
	return comm;
}

int hasCommand(CommandQueue *q)
{
	if(q->size>0)
		return 1;
	return 0;
}

void flushCommands(CommandQueue *q)
{
	SDL_LockMutex(q->q_mutex);
	Command *prev;
	Command *cur = q->first;
	while(cur!=NULL)
	{
		prev=cur;
		cur=cur->next;
		PyMem_Free(prev);
		q->size--;
	}
	SDL_UnlockMutex(q->q_mutex);
}
