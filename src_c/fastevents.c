/*
  NET2 is a threaded, event based, network IO library for SDL.
  Copyright (C) 2002 Bob Pendleton

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public License
  as published by the Free Software Foundation; either version 2.1
  of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free
  Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
  02111-1307 USA

  If you do not wish to comply with the terms of the LGPL please
  contact the author as other terms are available for a fee.

  Bob Pendleton
  Bob@Pendleton.com
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "SDL.h"
#include "SDL_thread.h"

#include "fastevents.h"

// ----------------------------------------
//
//error handling code
//

static char *error = NULL;

static __inline__ void
setError (char *err)
{
    error = err;
}

char*
FE_GetError ()
{
    return error;
}

//----------------------------------------
//
//Threads, mutexs, thread utils, and
// thread safe wrappers
//

static SDL_mutex *eventLock = NULL;
static SDL_cond *eventWait = NULL;
static SDL_TimerID eventTimer = 0;

//----------------------------------------
//
//
//

int
FE_PushEvent (SDL_Event * ev)
{
    SDL_LockMutex (eventLock);
    while (-1 == SDL_PushEvent (ev))
        SDL_CondWait (eventWait, eventLock);
    SDL_CondSignal (eventWait);
    SDL_UnlockMutex (eventLock);

    return 1;
}

//----------------------------------------
//
//
//

void
FE_PumpEvents ()
{
    SDL_LockMutex (eventLock);
    SDL_PumpEvents ();
    SDL_UnlockMutex (eventLock);
}

//----------------------------------------
//
//
//

int
FE_PollEvent (SDL_Event * event)
{
    int val = 0;

    SDL_LockMutex (eventLock);
    val = SDL_PollEvent (event);
    if (0 < val)
        SDL_CondSignal (eventWait);
    SDL_UnlockMutex (eventLock);

    return val;
}

//----------------------------------------
//
//Replacement for SDL_WaitEvent
//

int
FE_WaitEvent (SDL_Event * event)
{
    int val = 0;

    SDL_LockMutex (eventLock);
    while (0 >= (val = SDL_PollEvent (event)))
        SDL_CondWait (eventWait, eventLock);
    SDL_CondSignal (eventWait);
    SDL_UnlockMutex (eventLock);

    return val;
}

//----------------------------------------
//
//
//

static Uint32
timerCallback (Uint32 interval, void *param)
{
    SDL_LockMutex (eventLock);
    SDL_CondBroadcast (eventWait);
    SDL_UnlockMutex (eventLock);
    return interval;
}

//----------------------------------------
//
//
//

int
FE_Init ()
{
    if (0 == (SDL_INIT_TIMER & SDL_WasInit (SDL_INIT_TIMER)))
        SDL_InitSubSystem (SDL_INIT_TIMER);

    eventLock = SDL_CreateMutex ();
    if (NULL == eventLock)
    {
        setError ("FE: can't create a mutex");
        return -1;
    }

    eventWait = SDL_CreateCond ();
    if (NULL == eventWait)
    {
        setError ("FE: can't create a condition variable");
        return -1;
    }

    eventTimer = SDL_AddTimer (10, timerCallback, NULL);
    if (NULL == eventTimer)
    {
        setError ("FE: can't add a timer");
        return -1;
    }

    return 0;
}

//----------------------------------------
//
//
//

void
FE_Quit ()
{
    SDL_DestroyMutex (eventLock);
    eventLock = NULL;

    SDL_DestroyCond (eventWait);
    eventWait = NULL;

    SDL_RemoveTimer (eventTimer);
}
