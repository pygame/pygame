/*
  pygame - Python Game Library

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

/*
 * _movie - movie support for pygame with ffmpeg
 * Author: Tyler Laing
 *
 * This module allows for the loading of, playing, pausing, stopping, and so on
 *  of a video file. Any format supported by ffmpeg is supported by this
 *  video player. Any bugs, please email trinioler@gmail.com :)
 */


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

//convience function for allocating a new command, and ensuring its type is set properly.
#define ALLOC_COMMAND(command, name) command* name = (command *)PyMem_Malloc(sizeof(command)); name->type=movie->command##Type;

#endif /*_GCOMMAND_H_*/
