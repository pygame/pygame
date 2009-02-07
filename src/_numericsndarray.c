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

#include "pygame.h"
#include "pygamedocs.h"
#include "mixer.h"
#include "numeric_arrayobject.h"
#include <SDL_byteorder.h>

static PyObject*
sndarray_samples (PyObject* self, PyObject* arg)
{
    int dim[2], numdims, type, formatbytes;
    PyObject *array, *chunkobj;
    Mix_Chunk* chunk;
    Uint16 format;
    int numchannels;

    if (!PyArg_ParseTuple (arg, "O!", &PySound_Type, &chunkobj))
        return NULL;
    chunk = PySound_AsChunk (chunkobj);

    if (!Mix_QuerySpec (NULL, &format, &numchannels))
        return RAISE (PyExc_SDLError, "Mixer not initialized");

    formatbytes = (abs (format) & 0xff) / 8;
    switch (format)
    {
    case AUDIO_S8:
        type = PyArray_CHAR;
        break;
    case AUDIO_U8:
        type = PyArray_UBYTE;
        break;
    case AUDIO_S16SYS:
        type = PyArray_SHORT;
        break;
    case AUDIO_U16SYS:
        type = PyArray_USHORT;
        break;
    default:
        return RAISE (PyExc_TypeError, "Unpresentable audio format");
    }

    numdims = (numchannels > 1) ? 2 : 1;
    dim[0] = chunk->alen / (numchannels*formatbytes);
    dim[1] = numchannels;
    
    array = PyArray_FromDimsAndData (numdims, dim, type, (char*)chunk->abuf);
    if(array)
    {
        Py_INCREF (chunkobj);
        ((PyArrayObject*) array)->base = chunkobj;
        ((PyArrayObject*) array)->flags |= SAVESPACE;
    }
    return array;
}

PyObject*
sndarray_array (PyObject* self, PyObject* arg)
{
    PyObject *array, *arraycopy=NULL;
    
    /*we'll let numeric do the copying for us*/
    array = sndarray_samples (self, arg);
    if(array)
    {
        arraycopy = PyArray_Copy ((PyArrayObject*) array);
        Py_DECREF (array);
    }
    return arraycopy;
}

PyObject*
sndarray_make_sound (PyObject* self, PyObject* arg)
{
    PyObject *arrayobj;
    PyArrayObject *array;
    Mix_Chunk *chunk;
    Uint16 format;
    int numchannels, mixerbytes;
    int loop1, loop2, step1, step2, length, length2=0;
    Uint8 *src, *dst;

    if (!PyArg_ParseTuple (arg, "O!", &PyArray_Type, &arrayobj))
	return NULL;
    array = (PyArrayObject*) arrayobj;
    
    if (!Mix_QuerySpec (NULL, &format, &numchannels))
        return RAISE (PyExc_SDLError, "Mixer not initialized");
    if (array->descr->type_num > PyArray_LONG)
        return RAISE (PyExc_ValueError, "Invalid array datatype for sound");
    
    if (format==AUDIO_S8 || format==AUDIO_U8)
        mixerbytes = 1;
    else
        mixerbytes = 2;
    
    /*test array dimensions*/
    if (numchannels == 1)
    {
        if (array->nd != 1)
            return RAISE (PyExc_ValueError,
                          "Array must be 1-dimensional for mono mixer");
    }
    else
    {
        if (array->nd != 2)
            return RAISE (PyExc_ValueError,
                          "Array must be 2-dimensional for stereo mixer");
        if (array->dimensions[1] != numchannels)
            return RAISE (PyExc_ValueError,
                          "Array depth must match number of mixer channels");
    }
    length = array->dimensions[0];
    step1 = array->strides[0];
    if (array->nd == 2)
    {
        length2 = array->dimensions[1];
	step2 = array->strides[1];
    }
    else 
    {
        length2 = 1;
        /*since length2 == 1, this won't be used for looping*/
	step2 = mixerbytes; 
    }

    /*create chunk, we are screwed if SDL_mixer ever does more than
     * malloc/free*/
    chunk = (Mix_Chunk *)malloc (sizeof (Mix_Chunk));
    if (chunk == NULL)
        return RAISE (PyExc_MemoryError, "Cannot allocate chunk\n");
    /*let's hope Mix_Chunk never changes also*/
    chunk->alen = length * numchannels * mixerbytes;
    chunk->abuf = (Uint8*) malloc (chunk->alen);
    chunk->allocated = 1;
    chunk->volume = 128;

    if (step1 == mixerbytes * numchannels && step2 == mixerbytes)
    {
        /*OPTIMIZATION: in these cases, we don't need to loop through
         *the samples individually, because the bytes are already layed
         *out correctly*/
        memcpy (chunk->abuf, array->data, chunk->alen);
    }
    else
    {
        dst = (Uint8*) chunk->abuf;
        if (mixerbytes == 1)
        {
            for (loop1 = 0; loop1 < length; loop1++)
            {
                src = (Uint8*) array->data + loop1*step1;
                switch (array->descr->elsize)
                {
                case 1:
                    for (loop2=0; loop2<length2; loop2++, dst+=1, src+=step2)
                        *(Uint8*)dst = (Uint8)*((Uint8*)src);
                    break;
                case 2:
                    for (loop2=0; loop2<length2; loop2++, dst+=1, src+=step2)
                        *(Uint8*)dst = (Uint8)*((Uint16*)src);
                    break;
                case 4:
                    for (loop2=0; loop2<length2; loop2++, dst+=1, src+=step2)
                        *(Uint8*)dst = (Uint8)*((Uint32*)src);
                    break;
                }
            }
        }
        else
        {
            for (loop1 = 0; loop1 < length; loop1++)
            {
                src = (Uint8*) array->data + loop1*step1;
                switch (array->descr->elsize)
                {
                case 1:
                    for (loop2=0; loop2<length2; loop2++, dst+=2, src+=step2)
                        *(Uint16*)dst = (Uint16)(*((Uint8*)src)<<8);
                    break;
                case 2:
                    for (loop2=0; loop2<length2; loop2++, dst+=2, src+=step2)
                        *(Uint16*)dst = (Uint16)*((Uint16*)src);
                    break;
                case 4:
                    for (loop2=0; loop2<length2; loop2++, dst+=2, src+=step2)
                        *(Uint16*)dst = (Uint16)*((Uint32*)src);
                    break;
                }
            }
        }
    }
    
    return PySound_New (chunk);
}

static PyMethodDef sndarray_builtins[] =
{
    { "samples", sndarray_samples, METH_VARARGS, DOC_PYGAMESNDARRAYSAMPLES },
    { "array", sndarray_array, METH_VARARGS, DOC_PYGAMESNDARRAYARRAY },
    { "make_sound", sndarray_make_sound, METH_VARARGS,
      DOC_PYGAMESNDARRAYMAKESOUND },
    { NULL, NULL, 0, NULL}
};

PYGAME_EXPORT
void init_numericsndarray (void)
{
    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base ();
    if (PyErr_Occurred ()) {
	return;
    }
    import_pygame_mixer ();
    if (PyErr_Occurred ()) {
	return;
    }
    import_array ();
    if (PyErr_Occurred ()) {
	return;
    }

    /* create the module */
    Py_InitModule3 ("_numericsndarray", sndarray_builtins, DOC_PYGAMESNDARRAY);
}
