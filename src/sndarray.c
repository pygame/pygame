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

#include"pygame.h"
#include"mixer.h"
#include<Numeric/arrayobject.h>
#include<SDL_byteorder.h>




    /*DOC*/ static char doc_samples[] =
    /*DOC*/    "pygame.sndarray.samples(Surface) -> Array\n"
    /*DOC*/    "get a reference to the sound samples\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will return an array that directly references the samples\n"
    /*DOC*/    "in the array.\n"
    /*DOC*/ ;

static PyObject* sndarray_samples(PyObject* self, PyObject* arg)
{
	int dim[2], numdims, type, formatbytes;
	PyObject *array, *chunkobj;
	Mix_Chunk* chunk;
	Uint16 format;
        int numchannels;
        
	if(!PyArg_ParseTuple(arg, "O!", &PySound_Type, &chunkobj))
		return NULL;
	chunk = PySound_AsChunk(chunkobj);

        if(!Mix_QuerySpec(NULL, &format, &numchannels))
            return RAISE(PyExc_SDLError, "Mixer not initialized");

        formatbytes = (abs(format)&0xff)/8;
        if(format == AUDIO_S8)
            type = PyArray_CHAR;
        else if(format == AUDIO_U8)
            type = PyArray_UBYTE;
        else if(formatbytes == 2)
            type = PyArray_SHORT;
        else
            return RAISE(PyExc_TypeError, "Unpresentable audio format");

        numdims = (numchannels>1)?2:1;
        dim[0] = chunk->alen / (formatbytes*numchannels);
        dim[1] = numchannels;

	array = PyArray_FromDimsAndData(numdims, dim, type, chunk->abuf);
	if(array)
	{
            Py_INCREF(chunkobj);
            ((PyArrayObject*)array)->base = chunkobj;
            ((PyArrayObject*)array)->flags |= SAVESPACE;
	}
	return array;
}

    /*DOC*/ static char doc_array[] =
    /*DOC*/    "pygame.sndarray.array(Sound) -> Array\n"
    /*DOC*/    "get an array copied from a sound\n"
    /*DOC*/    "\n"
    /*DOC*/    "Creates an array with a copy of the sound data.\n"
    /*DOC*/ ;

PyObject* sndarray_array(PyObject* self, PyObject* arg)
{
	PyObject *array, *arraycopy=NULL;

/*we'll let numeric do the copying for us*/
        array = sndarray_samples(self, arg);
        if(array)
        {
            arraycopy = PyArray_Copy(array);
            Py_DECREF(array);
        }
        return arraycopy;
}


    /*DOC*/ static char doc_make_sound[] =
    /*DOC*/    "pygame.sndarray.make_sound(array) -> Sound\n"
    /*DOC*/    "create a new Sound object from array data\n"
    /*DOC*/    "\n"
    /*DOC*/    "Create a new playable Sound object from array data\n"
    /*DOC*/    "the Sound will be a copy of the array samples.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The samples must be in the same format of the initialized\n"
    /*DOC*/    "mixer module. This means you must match the data type\n"
    /*DOC*/    "(8bit or 16bit) as well as match the number of channels\n"
    /*DOC*/    "(usually stereo or mono).\n"
    /*DOC*/ ;

PyObject* sndarray_make_sound(PyObject* self, PyObject* arg)
{
    PyObject *arrayobj;
    PyArrayObject *array;
    Mix_Chunk *chunk;
    Uint16 format;
    int numchannels, samplesize;
    int loop1, loop2, step1, step2;
    Uint8 *src, *dst;

    if(!PyArg_ParseTuple(arg, "O!", &PyArray_Type, &arrayobj))
	return NULL;
    array = (PyArrayObject*)arrayobj;
    
    if(!Mix_QuerySpec(NULL, &format, &numchannels))
        return RAISE(PyExc_SDLError, "Mixer not initialized");

    /*test sample size*/
    if(format==AUDIO_S8 || format==AUDIO_U8)
    {
        if(array->descr->elsize != 1)
            return RAISE(PyExc_ValueError, "Array must contain 8bit audio, to match mixer");
    }
    else if(array->descr->elsize != 2)
        return RAISE(PyExc_ValueError, "Array must contain 16bit audio, to match mixer");
    
    /*test array dimensions*/
    if(numchannels==1)
    {
        if(array->nd != 1)
            return RAISE(PyExc_ValueError, "Array must be 1-dimensional for mono mixer");
    }
    else
    {
        if(array->nd != 2)
            return RAISE(PyExc_ValueError, "Array must be 2-dimensional for stereo mixer");
        if(array->dimensions[1] != numchannels)
            return RAISE(PyExc_ValueError, "Array depth must match number of mixer channels");
    }

    /*create chunk, we are screwed if SDL_mixer ever does more than malloc/free*/
    chunk = (Mix_Chunk *)malloc(sizeof(Mix_Chunk));
    if ( chunk == NULL )
        return RAISE(PyExc_MemoryError, "Cannot allocate chunk\n");
    /*let's hope Mix_Chunk never changes also*/
    chunk->alen = array->descr->elsize * array->dimensions[0] * numchannels;
    chunk->abuf = (Uint8*)malloc(chunk->alen);
    chunk->allocated = 1;
    chunk->volume = 128;
    
    
    if(array->nd == 1)
    {
        step1 = array->strides[0];
        src = (Uint8*)array->data;
        dst = (Uint8*)chunk->abuf;
        if(array->descr->elsize == 1)
        {
            for(loop1 = 0; loop1 < array->dimensions[0]; loop1++)
            {
                *dst = *src;
                dst += 1;
                src += step1;
            }
        }
        else
        {
            for(loop1 = 0; loop1 < array->dimensions[0]; loop1++)
            {
                *(Uint16*)dst = *(Uint16*)src;
                dst += 2;
                src += step1;
            }
        }
    }
    else
    {
        step1 = array->strides[0];
        step2 = array->strides[1];
        dst = (Uint8*)chunk->abuf;
        if(array->descr->elsize == 1)
        {
            for(loop1 = 0; loop1 < array->dimensions[0]; loop1++)
            {
                src = (Uint8*)array->data + loop1*step1;
                for(loop2 = 0; loop2 < array->dimensions[1]; loop2++)
                {
                    *dst = *(src+loop2*step2);
                    dst += 1;
                }
            }
        }
        else
        {
            for(loop1 = 0; loop1 < array->dimensions[0]; loop1++)
            {
                src = (Uint8*)array->data + loop1*step1;
                for(loop2 = 0; loop2 < array->dimensions[1]; loop2++)
                {
                    *(Uint16*)dst = *(Uint16*)(src+loop2*step2);
                    dst += 2;
                }
            }
        }
    }
    
printf("%%returnsound\n");
    return PySound_New(chunk);
}



static PyMethodDef sndarray_builtins[] =
{
	{ "samples", sndarray_samples, 1, doc_samples },
	{ "array", sndarray_array, 1, doc_array },
	{ "make_sound", sndarray_make_sound, 1, doc_make_sound },

	{ NULL, NULL }
};





    /*DOC*/ static char doc_pygame_sndarray_MODULE[] =
    /*DOC*/    "Contains routines for mixing numeric arrays with\n"
    /*DOC*/    "sounds\n"
    /*DOC*/ ;

PYGAME_EXPORT
void initsndarray(void)
{
	PyObject *module, *dict;

    /* create the module */
	module = Py_InitModule3("sndarray", sndarray_builtins, doc_pygame_sndarray_MODULE);
	dict = PyModule_GetDict(module);

	/*imported needed apis*/
	import_pygame_base();
	import_pygame_mixer();
	import_array();
}



