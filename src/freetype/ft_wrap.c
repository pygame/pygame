/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners
  Copyright (C) 2008 Marcus von Appen
  Copyright (C) 2009 Vicent Marti

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

#define PYGAME_FREETYPE_INTERNAL

#include "ft_mod.h"
#include "ft_wrap.h"
#include "pgfreetype.h"
#include "pgsdl.h"
#include "freetypebase_doc.h"

/* Externals */
FILE *PyFile_AsFile(PyObject *p);


void    _PGTF_SetError(const char *error_msg, FT_Error error_id);


static unsigned long 
_RWread(FT_Stream stream, 
        unsigned long offset, 
        unsigned char* buffer, 
        unsigned long count)
{
    PyObject *obj;
    FILE *file;
    unsigned long readc;

    obj = (PyObject *)stream->descriptor.pointer;

    Py_BEGIN_ALLOW_THREADS;

        file = PyFile_AsFile(obj);
        fseek(file, (long int)offset, SEEK_SET);

        readc = count ? fread(buffer, 1, count, file) : 0;

    Py_END_ALLOW_THREADS;

    return readc;
}

static FT_Error
_PGTF_face_request(FTC_FaceID face_id, 
        FT_Library library, 
        FT_Pointer request_data, 
        FT_Face *aface)
{
    FontId *id = GET_FONT_ID(face_id); 

    FT_Error error = 0;
    FT_Stream stream = NULL;
    FILE *file;

    stream = malloc(sizeof(FT_Stream));

	if (stream == NULL) 
        goto error_cleanup;

	memset(stream, 0, sizeof(FT_Stream));

	stream->read = _RWread;
    /* do not let FT close the stream, Python will take care of it */
    stream->close = NULL; 

	stream->descriptor.pointer = id->file_ptr;

    Py_BEGIN_ALLOW_THREADS;
        file = PyFile_AsFile(id->file_ptr);
        
        stream->size = (unsigned long)fseek(file, 0, SEEK_END);
        stream->pos = (unsigned long)fseek(file, 0, SEEK_SET);
    Py_END_ALLOW_THREADS;

	id->open_args.flags = FT_OPEN_STREAM;
	id->open_args.stream = stream;

	error = FT_Open_Face(library, &id->open_args, id->face_index, aface);

    if (error)
        goto error_cleanup;

    return 0;

error_cleanup:
    free(stream);
    return error ? error : -1;
}

void
_PGTF_SetError(const char *error_msg, FT_Error error_id)
{

}

int
PGFT_TryLoadFont(FreeTypeInstance *ft, PyFreeTypeFont *font)
{
    FT_Face face;
    FT_Error error;

    error = FTC_Manager_LookupFace(ft->cache_manager, (FTC_FaceID)font, &face);

    if (error)
    {
        _PGTF_SetError("Failed to load font", error);
        return error;
    }

    return 0;
}

void
PGFT_UnloadFont(FreeTypeInstance *ft, PyFreeTypeFont *font)
{
    FTC_Manager_RemoveFaceID(ft->cache_manager, (FTC_FaceID)font);
}

void
PGFT_Quit(FreeTypeInstance *ft)
{
    if (ft == NULL)
        return;

    /* TODO: Free caches */

    FTC_Manager_Done(ft->cache_manager);
    FT_Done_FreeType(ft->library);
    free(ft);
}

int
PGFT_Init(FreeTypeInstance **_instance)
{
    FT_Error error = 0;
    FreeTypeInstance *inst = NULL;

    inst = malloc(sizeof(FreeTypeInstance));

    if (!inst)
        goto error_cleanup;
    
    error = FT_Init_FreeType(&inst->library);

    if (error)
        goto error_cleanup;

    error = FTC_Manager_New(
            inst->library, 
            0, 0, 0, 
            &_PGTF_face_request, 
            NULL, 
            &inst->cache_manager);

    if (error)
        goto error_cleanup;

    return 0;

error_cleanup:
    free(inst);
    *_instance = NULL;

    return error;
}
