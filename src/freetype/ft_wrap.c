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
#include "pgfreetype.h"
#include "pgsdl.h"
#include "freetypebase_doc.h"

void    PGFT_Quit(FreeTypeInstance *);
int     PGFT_Init(FreeTypeInstance **);
int     PGFT_TryLoadFont(FreeTypeInstance *ft, PyFreeTypeFont *font);
void    PGFT_UnloadFont(FreeTypeInstance *, PyFreeTypeFont *);

void    _PGTF_SetError(const char *error_msg, FT_Error error_id);



static unsigned long 
_RWread(FT_Stream stream, 
        unsigned long offset, 
        unsigned char* buffer, 
        unsigned long count)
{
	SDL_RWops *src;

	src = (SDL_RWops *)stream->descriptor.pointer;
	SDL_RWseek(src, (int)offset, SEEK_SET);

	if (count == 0)
		return 0;

	return SDL_RWread(src, buffer, 1, (int)count);
}



static FT_Error
_PGTF_face_request(FTC_FaceID face_id, 
        FT_Library library, 
        FT_Pointer request_data, 
        FT_Face *aface)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)face_id;

    FT_Error error = 0;
    FT_Stream stream = NULL;
    int position;
    
	position = SDL_RWtell(font->rwops);

	if (position < 0)
        goto error_cleanup;

    stream = malloc(sizeof(FT_Stream));

	if (stream == NULL) 
        goto error_cleanup;

	memset(stream, 0, sizeof(FT_Stream));

	stream->read = _RWread;
	stream->descriptor.pointer = font->rwops;
	stream->pos = (unsigned long)position;
	SDL_RWseek(font->rwops, 0, SEEK_END);
	stream->size = (unsigned long)(SDL_RWtell(font->rwops) - position);
	SDL_RWseek(font->rwops, position, SEEK_SET);

	font->open_args.flags = FT_OPEN_STREAM;
	font->open_args.stream = stream;

	error = FT_Open_Face(library, &font->open_args, font->face_index, aface);

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
