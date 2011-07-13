/*
  pygame - Python Game Library
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

#include "ft_wrap.h"
#include FT_MODULE_H

static unsigned long _RWops_read(FT_Stream stream, unsigned long offset,
	unsigned char *buffer, unsigned long count);
static int _PGFT_Init_INTERNAL(FreeTypeInstance *ft, PyFreeTypeFont *font);
static void _PGFT_Free_INTERNAL(PyFreeTypeFont *font);


/*********************************************************
 *
 * Error management
 *
 *********************************************************/
void
_PGFT_SetError(FreeTypeInstance *ft, const char *error_msg, FT_Error error_id)
{
#undef __FTERRORS_H__
#define FT_ERRORDEF( e, v, s )  { e, s },
#define FT_ERROR_START_LIST     {
#define FT_ERROR_END_LIST       {0, 0}};
    static const struct
    {
        int          err_code;
        const char*  err_msg;
    } ft_errors[] = 
#include FT_ERRORS_H

    const int maxlen = (int)(sizeof (ft->_error_msg)) - 1;
    int i;
    const char *ft_msg;
    int error_msg_len = (int)strlen(error_msg);

    ft_msg = NULL;
    for (i = 0; ft_errors[i].err_msg != NULL; ++i)
    {
        if (error_id == ft_errors[i].err_code)
        {
            ft_msg = ft_errors[i].err_msg;
            break;
        }
    }

    if (error_id && ft_msg && maxlen > error_msg_len - 42)
        sprintf(ft->_error_msg, "%.*s: %.*s",
                maxlen - 2, error_msg, maxlen - error_msg_len - 2, ft_msg);
    else
    {
        strncpy(ft->_error_msg, error_msg, maxlen);
        ft->_error_msg[maxlen] = '\0';  /* in case of message truncation */
    }
}

const char *
PGFT_GetError(FreeTypeInstance *ft)
{
    return ft->_error_msg;
}




/*********************************************************
 *
 * Misc getters
 *
 *********************************************************/
int
PGFT_Face_IsFixedWidth(FreeTypeInstance *ft, PyFreeTypeFont *font)
{
    FT_Face face = _PGFT_GetFace(ft, font);

    if (face == NULL) {
        RAISE(PyExc_RuntimeError, PGFT_GetError(ft));
        return -1;
    }
    return FT_IS_FIXED_WIDTH(face);
}

const char *
PGFT_Face_GetName(FreeTypeInstance *ft, PyFreeTypeFont *font)
{
    FT_Face face;
    face = _PGFT_GetFace(ft, font);

    if (face == NULL) {
        RAISE(PyExc_RuntimeError, PGFT_GetError(ft));
        return NULL;
    }
    return face->family_name; 
}

int
PGFT_Face_GetHeight(FreeTypeInstance *ft, PyFreeTypeFont *font)
{
    FT_Face face;
    face = _PGFT_GetFace(ft, font);

    if (face == NULL) {
        RAISE(PyExc_RuntimeError, PGFT_GetError(ft));
        return -1;
    }
    return face->height;
}




/*********************************************************
 *
 * Face access
 *
 *********************************************************/
FT_Face
_PGFT_GetFaceSized(FreeTypeInstance *ft,
    PyFreeTypeFont *font,
    int face_size)
{
    FT_Error error;
    FTC_ScalerRec scale;
    FT_Size _fts;

    _PGFT_BuildScaler(font, &scale, face_size);

    error = FTC_Manager_LookupSize(ft->cache_manager, 
        &scale, &_fts);

    if (error)
    {
        _PGFT_SetError(ft, "Failed to resize face", error);
        return NULL;
    }

    return _fts->face;
}

FT_Face
_PGFT_GetFace(FreeTypeInstance *ft,
    PyFreeTypeFont *font)
{
    FT_Error error;
    FT_Face face;

    error = FTC_Manager_LookupFace(ft->cache_manager,
        (FTC_FaceID)(&font->id),
        &face);

    if (error)
    {
        _PGFT_SetError(ft, "Failed to load face", error);
        return NULL;
    }

    return face;
}




/*********************************************************
 *
 * Scaling
 *
 *********************************************************/
void
_PGFT_BuildScaler(PyFreeTypeFont *font, FTC_Scaler scale, int size)
{
    scale->face_id = (FTC_FaceID)(&font->id);
    scale->width = scale->height = (FT_UInt32)(size * 64);
    scale->pixel = 0;
    scale->x_res = scale->y_res = font->resolution;
}







/*********************************************************
 *
 * Font loading
 *
 * TODO:
 *  - Loading from rwops, existing files, etc
 *
 *********************************************************/
static FT_Error
_PGFT_face_request(FTC_FaceID face_id, 
    FT_Library library, 
    FT_Pointer request_data, 
    FT_Face *aface)
{
    FontId *id = (FontId *)face_id; 
    FT_Error error;
    
    Py_BEGIN_ALLOW_THREADS;
    error = FT_Open_Face(library, &id->open_args, id->face_index, aface);
    Py_END_ALLOW_THREADS;

    return error;
}



int _PGFT_Init_INTERNAL(FreeTypeInstance *ft, PyFreeTypeFont *font)
{
    font->_internals = _PGFT_malloc(sizeof(FontInternals));
    if (font->_internals == NULL)
    {
        PyErr_NoMemory();
        return -1;
    }
    memset(font->_internals, 0x0, sizeof(FontInternals));

    if (PGFT_Cache_Init(ft, &PGFT_INTERNALS(font)->cache, font))
    {
        _PGFT_free(font->_internals);
        font->_internals = NULL;
        PyErr_NoMemory();
        return -1;
    }

    if (!_PGFT_GetFace(ft, font))
    {
        PGFT_Cache_Destroy(&PGFT_INTERNALS(font)->cache);
        _PGFT_free(font->_internals);
        font->_internals = NULL;
        RAISE(PyExc_RuntimeError, PGFT_GetError(ft));
        return -1;
    }

    PGFT_INTERNALS(font)->active_text.glyphs = NULL;
    PGFT_INTERNALS(font)->active_text.posns = NULL;

    return 0;
}

void
_PGFT_Free_INTERNAL(PyFreeTypeFont *font)
{
    if (font->_internals)
    {
        _PGFT_free(PGFT_INTERNALS(font)->active_text.glyphs);
        _PGFT_free(PGFT_INTERNALS(font)->active_text.posns);
        _PGFT_free(font->_internals);
        font->_internals = NULL;
    }
}

int
PGFT_TryLoadFont_Filename(FreeTypeInstance *ft, 
    PyFreeTypeFont *font, 
    const char *filename, 
    int face_index)
{
    char *filename_alloc;
    size_t file_len;

    file_len = strlen(filename);
    filename_alloc = _PGFT_malloc(file_len + 1);
    if (!filename_alloc)
    {
        PyErr_NoMemory();
        return -1;
    }

    strcpy(filename_alloc, filename);
    filename_alloc[file_len] = 0;

    font->id.face_index = face_index;
    font->id.open_args.flags = FT_OPEN_PATHNAME;
    font->id.open_args.pathname = filename_alloc;

    return _PGFT_Init_INTERNAL(ft, font);
}

#ifdef HAVE_PYGAME_SDL_RWOPS
static unsigned long 
_RWops_read(FT_Stream stream, unsigned long offset,
	unsigned char *buffer, unsigned long count)
{
	SDL_RWops *src;

	src = (SDL_RWops *)stream->descriptor.pointer;
	SDL_RWseek(src, (int)offset, SEEK_SET);

	if (count == 0)
		return 0;

	return SDL_RWread(src, buffer, 1, (int)count);
}

int PGFT_TryLoadFont_RWops(FreeTypeInstance *ft, 
        PyFreeTypeFont *font, SDL_RWops *src, int face_index)
{
    FT_Stream stream;
    int position;

    position = SDL_RWtell(src);

    if (position < 0)
    {
        RAISE(PyExc_SDLError, "Failed to seek in font stream");
        return -1;
    }

    stream = _PGFT_malloc(sizeof(*stream));

    if (stream == NULL)
    {
        PyErr_NoMemory();
        return -1;
    }

    memset(stream, 0, sizeof(*stream));

    stream->read = _RWops_read;
    stream->descriptor.pointer = src;
    stream->pos = (unsigned long)position;
    SDL_RWseek(src, 0, SEEK_END);
    stream->size = (unsigned long)(SDL_RWtell(src) - position);
    SDL_RWseek(src, position, SEEK_SET);

    font->id.face_index = face_index;
    font->id.open_args.flags = FT_OPEN_STREAM;
    font->id.open_args.stream = stream;

    return _PGFT_Init_INTERNAL(ft, font);
}
#endif

void
PGFT_UnloadFont(FreeTypeInstance *ft, PyFreeTypeFont *font)
{
    if (font->id.open_args.flags == 0)
        return;

    if (ft != NULL)
    {
        FTC_Manager_RemoveFaceID(ft->cache_manager, (FTC_FaceID)(&font->id));
        if (PGFT_INTERNALS(font))
            PGFT_Cache_Destroy(&PGFT_INTERNALS(font)->cache);
    }

    if (font->id.open_args.flags == FT_OPEN_STREAM)
    {
        _PGFT_free(font->id.open_args.pathname);
    }
    else if (font->id.open_args.flags == FT_OPEN_PATHNAME)
    {
        _PGFT_free(font->id.open_args.stream);
    }
    font->id.open_args.flags = 0;

    _PGFT_Free_INTERNAL(font);
}




/*********************************************************
 *
 * Library (de)initialization
 *
 *********************************************************/
int
PGFT_Init(FreeTypeInstance **_instance, int cache_size)
{
    FreeTypeInstance *inst = NULL;
    int error;

    inst = _PGFT_malloc(sizeof(FreeTypeInstance));

    if (!inst)
    {
        PyErr_NoMemory();
        goto error_cleanup;
    }

    memset(inst, 0, sizeof(FreeTypeInstance));
    inst->cache_size = cache_size;

    error = FT_Init_FreeType(&inst->library);
    if (error)
    {
        RAISE(PyExc_RuntimeError,
              "pygame (PGFT_Init): failed to initialize FreeType library");
        goto error_cleanup;
    }

    if (FTC_Manager_New(inst->library, 0, 0, 0,
            &_PGFT_face_request, NULL,
            &inst->cache_manager) != 0)
    {
        RAISE(PyExc_RuntimeError,
              "pygame (PGFT_Init): failed to create new FreeType manager");
        goto error_cleanup;
    }

    if (FTC_CMapCache_New(inst->cache_manager, 
            &inst->cache_charmap) != 0)
    {
        RAISE(PyExc_RuntimeError,
              "pygame (PGFT_Init): failed to create new FreeType cache");
        goto error_cleanup;
    }

    *_instance = inst;
    return 0;

error_cleanup:
    PGFT_Quit(inst);
    *_instance = NULL;

    return -1;
}

void
PGFT_Quit(FreeTypeInstance *ft)
{
    if (ft == NULL)
        return;

    if (ft->cache_manager)
        FTC_Manager_Done(ft->cache_manager);

    if (ft->library)
        FT_Done_FreeType(ft->library);

    _PGFT_free (ft);
}

