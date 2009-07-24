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

#include "ft_mod.h"
#include "ft_wrap.h"
#include "pgfreetype.h"
#include "pgtypes.h"
#include "freetypebase_doc.h"

#include FT_MODULE_H


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

          int i;
    const char *ft_msg;

    ft_msg = NULL;
    for (i = 0; ft_errors[i].err_msg != NULL; ++i)
    {
        if (error_id == ft_errors[i].err_code)
        {
            ft_msg = ft_errors[i].err_msg;
            break;
        }
    }

    if (ft_msg)
        sprintf(ft->_error_msg, "%s: %s", error_msg, ft_msg);
    else
        strcpy(ft->_error_msg, error_msg);
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
    FT_Face face;
    face = _PGFT_GetFace(ft, font);

    return face ? FT_IS_FIXED_WIDTH(face) : 0;
}

const char *
PGFT_Face_GetName(FreeTypeInstance *ft, PyFreeTypeFont *font)
{
    FT_Face face;
    face = _PGFT_GetFace(ft, font);

    return face ? face->family_name : ""; 
}

int
PGFT_Face_GetHeight(FreeTypeInstance *ft, PyFreeTypeFont *font)
{
    FT_Face face;
    face = _PGFT_GetFace(ft, font);

    return face ? face->height : 0;
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

    /*
     * TODO: Check if face has already been sized?
     */

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
    scale->width = scale->height = (pguint32)(size * 64);
    scale->pixel = 0;
    scale->x_res = scale->y_res = 0;
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


int
PGFT_TryLoadFont_Filename(FreeTypeInstance *ft, 
    PyFreeTypeFont *font, 
    const char *filename, 
    int face_index)
{
    char *filename_alloc;
    size_t file_len;

    file_len = strlen(filename);
    filename_alloc = malloc(file_len + 1);
    if (!filename_alloc)
    {
        _PGFT_SetError(ft, "Could not allocate memory", 0);
        return -1;
    }

    strcpy(filename_alloc, filename);
    filename_alloc[file_len] = 0;

    font->id.face_index = face_index;
    font->id.open_args.flags = FT_OPEN_PATHNAME;
    font->id.open_args.pathname = filename_alloc;

    font->_internals = malloc(sizeof(FontInternals));
    memset(font->_internals, 0x0, sizeof(FontInternals));

    PGFT_Cache_Init(ft, &PGFT_INTERNALS(font)->cache, font);

    return _PGFT_GetFace(ft, font) ? 0 : -1;
}

void
PGFT_UnloadFont(FreeTypeInstance *ft, PyFreeTypeFont *font)
{
    if (ft != NULL)
        FTC_Manager_RemoveFaceID(ft->cache_manager, (FTC_FaceID)(&font->id));

    if (PGFT_INTERNALS(font))
    {
        PGFT_Cache_Destroy(&PGFT_INTERNALS(font)->cache);
        free(PGFT_INTERNALS(font)->active_text.glyphs);
        free(PGFT_INTERNALS(font));
    }

    free(font->id.open_args.pathname);
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

    inst = malloc(sizeof(FreeTypeInstance));

    if (!inst)
        goto error_cleanup;

    memset(inst, 0, sizeof(FreeTypeInstance));
    inst->_error_msg = calloc(1024, sizeof(char));
    inst->cache_size = cache_size;
    
    if (FT_Init_FreeType(&inst->library) != 0)
        goto error_cleanup;

    if (FTC_Manager_New(inst->library, 0, 0, 0,
            &_PGFT_face_request, NULL,
            &inst->cache_manager) != 0)
        goto error_cleanup;

    if (FTC_CMapCache_New(inst->cache_manager, 
            &inst->cache_charmap) != 0)
        goto error_cleanup;

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

    /* TODO: Free caches */

    if (ft->cache_manager)
        FTC_Manager_Done(ft->cache_manager);

    if (ft->library)
        FT_Done_FreeType(ft->library);

    if (ft->_error_msg)
        free (ft->_error_msg);
    
    free (ft);
}

