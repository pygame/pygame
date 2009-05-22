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

void    _PGTF_SetError(FreeTypeInstance *, const char *, FT_Error);

static FT_Error
_PGTF_face_request(FTC_FaceID face_id, 
        FT_Library library, 
        FT_Pointer request_data, 
        FT_Face *aface)
{
    FontId *id = GET_FONT_ID(face_id); 
    FT_Error error = 0;
    
    Py_BEGIN_ALLOW_THREADS;
        error = FT_Open_Face(library, &id->open_args, id->face_index, aface);
    Py_END_ALLOW_THREADS;

    return error;
}

void
_PGTF_SetError(FreeTypeInstance *ft, const char *error_msg, FT_Error error_id)
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

int
PGFT_TryLoadFont(FreeTypeInstance *ft, PyFreeTypeFont *font)
{
    FT_Face face;
    FT_Error error;

    error = FTC_Manager_LookupFace(ft->cache_manager, (FTC_FaceID)font, &face);

    if (error)
    {
        _PGTF_SetError(ft, "Failed to load font", error);
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

    free(ft->_error_msg);
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

    memset(inst, 0, sizeof(FreeTypeInstance));
    inst->_error_msg = malloc(1024);
    
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

    *_instance = inst;
    return 0;

error_cleanup:
    free(inst);
    *_instance = NULL;

    return error ? error : -1;
}
