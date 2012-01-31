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

static unsigned long RWops_read(FT_Stream, unsigned long,
                                unsigned char *, unsigned long);
static int init(FreeTypeInstance *, PgFontObject *);
static void quit(PgFontObject *);


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
    static const struct {
        int          err_code;
        const char*  err_msg;
    } ft_errors[] =
#include FT_ERRORS_H

    const int maxlen = (int)(sizeof(ft->_error_msg)) - 1;
    int i;
    const char *ft_msg;
    int error_msg_len = (int)strlen(error_msg);

    ft_msg = 0;
    for (i = 0; ft_errors[i].err_msg; ++i) {
        if (error_id == ft_errors[i].err_code) {
            ft_msg = ft_errors[i].err_msg;
            break;
        }
    }

    if (error_id && ft_msg && maxlen > error_msg_len - 42)
        sprintf(ft->_error_msg, "%.*s: %.*s",
                maxlen - 2, error_msg, maxlen - error_msg_len - 2, ft_msg);
    else {
        strncpy(ft->_error_msg, error_msg, maxlen);
        ft->_error_msg[maxlen] = '\0';  /* in case of message truncation */
    }
}

const char *
_PGFT_GetError(FreeTypeInstance *ft)
{
    return ft->_error_msg;
}




/*********************************************************
 *
 * Misc getters
 *
 *********************************************************/
int
_PGFT_Font_IsFixedWidth(FreeTypeInstance *ft, PgFontObject *fontobj)
{
    FT_Face font = _PGFT_GetFont(ft, fontobj);

    if (!font) {
        RAISE(PyExc_RuntimeError, _PGFT_GetError(ft));
        return -1;
    }
    return FT_IS_FIXED_WIDTH(font);
}

const char *
_PGFT_Font_GetName(FreeTypeInstance *ft, PgFontObject *fontobj)
{
    FT_Face font;
    font = _PGFT_GetFont(ft, fontobj);

    if (!font) {
        RAISE(PyExc_RuntimeError, _PGFT_GetError(ft));
        return 0;
    }
    return font->family_name;
}

/* All the font metric functions raise an exception and return 0 on an error.
 * It is up to the caller to check PyErr_Occurred for a 0 return value.
 */
long
_PGFT_Font_GetHeight(FreeTypeInstance *ft, PgFontObject *fontobj)
{
    FT_Face font = _PGFT_GetFont(ft, fontobj);

    if (!font) {
        RAISE(PyExc_RuntimeError, _PGFT_GetError(ft));
        return 0;
    }
    return (long)font->height;
}

long
_PGFT_Font_GetHeightSized(FreeTypeInstance *ft, PgFontObject *fontobj,
                          FT_UInt16 ptsize)
{
    FT_Face font = _PGFT_GetFontSized(ft, fontobj, ptsize);

    if (!font) {
        RAISE(PyExc_RuntimeError, _PGFT_GetError(ft));
        return 0;
    }
    return (long)FX6_TRUNC(FX6_CEIL(font->size->metrics.height));
}

long
_PGFT_Font_GetAscender(FreeTypeInstance *ft, PgFontObject *fontobj)
{
    FT_Face font = _PGFT_GetFont(ft, fontobj);

    if (!font) {
        RAISE(PyExc_RuntimeError, _PGFT_GetError(ft));
        return 0;
    }
    return (long)font->ascender;
}

long
_PGFT_Font_GetAscenderSized(FreeTypeInstance *ft, PgFontObject *fontobj,
                           FT_UInt16 ptsize)
{
    FT_Face font = _PGFT_GetFontSized(ft, fontobj, ptsize);

    if (!font) {
        RAISE(PyExc_RuntimeError, _PGFT_GetError(ft));
        return 0;
    }
    return (long)FX6_TRUNC(FX6_CEIL(font->size->metrics.ascender));
}

long
_PGFT_Font_GetDescender(FreeTypeInstance *ft, PgFontObject *fontobj)
{
    FT_Face font = _PGFT_GetFont(ft, fontobj);

    if (!font) {
        RAISE(PyExc_RuntimeError, _PGFT_GetError(ft));
        return 0;
    }
    return (long)font->descender;
}

long
_PGFT_Font_GetDescenderSized(FreeTypeInstance *ft, PgFontObject *fontobj,
                             FT_UInt16 ptsize)
{
    FT_Face font = _PGFT_GetFontSized(ft, fontobj, ptsize);

    if (!font) {
        RAISE(PyExc_RuntimeError, _PGFT_GetError(ft));
        return 0;
    }
    return (long)FX6_TRUNC(FX6_FLOOR(font->size->metrics.descender));
}

long
_PGFT_Font_GetGlyphHeightSized(FreeTypeInstance *ft, PgFontObject *fontobj,
                               FT_UInt16 ptsize)
{
    /*
     * Based on the SDL_ttf height calculation.
     */
    FT_Face font = _PGFT_GetFontSized(ft, fontobj, ptsize);
    FT_Size_Metrics *metrics;

    if (!font) {
        RAISE(PyExc_RuntimeError, _PGFT_GetError(ft));
        return 0;
    }
    metrics = &font->size->metrics;
    return (long)FX6_TRUNC(FX6_CEIL(metrics->ascender) -
                           FX6_FLOOR(metrics->descender)) + /* baseline */ 1;
}

int
_PGFT_GetTextRect(FreeTypeInstance *ft, PgFontObject *fontobj,
                  const FontRenderMode *mode, PGFT_String *text, SDL_Rect *r)
{
    FontText *font_text;
    unsigned width;
    unsigned height;
    FT_Vector offset;
    FT_Pos underline_size;
    FT_Pos underline_top;

    font_text = _PGFT_LoadFontText(ft, fontobj, mode, text);
    if (!font_text) {
        return -1;
    }
    _PGFT_GetRenderMetrics(mode, font_text, &width, &height, &offset,
                           &underline_size, &underline_top);
    r->x = -(Sint16)FX6_TRUNC(FX6_FLOOR(offset.x));
    r->y = (Sint16)FX6_TRUNC(FX6_CEIL(offset.y));
    r->w = (Uint16)width;
    r->h = (Uint16)height;
    return 0;
}


/*********************************************************
 *
 * Font access
 *
 *********************************************************/
FT_Face
_PGFT_GetFontSized(FreeTypeInstance *ft,
    PgFontObject *fontobj,
    int font_size)
{
    FT_Error error;
    FTC_ScalerRec scale;
    FT_Size _fts;

    _PGFT_BuildScaler(fontobj, &scale, font_size);

    error = FTC_Manager_LookupSize(ft->cache_manager,
        &scale, &_fts);

    if (error) {
        _PGFT_SetError(ft, "Failed to resize font", error);
        return 0;
    }

    return _fts->face;
}

FT_Face
_PGFT_GetFont(FreeTypeInstance *ft, PgFontObject *fontobj)
{
    FT_Error error;
    FT_Face font;

    error = FTC_Manager_LookupFace(ft->cache_manager,
        (FTC_FaceID)(&fontobj->id),
        &font);

    if (error) {
        _PGFT_SetError(ft, "Failed to load font", error);
        return 0;
    }

    return font;
}




/*********************************************************
 *
 * Scaling
 *
 *********************************************************/
void
_PGFT_BuildScaler(PgFontObject *fontobj, FTC_Scaler scale, int size)
{
    scale->face_id = (FTC_FaceID)(&fontobj->id);
    scale->width = scale->height = (FT_UInt32)(size * 64);
    scale->pixel = 0;
    scale->x_res = scale->y_res = fontobj->resolution;
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
_PGFT_font_request(FTC_FaceID font_id, FT_Library library,
                   FT_Pointer request_data, FT_Face *afont)
{
    PgFontId *id = (PgFontId *)font_id;
    FT_Error error;

    Py_BEGIN_ALLOW_THREADS;
    error = FT_Open_Face(library, &id->open_args, id->font_index, afont);
    Py_END_ALLOW_THREADS;

    return error;
}



static int init(FreeTypeInstance *ft, PgFontObject *fontobj)
{
    fontobj->_internals = 0;

    if (!_PGFT_GetFont(ft, fontobj)) {
        RAISE(PyExc_IOError, _PGFT_GetError(ft));
        return -1;
    }

    fontobj->_internals = _PGFT_malloc(sizeof(FontInternals));
    if (!fontobj->_internals) {
        PyErr_NoMemory();
        return -1;
    }
    memset(fontobj->_internals, 0x0, sizeof(FontInternals));

    if (_PGFT_FontTextInit(ft, fontobj)) {
        _PGFT_free(fontobj->_internals);
        fontobj->_internals = 0;
        return -1;
    }

    return 0;
}

static void
quit(PgFontObject *fontobj)
{
    if (fontobj->_internals) {
        _PGFT_FontTextFree(fontobj);
        _PGFT_free(fontobj->_internals);
        fontobj->_internals = 0;
    }
}

int
_PGFT_TryLoadFont_Filename(FreeTypeInstance *ft,
    PgFontObject *fontobj,
    const char *filename,
    int font_index)
{
    char *filename_alloc;
    size_t file_len;

    file_len = strlen(filename);
    filename_alloc = _PGFT_malloc(file_len + 1);
    if (!filename_alloc) {
        PyErr_NoMemory();
        return -1;
    }

    strcpy(filename_alloc, filename);
    filename_alloc[file_len] = 0;

    fontobj->id.font_index = font_index;
    fontobj->id.open_args.flags = FT_OPEN_PATHNAME;
    fontobj->id.open_args.pathname = filename_alloc;

    return init(ft, fontobj);
}

#ifdef HAVE_PYGAME_SDL_RWOPS
static unsigned long
RWops_read(FT_Stream stream, unsigned long offset,
           unsigned char *buffer, unsigned long count)
{
    SDL_RWops *src;

    src = (SDL_RWops *)stream->descriptor.pointer;
    SDL_RWseek(src, (int)offset, SEEK_SET);

    if (count == 0)
        return 0;

    return SDL_RWread(src, buffer, 1, (int)count);
}

int
_PGFT_TryLoadFont_RWops(FreeTypeInstance *ft, PgFontObject *fontobj,
                        SDL_RWops *src, int font_index)
{
    FT_Stream stream;
    int position;

    position = SDL_RWtell(src);
    if (position < 0) {
        RAISE(PyExc_SDLError, "Failed to seek in font stream");
        return -1;
    }

    stream = _PGFT_malloc(sizeof(*stream));
    if (!stream) {
        PyErr_NoMemory();
        return -1;
    }
    memset(stream, 0, sizeof(*stream));
    stream->read = RWops_read;
    stream->descriptor.pointer = src;
    stream->pos = (unsigned long)position;
    SDL_RWseek(src, 0, SEEK_END);
    stream->size = (unsigned long)(SDL_RWtell(src) - position);
    SDL_RWseek(src, position, SEEK_SET);

    fontobj->id.font_index = font_index;
    fontobj->id.open_args.flags = FT_OPEN_STREAM;
    fontobj->id.open_args.stream = stream;

    return init(ft, fontobj);
}
#endif

void
_PGFT_UnloadFont(FreeTypeInstance *ft, PgFontObject *fontobj)
{
    if (fontobj->id.open_args.flags == 0)
        return;

    if (ft) {
        FTC_Manager_RemoveFaceID(ft->cache_manager, (FTC_FaceID)(&fontobj->id));
        quit(fontobj);
    }

    if (fontobj->id.open_args.flags == FT_OPEN_STREAM) {
        _PGFT_free(fontobj->id.open_args.pathname);
    fontobj->id.open_args.pathname = 0;
    }
    else if (fontobj->id.open_args.flags == FT_OPEN_PATHNAME) {
        _PGFT_free(fontobj->id.open_args.stream);
    }
    fontobj->id.open_args.flags = 0;
}


/*********************************************************
 *
 * Library (de)initialization
 *
 *********************************************************/
int
_PGFT_Init(FreeTypeInstance **_instance, int cache_size)
{
    FreeTypeInstance *inst = 0;
    int error;

    inst = _PGFT_malloc(sizeof(FreeTypeInstance));

    if (!inst) {
        PyErr_NoMemory();
        goto error_cleanup;
    }

    memset(inst, 0, sizeof(FreeTypeInstance));
    inst->cache_size = cache_size;

    error = FT_Init_FreeType(&inst->library);
    if (error) {
        RAISE(PyExc_RuntimeError,
              "pygame (_PGFT_Init): failed to initialize FreeType library");
        goto error_cleanup;
    }

    if (FTC_Manager_New(inst->library, 0, 0, 0,
            &_PGFT_font_request, 0,
            &inst->cache_manager) != 0) {
        RAISE(PyExc_RuntimeError,
              "pygame (_PGFT_Init): failed to create new FreeType manager");
        goto error_cleanup;
    }

    if (FTC_CMapCache_New(inst->cache_manager,
            &inst->cache_charmap) != 0) {
        RAISE(PyExc_RuntimeError,
              "pygame (_PGFT_Init): failed to create new FreeType cache");
        goto error_cleanup;
    }

    *_instance = inst;
    return 0;

error_cleanup:
    _PGFT_Quit(inst);
    *_instance = 0;

    return -1;
}

void
_PGFT_Quit(FreeTypeInstance *ft)
{
    if (!ft)
        return;

    if (ft->cache_manager)
        FTC_Manager_Done(ft->cache_manager);

    if (ft->library)
        FT_Done_FreeType(ft->library);

    _PGFT_free(ft);
}

