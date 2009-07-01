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

#ifdef HAVE_PYGAME_SDL_VIDEO
#   include "surface.h"
#endif

typedef void (* FontRenderPtr)(int, int, FontSurface *, FT_Bitmap *, PyColor *);

/* blitters */
void __render_glyph_SDL8(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, PyColor *color);
void __render_glyph_SDL16(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, PyColor *color);
void __render_glyph_SDL24(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, PyColor *color);
void __render_glyph_SDL32(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, PyColor *color);
void __render_glyph_ByteArray(int x, int y, FontSurface *surface, FT_Bitmap *bitmap, PyColor *color);

void 
PGFT_BuildRenderMode(FontRenderMode *mode, float center, int vertical, int antialias, int rotation)
{
    double      radian;
    FT_Fixed    cosinus;
    FT_Fixed    sinus;
    int         angle;

    angle = rotation % 360;
    while (angle < 0) angle += 360;

    mode->kerning_mode = 1;
    mode->kerning_degree = 0;
    mode->center = (FT_Fixed)(center * (1 << 16));
    mode->vertical = (FT_Byte)vertical;
    mode->hinted = (FT_Byte)1;
    mode->autohint = 0;
    mode->antialias = (FT_Byte)antialias;
    mode->matrix = NULL;

    if (angle != 0)
    {
        radian  = angle * 3.14159 / 180.0;
        cosinus = (FT_Fixed)( cos( radian ) * 65536.0 );
        sinus   = (FT_Fixed)( sin( radian ) * 65536.0 );

        mode->_rotation_matrix.xx = cosinus;
        mode->_rotation_matrix.yx = sinus;
        mode->_rotation_matrix.xy = -sinus;
        mode->_rotation_matrix.yy = cosinus;

        mode->matrix = &mode->_rotation_matrix;
    }
}


/*********************************************************
 *
 * Rendering on SDL-specific surfaces
 *
 *********************************************************/
#ifdef HAVE_PYGAME_SDL_VIDEO

void __render_glyph_SDL24(int rx, int ry, FontSurface *surface,
    FT_Bitmap *bitmap, PyColor *color)
{
    const int max_x = MIN(rx + bitmap->width, surface->width);
    const int max_y = MIN(ry + bitmap->rows, surface->height);

    FT_Byte *dst = ((FT_Byte *)surface->buffer) + 
        (rx * 3) + (ry * surface->pitch * 3);

    FT_Byte *dst_cpy;

    const FT_Byte *src = bitmap->buffer;
    const FT_Byte *src_cpy;

    FT_UInt32 bgR, bgG, bgB, bgA;
    int j, i;

    if (rx < 0 || ry < 0)
        return;

    for (j = ry; j < max_y; ++j)
    {
        src_cpy = src;
        dst_cpy = dst;

        for (i = rx; i < max_x; ++i, dst_cpy += 3)
        {
            FT_UInt32 alpha = (*src_cpy++); 
            alpha = (alpha * color->a) / 255;

            if (alpha > 0)
            {
                const FT_UInt32 pixel = (FT_UInt32)GET_PIXEL24(dst_cpy);

                GET_RGB_VALS(
                        pixel, surface->format,              
                        bgR, bgG, bgB, bgA);

                ALPHA_BLEND(
                        color->r, color->g, color->b, alpha,
                        bgR, bgG, bgB, bgA);

                SET_PIXEL24_RGB(
                        dst_cpy, surface->format,
                        bgR, bgG, bgB);
            }                                                   
        }                                                       

        dst += surface->pitch * 3;
        src += bitmap->pitch;
    }                                                           
} 

#define _CREATE_SDL_RENDER(bpp, T, _build_pixel)                    \
    void __render_glyph_SDL##bpp(int rx, int ry, FontSurface *surface,\
        FT_Bitmap *bitmap, PyColor *color)                          \
    {                                                               \
        const int max_x = MIN(rx + bitmap->width, surface->width);  \
        const int max_y = MIN(ry + bitmap->rows, surface->height);  \
                                                                    \
        T *dst = ((T*)surface->buffer) + rx + (ry * surface->pitch);\
        T *dst_cpy;                                                 \
                                                                    \
        const FT_Byte *src = bitmap->buffer;                        \
        const FT_Byte *src_cpy;                                     \
                                                                    \
        FT_UInt32 bgR, bgG, bgB, bgA;                               \
        int j, i;                                                   \
                                                                    \
        if (rx < 0 || ry < 0) return;                               \
                                                                    \
        for (j = ry; j < max_y; ++j)                                \
        {                                                           \
            src_cpy = src;                                          \
            dst_cpy = dst;                                          \
                                                                    \
            for (i = rx; i < max_x; ++i, ++dst_cpy)                 \
            {                                                       \
                FT_UInt32 alpha = (*src_cpy++);                     \
                alpha = (alpha * color->a) / 255;                   \
                                                                    \
                if (alpha > 0)                                      \
                {                                                   \
                    GET_RGB_VALS(                                   \
                            *dst_cpy, surface->format,              \
                            bgR, bgG, bgB, bgA);                    \
                                                                    \
                    ALPHA_BLEND(                                    \
                            color->r, color->g, color->b, alpha,    \
                            bgR, bgG, bgB, bgA);                    \
                                                                    \
                    *dst_cpy = (T)(_build_pixel);                   \
                }                                                   \
            }                                                       \
                                                                    \
            dst += surface->pitch;                                  \
            src += bitmap->pitch;                                   \
        }                                                           \
    }

#define BUILD_PIXEL_TRUECOLOR (                                     \
    ((bgR >> surface->format->Rloss) << surface->format->Rshift) |  \
    ((bgG >> surface->format->Gloss) << surface->format->Gshift) |  \
    ((bgB >> surface->format->Bloss) << surface->format->Bshift) |  \
    ((255 >> surface->format->Aloss) << surface->format->Ashift  &  \
     surface->format->Amask)                                        )

#define BUILD_PIXEL_GENERIC (                           \
        SDL_MapRGB(surface->format,                     \
            (FT_Byte)bgR, (FT_Byte)bgG, (FT_Byte)bgB)   )

_CREATE_SDL_RENDER(32,  FT_UInt32,  BUILD_PIXEL_TRUECOLOR)
_CREATE_SDL_RENDER(16,  FT_UInt16,  BUILD_PIXEL_TRUECOLOR)
_CREATE_SDL_RENDER(8,   FT_Byte,    BUILD_PIXEL_GENERIC)

int PGFT_Render_ExistingSurface(FreeTypeInstance *ft, PyFreeTypeFont *font,
    int font_size, FontRenderMode *render, PyObject *text, PySDLSurface *_surface, 
    int x, int y, PyColor *fgcolor, int *_width, int *_height)
{
    static const FontRenderPtr __renderFuncs[] =
    {
        NULL,
        __render_glyph_SDL8,
        __render_glyph_SDL16,
        __render_glyph_SDL24,
        __render_glyph_SDL32
    };

    int width, glyph_height, height, locked = 0;

    SDL_Surface *surface = PySDLSurface_AsSDLSurface(_surface);
    FontSurface font_surf;
    FontText *font_text;

    if (SDL_MUSTLOCK(surface))
    {
        if (SDL_LockSurface(surface) == -1)
        {
            _PGFT_SetError(ft, SDL_GetError (), 0);
            SDL_FreeSurface(surface);
            return -1;
        }
        locked = 1;
    }

    /* build font text */
    font_text = PGFT_LoadFontText(ft, font, font_size, render, text);

    if (!font_text)
        return -1;

    _PGFT_GetTextSize_INTERNAL(ft, font, font_size, render, font_text, &width, &height);

#if 1
    fprintf(stderr, "Drawing @ (%d, %d, %d, %d)\n", x, y, width, height);

    SDL_Rect fill = {x, y, width, height};
    SDL_FillRect(surface, &fill, 0x00FFAA00);
#endif

    /*
     * Setup target surface struct
     */
    font_surf.buffer = surface->pixels;
    font_surf.buffer_cap = ((FT_Byte *)surface->pixels) + (surface->pitch * surface->h);
    font_surf.x_offset = x;
    font_surf.y_offset = y;

    font_surf.width = surface->w;
    font_surf.height = surface->h;
    font_surf.pitch = surface->pitch / surface->format->BytesPerPixel;

    font_surf.format = surface->format;
    font_surf.render = __renderFuncs[surface->format->BytesPerPixel];

    /*
     * Render!
     */
    if (_PGFT_Render_NEW(ft, font, font_text, font_size, fgcolor, &font_surf, render) != 0)
        return -1;

    *_width = width;
    *_height = height;

    if (locked)
        SDL_UnlockSurface(surface);

    return 0;
}

PyObject *PGFT_Render_NewSurface(FreeTypeInstance *ft, PyFreeTypeFont *font, 
    int ptsize, FontRenderMode *render, PyObject *text,
    PyColor *fgcolor, PyColor *bgcolor, int *_width, int *_height)
{
    int width, height, locked = 0;
    FT_UInt32 fillcolor, rmask, gmask, bmask, amask;
    SDL_Surface *surface = NULL;

    FontSurface font_surf;
    FontText *font_text;

    /* build font text */
    font_text = PGFT_LoadFontText(ft, font, ptsize, render, text);

    if (!font_text)
        return NULL;

    if (_PGFT_GetTextSize_INTERNAL(ft, font, ptsize, render, 
                font_text, &width, &height) != 0)
        return NULL;

#if SDL_BYTEORDER == SDL_BIG_ENDIAN
    rmask = 0xff000000;
    gmask = 0x00ff0000;
    bmask = 0x0000ff00;
    amask = 0x000000ff;
#else
    rmask = 0x000000ff;
    gmask = 0x0000ff00;
    bmask = 0x00ff0000;
    amask = 0xff000000;
#endif

    surface = SDL_CreateRGBSurface(SDL_SWSURFACE, width, height, 
            32, rmask, gmask, bmask, amask);

    if (!surface)
    {
        _PGFT_SetError(ft, SDL_GetError (), 0);
        return NULL;
    }

    if (SDL_MUSTLOCK(surface))
    {
        if (SDL_LockSurface(surface) == -1)
        {
            _PGFT_SetError(ft, SDL_GetError (), 0);
            SDL_FreeSurface(surface);
            return NULL;
        }
        locked = 1;
    }

    if (bgcolor)
    {
        fillcolor = SDL_MapRGBA(surface->format, 
                bgcolor->r, bgcolor->g, bgcolor->b, 255);
    }
    else
    {
        fillcolor = SDL_MapRGBA(surface->format, 0, 0, 0, 0);
    }

    SDL_FillRect(surface, NULL, fillcolor);

    font_surf.buffer = surface->pixels;
    font_surf.buffer_cap = ((FT_Byte *)surface->pixels) + (surface->pitch * surface->h);
    font_surf.x_offset = font_surf.y_offset = 0;

    font_surf.width = surface->w;
    font_surf.height = surface->h;
    font_surf.pitch = surface->pitch / sizeof(FT_UInt32);

    font_surf.format = surface->format;
    font_surf.render = __render_glyph_SDL32;

    if (_PGFT_Render_NEW(ft, font, font_text, ptsize, fgcolor, &font_surf, render) != 0)
    {
        SDL_FreeSurface(surface);
        return NULL;
    }

    *_width = width;
    *_height = height;

    if (locked)
        SDL_UnlockSurface(surface);

    return PySDLSurface_NewFromSDLSurface(surface);
}
#endif




/*********************************************************
 *
 * Rendering on generic arrays
 *
 *********************************************************/
void __render_glyph_ByteArray(int x, int y, FontSurface *surface,
    FT_Bitmap *bitmap, PyColor *fg_color)
{
    FT_Byte *dst = ((FT_Byte *)surface->buffer) + x + (y * surface->pitch);
    FT_Byte *dst_cpy;

    const FT_Byte *src = bitmap->buffer;
    const FT_Byte *src_cpy;

    int j, i;

    for (j = 0; j < bitmap->rows; ++j)
    {
        src_cpy = src;
        dst_cpy = dst;

        LOOP_UNROLLED4({ *dst_cpy++ = (FT_Byte)(~(*src_cpy++)); }, i, bitmap->width);

        dst += surface->pitch;
        src += bitmap->pitch;
    }
}

PyObject *PGFT_Render_PixelArray(FreeTypeInstance *ft, PyFreeTypeFont *font,
    int ptsize, PyObject *text, int *_width, int *_height)
{
    int width, height, glyph_height;
    FT_Byte *buffer = NULL;
    PyObject *array = NULL;
    FontSurface surf;

    FontText *font_text;
    FontRenderMode render;

    PGFT_BuildRenderMode(&render, 0.0f, 
            FONT_RENDER_HORIZONTAL, FONT_RENDER_ANTIALIAS, 0);

    /* build font text */
    font_text = PGFT_LoadFontText(ft, font, ptsize, &render, text);

    if (!font_text)
        goto cleanup;

    if (_PGFT_GetTextSize_INTERNAL(ft, font, ptsize, &render, 
                font_text, &width, &height) != 0)
        goto cleanup;

    buffer = malloc((size_t)(width * height));
    if (!buffer)
    {
        _PGFT_SetError(ft, "Could not allocate memory", 0);
        goto cleanup;
    }

    memset(buffer, 0xFF, (size_t)(width * height));

    surf.buffer = buffer;
    surf.width = surf.pitch = width;
    surf.height = height;

    surf.format = NULL;
    surf.render = __render_glyph_ByteArray;

    if (_PGFT_Render_NEW(ft, font, font_text, ptsize, 0x0, &surf, &render) != 0)
        goto cleanup;

    *_width = width;
    *_height = height;

    array = Bytes_FromStringAndSize((char *)buffer, width * height);

cleanup:
    if (buffer)
        free(buffer);

    return array;
}




/*********************************************************
 *
 * New rendering algorithm (rotation + veritical drawing)
 *
 *********************************************************/
int _PGFT_Render_NEW(FreeTypeInstance *ft, PyFreeTypeFont *font, 
    FontText *text, int font_size, PyColor *fg_color, FontSurface *surface, 
    FontRenderMode *render)
{
    int n;
    FT_Vector pen, advances[MAX_GLYPHS];
    FT_Face face;
    FT_Error error;

    int x = surface->x_offset;
    int y = surface->y_offset;

    if (render->vertical)
        x += PGFT_TRUNC(PGFT_ROUND(text->size.x / 2));
    else
    {
        y += PGFT_TRUNC(text->size.y);
        y -= PGFT_TRUNC(text->baseline_offset.y);
    }


    /* TODO: return if drawing outside surface */


    /******************************************************
     * Load scaler, size & face
     ******************************************************/
    face = _PGFT_GetFaceSized(ft, font, font_size);

    if (!face)
    {
        _PGFT_SetError(ft, "Failed to resize face", 0);
        return -1;
    }

    /******************************************************
     * Load advance information
     ******************************************************/

    error = PGFT_GetTextAdvances(ft, font, font_size, render, text, advances);

    if (error)
    {
        _PGFT_SetError(ft, "Failed to load advance glyph advances", error);
        return error;
    }

    /******************************************************
     * Prepare pen for drawing 
     ******************************************************/

    /* change to Cartesian coordinates */
    y = surface->height - y;

    /* get the extent, which we store in the last slot */
    pen = advances[text->length - 1];

    pen.x = FT_MulFix(pen.x, render->center); 
    pen.y = FT_MulFix(pen.y, render->center);

    /* get pen position */
    if (render->matrix && FT_IS_SCALABLE(face))
    {
        FT_Vector_Transform(&pen, render->matrix);
        pen.x = (x << 6) - pen.x;
        pen.y = (y << 6) - pen.y;
    }
    else
    {
        pen.x = PGFT_ROUND(( x << 6 ) - pen.x);
        pen.y = PGFT_ROUND(( y << 6 ) - pen.y);
    }


    /******************************************************
     * Draw text
     ******************************************************/

    for (n = 0; n < text->length; ++n)
    {
        FT_Glyph image;
        FT_BBox bbox;

        FontGlyph *glyph = &(text->glyphs[n]);

        if (!glyph->image)
            continue;

        /* copy image */
        error = FT_Glyph_Copy(glyph->image, &image);
        if (error)
            continue;

        if (image->format != FT_GLYPH_FORMAT_BITMAP)
        {
            if (render->vertical)
                error = FT_Glyph_Transform(image, NULL, &glyph->vvector);

            if (!error)
                error = FT_Glyph_Transform(image, render->matrix, &pen);

            if (error)
            {
                FT_Done_Glyph(image);
                continue;
            }
        }
        else
        {
            FT_BitmapGlyph  bitmap = (FT_BitmapGlyph)image;

            if (render->vertical)
            {
                bitmap->left += (glyph->vvector.x + pen.x) >> 6;
                bitmap->top  += (glyph->vvector.x + pen.y) >> 6;
            }
            else
            {
                bitmap->left += pen.x >> 6;
                bitmap->top  += pen.y >> 6;
            }
        }

        if (render->matrix)
            FT_Vector_Transform(advances + n, render->matrix);

        pen.x += advances[n].x;
        pen.y += advances[n].y;

        FT_Glyph_Get_CBox(image, FT_GLYPH_BBOX_PIXELS, &bbox);

        if (bbox.xMax > 0 && bbox.yMax > 0 &&
            bbox.xMin < surface->width &&
            bbox.yMin < surface->height)
        {
            int         left, top;
            FT_Bitmap*  source;
            FT_BitmapGlyph  bitmap;

            if (image->format == FT_GLYPH_FORMAT_OUTLINE)
            {
                FT_Render_Mode render_mode = 
                    render->antialias ?
                    FT_RENDER_MODE_NORMAL :
                    FT_RENDER_MODE_MONO;

                /* render the glyph to a bitmap, don't destroy original */
                FT_Glyph_To_Bitmap(&image, render_mode, NULL, 0);
            }

            if (image->format != FT_GLYPH_FORMAT_BITMAP)
                continue;

            bitmap = (FT_BitmapGlyph)image;
            source = &bitmap->bitmap;

            left = bitmap->left;
            top = bitmap->top;

            top = surface->height - top;

            surface->render(left, top, surface, source, fg_color);
        }

        FT_Done_Glyph(image);
    } /* END OF RENDERING LOOP */

    return error;
}






/*********************************************************
 *
 * Old rendering algorithm (DEPRECATE)
 *
 *********************************************************/
int _PGFT_Render_INTERNAL(FreeTypeInstance *ft, PyFreeTypeFont *font, 
    const FT_UInt16 *text, int font_size, PyColor *fg_color, FontSurface *surface)
{
    const FT_UInt16 *ch;

    FTC_ScalerRec scale;
    FT_Face face;
    FT_Glyph glyph;
    FT_Bitmap *bitmap;

    FT_UInt32 prev_index, cur_index;

    int swapped, use_kerning;
    int pen_x, pen_y;
    int x_advance;

    _PGFT_BuildScaler(font, &scale, font_size);
    face = _PGFT_GetFace(ft, font);

    if (!face)
    {
        _PGFT_SetError(ft, "Failed to cache font face", 0);
        return -1;
    }

    use_kerning = FT_HAS_KERNING(face);
    prev_index = 0;

    /* FIXME: Some way to set the system's default ? */
    swapped = 0;

    pen_x = 0;
    pen_y = surface->glyph_height;

    for (ch = text; *ch; ++ch)
    {
        FT_UInt16 c = *ch;

        if (c == UNICODE_BOM_NATIVE || c == UNICODE_BOM_SWAPPED)
        {
            swapped = (c == UNICODE_BOM_SWAPPED);
            if (text == ch)
                ++text;

            continue;
        }

        if (swapped)
            c = (FT_UInt16)((c << 8) | (c >> 8));

        if (_PGFT_LoadGlyph(ft, font, 1 /* RENDER! */, &scale, c, &glyph, &cur_index) != 0)
            continue; /* FIXME: fail if we cannot find a char? */

        assert(glyph->format == FT_GLYPH_FORMAT_BITMAP);
        bitmap = &((FT_BitmapGlyph)glyph)->bitmap;

        if (use_kerning && prev_index)
        {
            FT_Vector delta;
            FT_Get_Kerning(face, prev_index, cur_index, ft_kerning_default, &delta); 
            pen_x += delta.x >> 6;
        }

        x_advance = (glyph->advance.x + 0x8000) >> 16;

        /*
         * Render bitmap on the surface at coords:
         *      pen_x + bitmap->left, pen_y - bitmap->top
         */
        {
            const int left = pen_x + ((FT_BitmapGlyph)glyph)->left;
            const int top = pen_y - ((FT_BitmapGlyph)glyph)->top;

            surface->render(left, top, surface, bitmap, fg_color);
        }

        /* FIXME: Why the extra pixel? */
        pen_x += x_advance; /* + 1; */
        prev_index = cur_index;
    }

    return 0;
}
