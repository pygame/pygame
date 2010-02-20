/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners, 2006 Rene Dudfield

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
#ifdef HAVE_JPEG

#include "jpg.h"
#include <jpeglib.h>

#define OUTPUT_SIZE 4096

struct dest_mgr
{
    struct jpeg_destination_mgr  pub;
    SDL_RWops                   *rw;
    Uint8                       buf[OUTPUT_SIZE];
};

static void _init_jpegdest (j_compress_ptr cinfo);
static boolean _empty_jpegbuffer (j_compress_ptr cinfo);
static void _term_jpegdest (j_compress_ptr cinfo);
static int _write_jpeg (SDL_RWops *rw, unsigned char** image_buffer,
    int image_width, int image_height, int quality);

static void
_init_jpegdest (j_compress_ptr cinfo)
{
    struct dest_mgr *dest = (struct dest_mgr *) cinfo->dest;
    dest->pub.next_output_byte = dest->buf;
    dest->pub.free_in_buffer = OUTPUT_SIZE;
}

static boolean
_empty_jpegbuffer (j_compress_ptr cinfo)
{
    struct dest_mgr *dest = (struct dest_mgr *) cinfo->dest;
    int size;
    
    size = SDL_RWwrite (dest->rw, dest->buf, 1, OUTPUT_SIZE);
    if (size != OUTPUT_SIZE)
        return FALSE;
    dest->pub.next_output_byte = dest->buf;
    dest->pub.free_in_buffer = OUTPUT_SIZE;
    return TRUE;
}

static void
_term_jpegdest (j_compress_ptr cinfo)
{
    struct dest_mgr *dest = (struct dest_mgr *) cinfo->dest;
    SDL_RWwrite (dest->rw, dest->buf, 1,
        OUTPUT_SIZE - dest->pub.free_in_buffer);
}

static int
_write_jpeg (SDL_RWops *rw, unsigned char** image_buffer, int image_width,
    int image_height, int quality)
{
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    struct dest_mgr *dest;
    JSAMPROW row_pointer[1];

    cinfo.err = jpeg_std_error (&jerr);
    jpeg_create_compress (&cinfo);

    /*jpeg_stdio_dest (&cinfo, outfile);*/

    cinfo.image_width = image_width;
    cinfo.image_height = image_height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults (&cinfo);
    jpeg_set_quality (&cinfo, quality, TRUE);

    cinfo.dest = (struct jpeg_destination_mgr *)
        (*cinfo.mem->alloc_small) ((j_common_ptr) &cinfo, JPOOL_PERMANENT,
            sizeof(struct dest_mgr));
    if (!cinfo.dest)
        return 0;

    dest = (struct dest_mgr *) cinfo.dest;
    dest->pub.init_destination = _init_jpegdest;
    dest->pub.term_destination = _term_jpegdest;
    dest->pub.empty_output_buffer = _empty_jpegbuffer;
    dest->rw = rw;
    dest->pub.free_in_buffer = 0;
    dest->pub.next_output_byte = NULL;

    jpeg_start_compress (&cinfo, TRUE);

    while (cinfo.next_scanline < cinfo.image_height)
    {
        row_pointer[0] = image_buffer[cinfo.next_scanline];
        (void) jpeg_write_scanlines (&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress (&cinfo);
    jpeg_destroy_compress (&cinfo);
    return 1;
}

int
pyg_save_jpeg (SDL_Surface *surface, char *file)
{
    SDL_RWops *out;

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return 0;
    }
    if (!file)
    {
        SDL_SetError ("file argument NULL");
        return 0;
    }

    out = SDL_RWFromFile (file, "wb");
    if (!out)
        return 0;
    return pyg_save_jpeg_rw (surface, out, 1);
}

int
pyg_save_jpeg_rw (SDL_Surface *surface, SDL_RWops *rw, int freerw)
{
    static unsigned char** ss_rows;
    static int ss_size;
    static int ss_w, ss_h;
    SDL_Surface *ss_surface;
    SDL_Rect ss_rect;
    int r, i;
    int pixel_bits = 32;

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return 0;
    }
    if (!rw)
    {
        SDL_SetError ("rw argument NULL");
        return 0;
    }

    ss_rows = 0;
    ss_size = 0;
    ss_surface = NULL;

    ss_w = surface->w;
    ss_h = surface->h;

    pixel_bits = 24;

#if SDL_BYTEORDER == SDL_BIG_ENDIAN
    ss_surface = SDL_CreateRGBSurface (SDL_SWSURFACE|SDL_SRCALPHA,
        ss_w, ss_h, pixel_bits, 0xff0000, 0xff00, 0xff, 0x000000ff);
#else
    ss_surface = SDL_CreateRGBSurface (SDL_SWSURFACE|SDL_SRCALPHA,
        ss_w, ss_h, pixel_bits, 0xff, 0xff00, 0xff0000, 0xff000000);
#endif

    if (ss_surface == NULL)
        return 0;

    ss_rect.x = 0;
    ss_rect.y = 0;
    ss_rect.w = ss_w;
    ss_rect.h = ss_h;
    SDL_BlitSurface (surface, &ss_rect, ss_surface, NULL);

    if (ss_size == 0)
    {
        ss_size = ss_h;
        ss_rows = (unsigned char**) malloc (sizeof (unsigned char*) * ss_size);
        if(ss_rows == NULL)
        {
            SDL_FreeSurface (ss_surface);
            return 0;
        }
    }

    for (i = 0; i < ss_h; i++)
    {
        ss_rows[i] = ((unsigned char*)ss_surface->pixels) +
            i * ss_surface->pitch;
    }
    r = _write_jpeg (rw, ss_rows, surface->w, surface->h, 100);

    free (ss_rows);
    SDL_FreeSurface (ss_surface);
    ss_surface = NULL;
    if (freerw)
        SDL_RWclose (rw);
    return r;
}

#endif /* HAVE_JPEG */
