/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners, 2008 Marcus von Appen

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
#include "surface.h"
#include "pgopengl.h"
#include "tga.h"

#ifdef HAVE_PNG
#include "pgpng.h"
#endif

#ifdef HAVE_JPEG
#include "jpg.h"
#endif

static SDL_Surface* _convert_opengl_sdl (void);

static SDL_Surface*
_convert_opengl_sdl (void)
{
    SDL_Surface *surf;
    Uint32 rmask, gmask, bmask;
    int i;
    unsigned char *pixels;
    GL_glReadPixels_Func p_glReadPixels = NULL;

    pixels = NULL;
    surf = NULL;

    p_glReadPixels = (GL_glReadPixels_Func)
        SDL_GL_GetProcAddress ("glReadPixels"); 

    surf = SDL_GetVideoSurface ();

    if(!surf)
        return NULL;

    if (!p_glReadPixels)
    {
        SDL_SetError ("cannot find glReadPixels function");
        return NULL;
    }

    pixels = malloc ((size_t) surf->w * surf->h * 3);

    if (!pixels)
    {
        SDL_SetError ("could not allocate memory");
        return NULL;
    }

    /* GL_RGB, GL_UNSIGNED_BYTE */
    p_glReadPixels (0, 0, surf->w, surf->h, 0x1907, 0x1401, pixels);

    if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
    {
        rmask = 0x000000FF;
        gmask = 0x0000FF00;
        bmask = 0x00FF0000;
    }
    else
    {
        rmask = 0x00FF0000;
        gmask = 0x0000FF00;
        bmask = 0x000000FF;
    }
    surf = SDL_CreateRGBSurface (SDL_SWSURFACE, surf->w, surf->h, 24,
        rmask, gmask, bmask, 0);

    if (!surf)
    {
        free (pixels);
        return NULL;
    }

    for (i = 0; i < surf->h; ++i)
    {
        memcpy (((char *) surf->pixels) + surf->pitch * i,
            pixels + 3 * surf->w * (surf->h - i - 1), (size_t) surf->w * 3);
    }

    free (pixels);
    return surf;
}

int
pyg_sdlsurface_save (SDL_Surface *surface, char *filename, char *type)
{    
    size_t len;
    SDL_Surface *tmpsf = NULL;
    SDL_RWops *rw;

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return 0;
    }
    if (!filename)
    {
        SDL_SetError ("filename argument NULL");
        return 0;
    }

    if (!type)
    {
        len = strlen (filename);
        if (len < 4)
        {
            SDL_SetError ("unknown file type");
            return 0;
        }
        type = filename + (len - 3);
    }

    len = strlen (type);
    if (len < 3 || len > 4)
    {
        SDL_SetError ("unknown file type");
        if (tmpsf)
            SDL_FreeSurface (tmpsf);
        return 0;
    }

    rw = SDL_RWFromFile (filename, "wb");
    if (!rw)
        return 0;

    return pyg_sdlsurface_save_rw (surface, rw, type, 1);
}

int
pyg_sdlsurface_save_rw (SDL_Surface *surface, SDL_RWops *rw, char *type,
    int freerw)
{
    int retval;
    size_t len;
    SDL_Surface *tmpsf = NULL;

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return 0;
    }

    if (!type)
    {
        SDL_SetError ("type argument NULL");
        return 0;
    }

    if (!rw)
    {
        SDL_SetError ("rw argument NULL");
        return 0;
    }

    len = strlen (type);
    if (len < 3 || len > 4)
    {
        SDL_SetError ("unknown file type");
        return 0;
    }
    
    /* Convert an OpenGL surface on demand. */
    if (surface->flags & SDL_OPENGL)
    {
        /* TODO: _convert_opengl_sdl() acquires the video surface -
         * is this correct? */
        tmpsf = _convert_opengl_sdl ();
        if (!tmpsf)
            return 0;
        surface = tmpsf;
    }

    if (len == 3)
    {
        /* Can be BMP, TGA, PNG, JPG */
        if ((type[0] == 'B' || type[0] == 'b') &&
            (type[1] == 'M' || type[1] == 'm') &&
            (type[2] == 'P' || type[2] == 'p'))
        {
            if (SDL_SaveBMP_RW (surface, rw, freerw) == 0)
                retval = 1;
            else
                retval = 0;
        }
        else if ((type[0] == 'T' || type[0] == 't') &&
            (type[1] == 'G' || type[1] == 'g') &&
            (type[2] == 'A' || type[2] == 'a'))
        {
            /* TGA saving */
            retval = pyg_save_tga_rw (surface, rw, 1, freerw);
        }
#ifdef HAVE_PNG
        else if ((type[0] == 'P' || type[0] == 'p') &&
            (type[1] == 'N' || type[1] == 'n') &&
            (type[2] == 'G' || type[2] == 'g'))
        {
            /* PNG saving. */
            retval = pyg_save_png_rw (surface, rw, freerw);
        }
#endif /* HAVE_PNG */
#ifdef HAVE_JPEG
        else if ((type[0] == 'J' || type[0] == 'j') &&
            (type[1] == 'P' || type[1] == 'p') &&
            (type[2] == 'G' || type[2] == 'g'))
        {
            /* JPG saving */
            retval = pyg_save_jpeg_rw (surface, rw, freerw);
        }
#endif /* HAVE_JPEG */
        else
        {
            SDL_SetError ("unknown file type");
            if (tmpsf)
                SDL_FreeSurface (tmpsf);
            return 0;
        }
    }
    else
    {
#ifdef HAVE_JPEG
        /* JPEG */
        if ((type[0] == 'J' || type[0] == 'j') &&
            (type[1] == 'P' || type[1] == 'p') &&
            (type[2] == 'E' || type[2] == 'e') &&
            (type[3] == 'G' || type[3] == 'g'))
        {
            retval = pyg_save_jpeg_rw (surface, rw, freerw);
        }
        else
#endif /* HAVE_JPEG */
        {
            SDL_SetError ("unknown file type");
            if (tmpsf)
                SDL_FreeSurface (tmpsf);
            return 0;
        }
    }

    return retval;
}
