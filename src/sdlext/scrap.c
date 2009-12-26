/*
  pygame - Python Game Library
  Copyright (C) 2006, 2007 Rene Dudfield, Marcus von Appen

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

#include "scrap.h"

#ifdef SDL_VIDEO_DRIVER_X11
#include "scrap_x11.h"
#endif
#if defined(SDL_VIDEO_DRIVER_WINDIB) || defined(SDL_VIDEO_DRIVER_DDRAW) || defined(SDL_VIDEO_DRIVER_GAPI)
#include "scrap_win.h"
#endif

struct _ScrapInfo
{
    int       (*init)(void);
    void      (*quit)(void);
    int       (*contains)(char*);
    int       (*lost)(void);
    ScrapType (*get_mode)(void);
    ScrapType (*set_mode)(ScrapType);
    int       (*get)(char*, char**, unsigned int*);
    int       (*put)(char*, char*, unsigned int);
    int       (*get_types)(char**);
};

static int _initialized = 0;
static struct _ScrapInfo _scrapinfo = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

#define ASSERT_INITIALIZED(x)                                   \
    if (!_initialized)                                          \
    {                                                           \
        SDL_SetError ("scrap system has not been initialized"); \
        return (x);                                             \
    }                                                           \

int
pyg_scrap_init (void)
{
#ifdef SDL_VIDEO_DRIVER_X11
    /* X11 clipboard */
    _scrapinfo.init = scrap_init_x11;
    _scrapinfo.quit = scrap_quit_x11;
    _scrapinfo.contains = scrap_contains_x11;
    _scrapinfo.lost = scrap_lost_x11;
    _scrapinfo.get_mode = scrap_get_mode_x11;
    _scrapinfo.set_mode = scrap_set_mode_x11;
    _scrapinfo.get = scrap_get_x11;
    _scrapinfo.put = scrap_put_x11;
    _scrapinfo.get_types = scrap_get_types_x11;
    _initialized = 1;
#elif defined(SDL_VIDEO_DRIVER_WINDIB) || defined(SDL_VIDEO_DRIVER_DDRAW) || defined(SDL_VIDEO_DRIVER_GAPI)
    /* Win32 clipboard */
    _scrapinfo.init = scrap_init_win;
    _scrapinfo.quit = scrap_quit_win;
    _scrapinfo.contains = scrap_contains_win;
    _scrapinfo.lost = scrap_lost_win;
    _scrapinfo.get_mode = scrap_get_mode_win;
    _scrapinfo.set_mode = scrap_set_mode_win;
    _scrapinfo.get = scrap_get_win;
    _scrapinfo.put = scrap_put_win;
    _scrapinfo.get_types = scrap_get_types_win;
    _initialized = 1;
#endif
    if (_initialized)
        return _scrapinfo.init ();

    SDL_SetError ("no scrap support for the target system available");
    return 0;
}

int
pyg_scrap_was_init (void)
{
    return _initialized;
}

void
pyg_scrap_quit (void)
{
    if (!_initialized)
        return;
    _initialized = 0;
    _scrapinfo.quit ();
}

int
pyg_scrap_contains (char *type)
{
    ASSERT_INITIALIZED(-1);
    return _scrapinfo.contains (type);
}

int
pyg_scrap_lost (void)
{
    ASSERT_INITIALIZED(-1);
    return _scrapinfo.lost ();
}

ScrapType
pyg_scrap_get_mode (void)
{
    ASSERT_INITIALIZED(-1);
    return _scrapinfo.get_mode ();
}

ScrapType
pyg_scrap_set_mode (ScrapType mode)
{
    ASSERT_INITIALIZED(-1);
    return _scrapinfo.set_mode (mode);
}

int
pyg_scrap_get (char *type, char** data, unsigned int *size)
{
    ASSERT_INITIALIZED(-1);
    return _scrapinfo.get (type, data, size);
}

int
pyg_scrap_put (char *type, char *data, unsigned int size)
{
    ASSERT_INITIALIZED(-1);
    return _scrapinfo.put (type, data, size);
}

int
pyg_scrap_get_types (char** data)
{
    ASSERT_INITIALIZED(-1);
    return _scrapinfo.get_types (data);
}
