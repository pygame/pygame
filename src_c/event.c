/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners

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

  Pete Shinners
  pete@shinners.org
*/

/*
 *  pygame event module
 */
#define PYGAMEAPI_EVENT_INTERNAL

#include "pygame.h"

#include "pgcompat.h"

#include "doc/event_doc.h"

#include "structmember.h"

// The system message code is only tested on windows, so only
//   include it there for now.
#include <SDL_syswm.h>

#if IS_SDLv2
/*only register one block of user events.*/
static int have_registered_events = 0;

#define JOYEVENT_INSTANCE_ID "instance_id"
#define JOYEVENT_DEVICE_INDEX "device_index"

/* Define custom functions for peep events, for SDL1/2 compat */
#define PG_PEEP_EVENT(a, b, c, d) SDL_PeepEvents(a, b, c, d, d)
#define PG_PEEP_EVENT_ALL(x, y, z) \
    SDL_PeepEvents(x, y, z, SDL_FIRSTEVENT, SDL_LASTEVENT)

#else /* IS_SLDv1 */

#define JOYEVENT_INSTANCE_ID "joy"
#define JOYEVENT_DEVICE_INDEX "joy"

#define PG_PEEP_EVENT(a, b, c, d) SDL_PeepEvents(a, b, c, SDL_EVENTMASK(d))
#define PG_PEEP_EVENT_ALL(x, y, z) SDL_PeepEvents(x, y, z, SDL_ALLEVENTS)

#endif /* IS_SLDv1 */

/* These are used for checks. The checks are kinda redundant because we
 * have proxy events anyways, but this is needed for SDL1 */
#define USEROBJ_CHECK (Sint32)0xFEEDF00D

#define MAX_UINT32 0xFFFFFFFF

#define PG_GET_LIST_LEN 128

// Map joystick instance IDs to device ids for partial backwards compatibility
static PyObject *joy_instance_map = NULL;

/* _custom_event stores the next custom user event type that will be
 * returned by pygame.event.custom_type() */
#define _PGE_CUSTOM_EVENT_INIT PGE_USEREVENT + 1

static int _custom_event = _PGE_CUSTOM_EVENT_INIT;
static int _pg_event_is_init = 0;

/* Length of our unicode string in bytes. We need 1 to 3 bytes to store
 * our unicode data, so we use a length of 4, to include the NULL byte
 * at the end as well */
#define UNICODE_LEN 4

#if IS_SDLv2

/* This defines the maximum values of key-press and unicode values we
 * can store at a time, it is used for determining the unicode attribute
 * for KEYUP events. Now that its set to 15, it means that a user can
 * simultaneously hold 15 keys (who would do that?) and on release, all
 * KEYUP events will have unicode attribute. Why 15? You can set any
 * arbitrary number you like ;) */
#define MAX_SCAN_UNICODE 15

static struct ScanAndUnicode {
    SDL_Scancode key;
    char unicode[UNICODE_LEN];
} scanunicode[MAX_SCAN_UNICODE] = {{ 0 }};

static int pg_key_repeat_delay = 0;
static int pg_key_repeat_interval = 0;

static SDL_TimerID _pg_repeat_timer = 0;
static SDL_Event _pg_repeat_event;

static Uint32
_pg_repeat_callback(Uint32 interval, void *param)
{
    _pg_repeat_event.type = PGE_KEYREPEAT;
    _pg_repeat_event.key.state = SDL_PRESSED;
    _pg_repeat_event.key.repeat = 1;
    SDL_PushEvent(&_pg_repeat_event);
    return pg_key_repeat_interval;
}

/* This function attempts to determine the unicode attribute from
 * the keydown/keyup event. This is used as a last-resort, incase we
 * could not determine the unicode from TEXTINPUT feild. Why?
 * Because this function is really basic and cannot determine the
 * fancy unicode characters, just the basic ones
 *
 * One more advantage of this function is that it can return unicode
 * for some keys which TEXTINPUT does not provide (which unicode
 * attribute of SDL1 provided) */
static char
_pg_unicode_from_event(SDL_Event *event)
{
    int capsheld = event->key.keysym.mod & KMOD_CAPS;
    int shiftheld = event->key.keysym.mod & KMOD_SHIFT;

    int capitalize = (capsheld && !shiftheld) || (shiftheld && !capsheld);
    SDL_Keycode key = event->key.keysym.sym;

    if (event->key.keysym.mod & KMOD_CTRL) {
        /* Contol Key held, send control-key related unicode. */
        if (key >= SDLK_a && key <= SDLK_z)
            return key - SDLK_a + 1;
        else {
            switch (key) {
                case SDLK_2:
                case SDLK_AT:
                    return '\0';
                case SDLK_3:
                case SDLK_LEFTBRACKET:
                    return '\x1b';
                case SDLK_4:
                case SDLK_BACKSLASH:
                    return '\x1c';
                case SDLK_5:
                case SDLK_RIGHTBRACKET:
                    return '\x1d';
                case SDLK_6:
                case SDLK_CARET:
                    return '\x1e';
                case SDLK_7:
                case SDLK_UNDERSCORE:
                    return '\x1f';
                case SDLK_8:
                    return '\x7f';
            }
        }
    }
    if (key < 128) {
        if (capitalize && key >= SDLK_a && key <= SDLK_z)
            return key + 'A' - 'a';
        return key;
    }

    switch (key) {
        case SDLK_KP_PERIOD:
            return '.';
        case SDLK_KP_DIVIDE:
            return '/';
        case SDLK_KP_MULTIPLY:
            return '*';
        case SDLK_KP_MINUS:
            return '-';
        case SDLK_KP_PLUS:
            return '+';
        case SDLK_KP_ENTER:
            return '\r';
        case SDLK_KP_EQUALS:
            return '=';
    }
    return '\0';
}

/* Strip a utf-8 encoded string to contain only first character. Also
 * ensure that character can be represented within 3 bytes, because SDL1
 * did not support unicode characters that took up 4 bytes. Incase this
 * bit of code is not clear, here is a python equivalent
def _pg_strip_utf8(string):
    if chr(string[0]) <= 0xFFFF:
        return string[0]
    else:
        return ""
*/
static char *
_pg_strip_utf8(char *str)
{
    char *retptr;
    char ret[UNICODE_LEN] = { 0 };
    Uint8 firstbyte;

    memcpy(&firstbyte, str, 1);

    /* 1111 0000 is 0xF0 */
    if (firstbyte < 0xF0) {
        /* 1110 0000 is 0xE0 */
        if (firstbyte >= 0xE0) {
            /* Copy first 3 bytes */
            memcpy(&ret, str, 3);
        }
        /* 1100 0000 is 0xC0 */
        else if (firstbyte >= 0xC0) {
            /* Copy first 2 bytes */
            memcpy(&ret, str, 2);
        }
        /* 1000 0000 is 0x80 */
        else if (firstbyte < 0x80) {
            /* Copy first byte */
            memcpy(&ret, str, 1);
        }
    }
    retptr = PyMem_New(char, UNICODE_LEN);
    memcpy(retptr, &ret, UNICODE_LEN);
    return retptr;
}

static int
_pg_put_event_unicode(SDL_Event *event, char *uni)
{
    int i;
    char *temp;
    for (i=0; i < MAX_SCAN_UNICODE; i++) {
        if (!scanunicode[i].key) {
            scanunicode[i].key = event->key.keysym.scancode;
            temp = _pg_strip_utf8(uni);
            memcpy(scanunicode[i].unicode, temp, UNICODE_LEN);
            PyMem_Del(temp);
            return 1;
        }
    }
    return 0;
}

static PyObject *
_pg_get_event_unicode(SDL_Event *event)
{
    char c;
    int i;
    for (i=0; i < MAX_SCAN_UNICODE; i++) {
        if (scanunicode[i].key == event->key.keysym.scancode) {
            if (event->type == SDL_KEYUP) {
                /* mark the position as free real estate for other
                 * events to occupy. */
                scanunicode[i].key = 0;
            }
            /* Dont use Text_FromUTF8 here */
            return PyUnicode_FromString(scanunicode[i].unicode);
        }
    }
    /* fallback to function that determines unicode from the event.
     * We try to get the unicode attribute, and store it in memory*/
    c = _pg_unicode_from_event(event);
    if (_pg_put_event_unicode(event, &c))
        return _pg_get_event_unicode(event);
    return PyUnicode_FromString("");
}

#else /* IS_SDLv1 */

/* Convert a Uint16 unicode codepoint to Python Unicode Object
 * This is the same as python 2 unichr() function and almost same as
 * python 3 chr() function, except it does not support numbers larger
 * than the limit of Uint16. */
static PyObject *
_pg_chr(Uint16 uni)
{
    char ret[UNICODE_LEN] = { 0 };

    if (uni < 0x80) {
        /* We can UTF-8 encode it within a single byte */
        ret[0] = (uni & 0xFF);
    }
    else if (uni < 0x0800) {
        /* We can UTF-8 encode it within 2 bytes */
        ret[0] = 0xC0; /* binary: 1100 0000 */
        ret[1] = 0x80; /* binary: 1000 0000 */

        /* binary: 0000 0111 1100 0000 is 0x07C0 */
        /* binary: 0000 0000 0011 1111 is 0x3F */
        ret[0] |= ((uni & 0x07C0) >> 6);
        ret[1] |= (uni & 0x3F);
    }
    else {
        /* We can UTF-8 encode it within 3 bytes */
        ret[0] = 0xE0; /* binary: 1110 0000 */
        ret[1] = 0x80; /* binary: 1000 0000 */
        ret[2] = 0x80; /* binary: 1000 0000 */

        /* binary: 1111 0000 0000 0000 is 0xF000 */
        /* binary: 0000 1111 1100 0000 is 0x0FC0 */
        /* binary: 0000 0000 0011 1111 is 0x003F */
        ret[0] |= ((uni & 0xF000) >> 12);
        ret[1] |= ((uni & 0x0FC0) >> 6);
        ret[2] |= (uni & 0x3F);
    }
    /* You may be thinking why we are not handling unicode that is
     * represented in 4 bytes. Because our input is Uint16, there is
     * no chance that our input needs 4 bytes for encoding */
    return PyUnicode_FromString(ret); /* Dont use Text_FromUTF8 here */
}

#endif /* IS_SDLv1 */

/* The next two functions are used for proxying SDL events to and from
 * PGPOST_* events. These functions do NOT proxy on SDL1.
 *
 * Some SDL1 events (SDL_ACTIVEEVENT, SDL_VIDEORESIZE and SDL_VIDEOEXPOSE)
 * are redefined with SDL2, they HAVE to be proxied.
 *
 * SDL_USEREVENT is not proxied, because with SDL2, pygame assignes a
 * different event in place of SDL_USEREVENT, and users use PGE_USEREVENT
 *
 * Each WINDOW_* event must be defined twice, once as an event, and also
 * again, as a proxy event. WINDOW_* events MUST be proxied.
 */

static Uint32
_pg_pgevent_proxify(Uint32 type)
{
#if IS_SDLv1
    return type;
#else /* IS_SDLv2 */
    switch (type) {
        case SDL_ACTIVEEVENT:
            return PGPOST_ACTIVEEVENT;
#ifdef SDL2_AUDIODEVICE_SUPPORTED
        case SDL_AUDIODEVICEADDED:
            return PGPOST_AUDIODEVICEADDED;
        case SDL_AUDIODEVICEREMOVED:
            return PGPOST_AUDIODEVICEREMOVED;
#endif /* SDL2_AUDIODEVICE_SUPPORTED */
        case SDL_CONTROLLERAXISMOTION:
            return PGPOST_CONTROLLERAXISMOTION;
        case SDL_CONTROLLERBUTTONDOWN:
            return PGPOST_CONTROLLERBUTTONDOWN;
        case SDL_CONTROLLERBUTTONUP:
            return PGPOST_CONTROLLERBUTTONUP;
        case SDL_CONTROLLERDEVICEADDED:
            return PGPOST_CONTROLLERDEVICEADDED;
        case SDL_CONTROLLERDEVICEREMOVED:
            return PGPOST_CONTROLLERDEVICEREMOVED;
        case SDL_CONTROLLERDEVICEREMAPPED:
            return PGPOST_CONTROLLERDEVICEREMAPPED;
        case SDL_DOLLARGESTURE:
            return PGPOST_DOLLARGESTURE;
        case SDL_DOLLARRECORD:
            return PGPOST_DOLLARRECORD;
        case SDL_DROPFILE:
            return PGPOST_DROPFILE;
#if SDL_VERSION_ATLEAST(2, 0, 5)
        case SDL_DROPTEXT:
            return PGPOST_DROPTEXT;
        case SDL_DROPBEGIN:
            return PGPOST_DROPBEGIN;
        case SDL_DROPCOMPLETE:
            return PGPOST_DROPCOMPLETE;
#endif /* SDL_VERSION_ATLEAST(2, 0, 5) */
        case SDL_FINGERMOTION:
            return PGPOST_FINGERMOTION;
        case SDL_FINGERDOWN:
            return PGPOST_FINGERDOWN;
        case SDL_FINGERUP:
            return PGPOST_FINGERUP;
        case SDL_KEYDOWN:
            return PGPOST_KEYDOWN;
        case SDL_KEYUP:
            return PGPOST_KEYUP;
        case SDL_JOYAXISMOTION:
            return PGPOST_JOYAXISMOTION;
        case SDL_JOYBALLMOTION:
            return PGPOST_JOYBALLMOTION;
        case SDL_JOYHATMOTION:
            return PGPOST_JOYHATMOTION;
        case SDL_JOYBUTTONDOWN:
            return PGPOST_JOYBUTTONDOWN;
        case SDL_JOYBUTTONUP:
            return PGPOST_JOYBUTTONUP;
        case SDL_JOYDEVICEADDED:
            return PGPOST_JOYDEVICEADDED;
        case SDL_JOYDEVICEREMOVED:
            return PGPOST_JOYDEVICEREMOVED;
        case PGE_MIDIIN:
            return PGPOST_MIDIIN;
        case PGE_MIDIOUT:
            return PGPOST_MIDIOUT;
        case SDL_MOUSEMOTION:
            return PGPOST_MOUSEMOTION;
        case SDL_MOUSEBUTTONDOWN:
            return PGPOST_MOUSEBUTTONDOWN;
        case SDL_MOUSEBUTTONUP:
            return PGPOST_MOUSEBUTTONUP;
        case SDL_MOUSEWHEEL:
            return PGPOST_MOUSEWHEEL;
        case SDL_MULTIGESTURE:
            return PGPOST_MULTIGESTURE;
        case SDL_NOEVENT:
            return PGPOST_NOEVENT;
        case SDL_QUIT:
            return PGPOST_QUIT;
        case SDL_SYSWMEVENT:
            return PGPOST_SYSWMEVENT;
        case SDL_TEXTEDITING:
            return PGPOST_TEXTEDITING;
        case SDL_TEXTINPUT:
            return PGPOST_TEXTINPUT;
        case SDL_VIDEORESIZE:
            return PGPOST_VIDEORESIZE;
        case SDL_VIDEOEXPOSE:
            return PGPOST_VIDEOEXPOSE;

        case PGE_WINDOWSHOWN:
            return PGPOST_WINDOWSHOWN;
        case PGE_WINDOWHIDDEN:
            return PGPOST_WINDOWHIDDEN;
        case PGE_WINDOWEXPOSED:
            return PGPOST_WINDOWEXPOSED;
        case PGE_WINDOWMOVED:
            return PGPOST_WINDOWMOVED;
        case PGE_WINDOWRESIZED:
            return PGPOST_WINDOWRESIZED;
        case PGE_WINDOWSIZECHANGED:
            return PGPOST_WINDOWSIZECHANGED;
        case PGE_WINDOWMINIMIZED:
            return PGPOST_WINDOWMINIMIZED;
        case PGE_WINDOWMAXIMIZED:
            return PGPOST_WINDOWMAXIMIZED;
        case PGE_WINDOWRESTORED:
            return PGPOST_WINDOWRESTORED;
        case PGE_WINDOWENTER:
            return PGPOST_WINDOWENTER;
        case PGE_WINDOWLEAVE:
            return PGPOST_WINDOWLEAVE;
        case PGE_WINDOWFOCUSGAINED:
            return PGPOST_WINDOWFOCUSGAINED;
        case PGE_WINDOWFOCUSLOST:
            return PGPOST_WINDOWFOCUSLOST;
        case PGE_WINDOWCLOSE:
            return PGPOST_WINDOWCLOSE;
        case PGE_WINDOWTAKEFOCUS:
            return PGPOST_WINDOWTAKEFOCUS;
        case PGE_WINDOWHITTEST:
            return PGPOST_WINDOWHITTEST;
        default:
            return type;
    }
#endif /* IS_SDLv2 */
}

static Uint32
_pg_pgevent_deproxify(Uint32 type)
{
#if IS_SDLv1
    return type;
#else /* IS_SDLv2 */
    switch (type) {
        case PGPOST_ACTIVEEVENT:
            return SDL_ACTIVEEVENT;
#ifdef SDL2_AUDIODEVICE_SUPPORTED
        case PGPOST_AUDIODEVICEADDED:
            return SDL_AUDIODEVICEADDED;
        case PGPOST_AUDIODEVICEREMOVED:
            return SDL_AUDIODEVICEREMOVED;
#endif /* SDL2_AUDIODEVICE_SUPPORTED */
        case PGPOST_CONTROLLERAXISMOTION:
            return SDL_CONTROLLERAXISMOTION;
        case PGPOST_CONTROLLERBUTTONDOWN:
            return SDL_CONTROLLERBUTTONDOWN;
        case PGPOST_CONTROLLERBUTTONUP:
            return SDL_CONTROLLERBUTTONUP;
        case PGPOST_CONTROLLERDEVICEADDED:
            return SDL_CONTROLLERDEVICEADDED;
        case PGPOST_CONTROLLERDEVICEREMOVED:
            return SDL_CONTROLLERDEVICEREMOVED;
        case PGPOST_CONTROLLERDEVICEREMAPPED:
            return SDL_CONTROLLERDEVICEREMAPPED;
        case PGPOST_DOLLARGESTURE:
            return SDL_DOLLARGESTURE;
        case PGPOST_DOLLARRECORD:
            return SDL_DOLLARRECORD;
        case PGPOST_DROPFILE:
            return SDL_DROPFILE;
#if SDL_VERSION_ATLEAST(2, 0, 5)
        case PGPOST_DROPTEXT:
            return SDL_DROPTEXT;
        case PGPOST_DROPBEGIN:
            return SDL_DROPBEGIN;
        case PGPOST_DROPCOMPLETE:
            return SDL_DROPCOMPLETE;
#endif /* SDL_VERSION_ATLEAST(2, 0, 5) */
        case PGPOST_FINGERMOTION:
            return SDL_FINGERMOTION;
        case PGPOST_FINGERDOWN:
            return SDL_FINGERDOWN;
        case PGPOST_FINGERUP:
            return SDL_FINGERUP;
        case PGPOST_KEYDOWN:
            return SDL_KEYDOWN;
        case PGPOST_KEYUP:
            return SDL_KEYUP;
        case PGPOST_JOYAXISMOTION:
            return SDL_JOYAXISMOTION;
        case PGPOST_JOYBALLMOTION:
            return SDL_JOYBALLMOTION;
        case PGPOST_JOYHATMOTION:
            return SDL_JOYHATMOTION;
        case PGPOST_JOYBUTTONDOWN:
            return SDL_JOYBUTTONDOWN;
        case PGPOST_JOYBUTTONUP:
            return SDL_JOYBUTTONUP;
        case PGPOST_JOYDEVICEADDED:
            return SDL_JOYDEVICEADDED;
        case PGPOST_JOYDEVICEREMOVED:
            return SDL_JOYDEVICEREMOVED;
        case PGPOST_MIDIIN:
            return PGE_MIDIIN;
        case PGPOST_MIDIOUT:
            return PGE_MIDIOUT;
        case PGPOST_MOUSEMOTION:
            return SDL_MOUSEMOTION;
        case PGPOST_MOUSEBUTTONDOWN:
            return SDL_MOUSEBUTTONDOWN;
        case PGPOST_MOUSEBUTTONUP:
            return SDL_MOUSEBUTTONUP;
        case PGPOST_MOUSEWHEEL:
            return SDL_MOUSEWHEEL;
        case PGPOST_MULTIGESTURE:
            return SDL_MULTIGESTURE;
        case PGPOST_NOEVENT:
            return SDL_NOEVENT;
        case PGPOST_QUIT:
            return SDL_QUIT;
        case PGPOST_SYSWMEVENT:
            return SDL_SYSWMEVENT;
        case PGPOST_TEXTEDITING:
            return SDL_TEXTEDITING;
        case PGPOST_TEXTINPUT:
            return SDL_TEXTINPUT;
        case PGPOST_VIDEORESIZE:
            return SDL_VIDEORESIZE;
        case PGPOST_VIDEOEXPOSE:
            return SDL_VIDEOEXPOSE;

        case PGPOST_WINDOWSHOWN:
            return PGE_WINDOWSHOWN;
        case PGPOST_WINDOWHIDDEN:
            return PGE_WINDOWHIDDEN;
        case PGPOST_WINDOWEXPOSED:
            return PGE_WINDOWEXPOSED;
        case PGPOST_WINDOWMOVED:
            return PGE_WINDOWMOVED;
        case PGPOST_WINDOWRESIZED:
            return PGE_WINDOWRESIZED;
        case PGPOST_WINDOWSIZECHANGED:
            return PGE_WINDOWSIZECHANGED;
        case PGPOST_WINDOWMINIMIZED:
            return PGE_WINDOWMINIMIZED;
        case PGPOST_WINDOWMAXIMIZED:
            return PGE_WINDOWMAXIMIZED;
        case PGPOST_WINDOWRESTORED:
            return PGE_WINDOWRESTORED;
        case PGPOST_WINDOWENTER:
            return PGE_WINDOWENTER;
        case PGPOST_WINDOWLEAVE:
            return PGE_WINDOWLEAVE;
        case PGPOST_WINDOWFOCUSGAINED:
            return PGE_WINDOWFOCUSGAINED;
        case PGPOST_WINDOWFOCUSLOST:
            return PGE_WINDOWFOCUSLOST;
        case PGPOST_WINDOWCLOSE:
            return PGE_WINDOWCLOSE;
        case PGPOST_WINDOWTAKEFOCUS:
            return PGE_WINDOWTAKEFOCUS;
        case PGPOST_WINDOWHITTEST:
            return PGE_WINDOWHITTEST;
        default:
            return type;
    }
#endif /* IS_SDLv2 */
}

#if IS_SDLv2
static SDL_Event *_pg_last_keydown_event = NULL;

static int
_pg_translate_windowevent(void *_, SDL_Event *event)
{
    if (event->type == SDL_WINDOWEVENT) {
        event->type = PGE_WINDOWSHOWN + event->window.event - 1;
        return SDL_EventState(_pg_pgevent_proxify(event->type), SDL_QUERY);
    }
    return 1;
}

static int SDLCALL
_pg_remove_pending_VIDEORESIZE(void * userdata, SDL_Event *event)
{
    SDL_Event *new_event = (SDL_Event *)userdata;

    if (event->type == SDL_VIDEORESIZE
        && event->window.windowID == new_event->window.windowID) {
        /* We're about to post a new size event, drop the old ones */
        return 0;
    }
    return 1;
}

static int SDLCALL
_pg_remove_pending_VIDEOEXPOSE(void * userdata, SDL_Event *event)
{
    SDL_Event *new_event = (SDL_Event *)userdata;

    if (event->type == SDL_VIDEOEXPOSE
        && event->window.windowID == new_event->window.windowID) {
        /* We're about to post a new videoexpose event, drop the old ones */
        return 0;
    }
    return 1;
}

/* SDL 2 to SDL 1.2 event mapping and SDL 1.2 key repeat emulation,
 * this can alter events in-place */
static int SDLCALL
pg_event_filter(void *_, SDL_Event *event)
{
    SDL_Event newdownevent, newupevent, newevent = *event;
    int x, y, i;

    if (event->type == SDL_WINDOWEVENT) {
        /* DON'T filter SDL_WINDOWEVENTs here. If we delete events, they
         * won't be available to low-level SDL2 either.*/
        switch (event->window.event) {
            case SDL_WINDOWEVENT_RESIZED:
                SDL_FilterEvents(_pg_remove_pending_VIDEORESIZE, &newevent);

                newevent.type = SDL_VIDEORESIZE;
                SDL_PushEvent(&newevent);
                break;
            case SDL_WINDOWEVENT_EXPOSED:
                SDL_FilterEvents(_pg_remove_pending_VIDEOEXPOSE, &newevent);

                newevent.type = SDL_VIDEOEXPOSE;
                SDL_PushEvent(&newevent);
                break;
            case SDL_WINDOWEVENT_ENTER:
            case SDL_WINDOWEVENT_LEAVE:
            case SDL_WINDOWEVENT_FOCUS_GAINED:
            case SDL_WINDOWEVENT_FOCUS_LOST:
            case SDL_WINDOWEVENT_MINIMIZED:
            case SDL_WINDOWEVENT_RESTORED:
                newevent.type = SDL_ACTIVEEVENT;
                SDL_PushEvent(&newevent);
        }
    }

    else if (event->type == SDL_KEYDOWN) {
        if (event->key.repeat)
            return 0;

        if (pg_key_repeat_delay > 0) {
            if (_pg_repeat_timer)
                SDL_RemoveTimer(_pg_repeat_timer);

            memcpy(&_pg_repeat_event, event, sizeof(SDL_Event));
            _pg_repeat_timer = SDL_AddTimer(pg_key_repeat_delay,
                                            _pg_repeat_callback,
                                            NULL);
        }

        /* store the keydown event for later in the SDL_TEXTINPUT */
        if (!_pg_last_keydown_event)
            _pg_last_keydown_event = PyMem_New(SDL_Event, 1);
        memcpy(_pg_last_keydown_event, event, sizeof(SDL_Event));
    }

    else if (event->type == SDL_TEXTINPUT) {
        if (_pg_last_keydown_event) {
            _pg_put_event_unicode(_pg_last_keydown_event, event->text.text);
            PyMem_Del(_pg_last_keydown_event);
            _pg_last_keydown_event = NULL;
        }
    }

    else if (event->type == PGE_KEYREPEAT) {
        event->type = SDL_KEYDOWN;
    }

    else if (event->type == SDL_KEYUP) {
        if (_pg_repeat_timer &&
            _pg_repeat_event.key.keysym.scancode == event->key.keysym.scancode) {
            SDL_RemoveTimer(_pg_repeat_timer);
            _pg_repeat_timer = 0;
        }
    }

    else if (event->type == SDL_MOUSEBUTTONDOWN ||
        event->type == SDL_MOUSEBUTTONUP) {
        if (event->button.button & PGM_BUTTON_KEEP)
            event->button.button ^= PGM_BUTTON_KEEP;
        else if (event->button.button >= PGM_BUTTON_WHEELUP)
            event->button.button += (PGM_BUTTON_X1 - PGM_BUTTON_WHEELUP);
    }

    else if (event->type == SDL_MOUSEWHEEL) {
        //#691 We are not moving wheel!
        if (!event->wheel.y)
            return 0;

        SDL_GetMouseState(&x, &y);
        /* Generate a MouseButtonDown event and MouseButtonUp for
         * compatibility. https://wiki.libsdl.org/SDL_MouseWheelEvent
         */
        newdownevent.type = SDL_MOUSEBUTTONDOWN;
        newdownevent.button.x = x;
        newdownevent.button.y = y;

        newupevent.type = SDL_MOUSEBUTTONUP;
        newupevent.button.x = x;
        newupevent.button.y = y;

        newdownevent.button.state = SDL_PRESSED;
        newdownevent.button.clicks = 1;

        newupevent.button.state = SDL_RELEASED;
        newupevent.button.clicks = 1;

        if (event->wheel.y > 0) {
            newdownevent.button.button =  PGM_BUTTON_WHEELUP | PGM_BUTTON_KEEP;
            newupevent.button.button = PGM_BUTTON_WHEELUP | PGM_BUTTON_KEEP;
        }
        else {
            newdownevent.button.button =  PGM_BUTTON_WHEELDOWN | PGM_BUTTON_KEEP;
            newupevent.button.button = PGM_BUTTON_WHEELDOWN | PGM_BUTTON_KEEP;
        }

        /* Use a for loop to simulate multiple events, because SDL 1
         * works that way */
        for (i = 0; i < abs(event->wheel.y); i++) {
            SDL_PushEvent(&newdownevent);
            SDL_PushEvent(&newupevent);
        }
        /* this doesn't work! This is called by SDL, not Python:
          if (SDL_PushEvent(&newdownevent) < 0)
            return RAISE(pgExc_SDLError, SDL_GetError()), 0;
        */
    }
    return SDL_EventState(_pg_pgevent_proxify(event->type), SDL_QUERY);
}

static int
pg_EnableKeyRepeat(int delay, int interval)
{
    if (delay < 0 || interval < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "delay and interval must equal at least 0");
        return -1;
    }
    pg_key_repeat_delay = delay;
    pg_key_repeat_interval = interval;
    return 0;
}

static void
pg_GetKeyRepeat(int *delay, int *interval)
{
    *delay = pg_key_repeat_delay;
    *interval = pg_key_repeat_interval;
}
#endif /* IS_SDLv2 */

static void
_pg_event_cleanup(void)
{
#if IS_SDLv2
    if (_pg_repeat_timer) {
        SDL_RemoveTimer(_pg_repeat_timer);
        _pg_repeat_timer = 0;
    }
#endif /* IS_SLDv2 */
    /* The main reason for _custom_event to be reset here is so we can have a
     * unit test that checks if pygame.event.custom_type() stops returning new
     * types when they are finished, without that test preventing further
     * tests from getting a custom event type.*/
    _custom_event = _PGE_CUSTOM_EVENT_INIT;
    _pg_event_is_init = 0;
}

static PyObject *
pgEvent_AutoInit(PyObject *self, PyObject *args)
{
    if (!_pg_event_is_init) {
#if IS_SDLv2
        pg_key_repeat_delay = 0;
        pg_key_repeat_interval = 0;
#endif /* IS_SLDv2 */

        pg_RegisterQuit(_pg_event_cleanup);
        _pg_event_is_init = 1;
    }

    return PyInt_FromLong(_pg_event_is_init);
}

/* This function can fill an SDL event from pygame event */
static int
pgEvent_FillUserEvent(pgEventObject *e, SDL_Event *event)
{
    Py_INCREF(e->dict);

    memset(event, 0, sizeof(SDL_Event));
    event->type = _pg_pgevent_proxify(e->type);
    event->user.code = USEROBJ_CHECK;
    event->user.data1 = (void *)e->dict;
    event->user.data2 = NULL;

    return 0;
}

static char *
_pg_name_from_eventtype(int type)
{
    switch (type) {
        case SDL_ACTIVEEVENT:
            return "ActiveEvent";
        case SDL_KEYDOWN:
            return "KeyDown";
        case SDL_KEYUP:
            return "KeyUp";
        case SDL_MOUSEMOTION:
            return "MouseMotion";
        case SDL_MOUSEBUTTONDOWN:
            return "MouseButtonDown";
        case SDL_MOUSEBUTTONUP:
            return "MouseButtonUp";
        case SDL_JOYAXISMOTION:
            return "JoyAxisMotion";
        case SDL_JOYBALLMOTION:
            return "JoyBallMotion";
        case SDL_JOYHATMOTION:
            return "JoyHatMotion";
        case SDL_JOYBUTTONUP:
            return "JoyButtonUp";
        case SDL_JOYBUTTONDOWN:
            return "JoyButtonDown";
        case SDL_QUIT:
            return "Quit";
        case SDL_SYSWMEVENT:
            return "SysWMEvent";
        case SDL_VIDEORESIZE:
            return "VideoResize";
        case SDL_VIDEOEXPOSE:
            return "VideoExpose";
        case PGE_MIDIIN:
            return "MidiIn";
        case PGE_MIDIOUT:
            return "MidiOut";
        case SDL_NOEVENT:
            return "NoEvent";
#if IS_SDLv2
        case SDL_FINGERMOTION:
            return "FingerMotion";
        case SDL_FINGERDOWN:
            return "FingerDown";
        case SDL_FINGERUP:
            return "FingerUp";
        case SDL_MULTIGESTURE:
            return "MultiGesture";
        case SDL_MOUSEWHEEL:
            return "MouseWheel";
        case SDL_TEXTINPUT:
            return "TextInput";
        case SDL_TEXTEDITING:
            return "TextEditing";
        case SDL_DROPFILE:
            return "DropFile";
#if SDL_VERSION_ATLEAST(2, 0, 5)
        case SDL_DROPTEXT:
            return "DropText";
        case SDL_DROPBEGIN:
            return "DropBegin";
        case SDL_DROPCOMPLETE:
            return "DropComplete";
#endif /* SDL_VERSION_ATLEAST(2, 0, 5) */
        case SDL_CONTROLLERAXISMOTION:
            return "ControllerAxisMotion";
        case SDL_CONTROLLERBUTTONDOWN:
            return "ControllerButtonDown";
        case SDL_CONTROLLERBUTTONUP:
            return "ControllerButtonUp";
        case SDL_CONTROLLERDEVICEADDED:
            return "ControllerDeviceAdded";
        case SDL_CONTROLLERDEVICEREMOVED:
            return "ControllerDeviceRemoved";
        case SDL_CONTROLLERDEVICEREMAPPED:
            return "ControllerDeviceMapped";
        case SDL_JOYDEVICEADDED:
            return "JoyDeviceAdded";
        case SDL_JOYDEVICEREMOVED:
            return "JoyDeviceRemoved";

#ifdef SDL2_AUDIODEVICE_SUPPORTED
        case SDL_AUDIODEVICEADDED:
            return "AudioDeviceAdded";
        case SDL_AUDIODEVICEREMOVED:
            return "AudioDeviceRemoved";
#endif /* SDL2_AUDIODEVICE_SUPPORTED */

        case PGE_WINDOWSHOWN:
            return "WindowShown";
        case PGE_WINDOWHIDDEN:
            return "WindowHidden";
        case PGE_WINDOWEXPOSED:
            return "WindowExposed";
        case PGE_WINDOWMOVED:
            return "WindowMoved";
        case PGE_WINDOWRESIZED:
            return "WindowResized";
        case PGE_WINDOWSIZECHANGED:
            return "WindowSizeChanged";
        case PGE_WINDOWMINIMIZED:
            return "WindowMinimized";
        case PGE_WINDOWMAXIMIZED:
            return "WindowMaximized";
        case PGE_WINDOWRESTORED:
            return "WindowRestored";
        case PGE_WINDOWENTER:
            return "WindowEnter";
        case PGE_WINDOWLEAVE:
            return "WindowLeave";
        case PGE_WINDOWFOCUSGAINED:
            return "WindowFocusGained";
        case PGE_WINDOWFOCUSLOST:
            return "WindowFocusLost";
        case PGE_WINDOWCLOSE:
            return "WindowClose";
        case PGE_WINDOWTAKEFOCUS:
            return "WindowTakeFocus";
        case PGE_WINDOWHITTEST:
            return "WindowHitTest";
#endif /* IS_SDLv2 */

    }
    if (type >= PGE_USEREVENT && type < PG_NUMEVENTS)
        return "UserEvent";
    return "Unknown";
}

/* Helper for adding objects to dictionaries. Check for errors with
   PyErr_Occurred() */
static void
_pg_insobj(PyObject *dict, char *name, PyObject *v)
{
    if (v) {
        PyDict_SetItemString(dict, name, v);
        Py_DECREF(v);
    }
}

#if IS_SDLv2
static PyObject *
get_joy_guid(int device_index) {
    char strguid[33];
    SDL_JoystickGUID guid = SDL_JoystickGetDeviceGUID(device_index);

    SDL_JoystickGetGUIDString(guid, strguid, 33);
    return Text_FromUTF8(strguid);
}
#endif

/** Try to insert the instance ID for a new device into the joystick mapping. */
void
_joy_map_add(int device_index) {
#if SDL_VERSION_ATLEAST(2, 0, 6)
    int instance_id = (int) SDL_JoystickGetDeviceInstanceID(device_index);
    PyObject *k, *v;
    if (instance_id != -1) {
        k = PyInt_FromLong(instance_id);
        v = PyInt_FromLong(device_index);
        if (k != NULL && v != NULL) {
            PyDict_SetItem(joy_instance_map, k, v);
        }
        Py_XDECREF(k);
        Py_XDECREF(v);
    }
#endif
}

/** Look up a device ID for an instance ID. */
PyObject *
_joy_map_instance(int instance_id) {
    PyObject *v, *k = PyInt_FromLong(instance_id);
    if (!k) {
        Py_RETURN_NONE;
    }
    v = PyDict_GetItem(joy_instance_map, k);
    if (v) {
        Py_DECREF(k);
        Py_INCREF(v);
        return v;
    }
    return k;
}

/** Discard a joystick from the joystick instance -> device mapping. */
void
_joy_map_discard(int instance_id) {
    PyObject *k = PyInt_FromLong(instance_id);

    if (k) {
        PyDict_DelItem(joy_instance_map, k);
        Py_DECREF(k);
    }
}

static PyObject *
dict_from_event(SDL_Event *event)
{
    PyObject *dict = NULL, *tuple, *obj;
    int hx, hy;
#if IS_SDLv2
    long gain;
    long state;
#endif /* IS_SDLv2 */

    /* check if a proxy event or userevent was posted */
    if (event->type >= PGPOST_EVENTBEGIN && event->user.code == USEROBJ_CHECK)
        return (PyObject *)event->user.data1;

    dict = PyDict_New();
    if (!dict)
        return NULL;

    switch (event->type) {
#if IS_SDLv1
        case SDL_VIDEORESIZE:
            obj = Py_BuildValue("(ii)", event->resize.w, event->resize.h);
            _pg_insobj(dict, "size", obj);
            _pg_insobj(dict, "w", PyInt_FromLong(event->resize.w));
            _pg_insobj(dict, "h", PyInt_FromLong(event->resize.h));
            break;
        case SDL_ACTIVEEVENT:
            _pg_insobj(dict, "gain", PyInt_FromLong(event->active.gain));
            _pg_insobj(dict, "state", PyInt_FromLong(event->active.state));
            break;
#else /* IS_SDLv2 */
        case SDL_VIDEORESIZE:
            obj = Py_BuildValue("(ii)", event->window.data1,
                                event->window.data2);
            _pg_insobj(dict, "size", obj);
            _pg_insobj(dict, "w", PyInt_FromLong(event->window.data1));
            _pg_insobj(dict, "h", PyInt_FromLong(event->window.data2));
            break;
        case SDL_ACTIVEEVENT:
            switch (event->window.event) {
                case SDL_WINDOWEVENT_ENTER:
                    gain = 1;
                    state = (long)SDL_APPFOCUSMOUSE;
                    break;
                case SDL_WINDOWEVENT_LEAVE:
                    gain = 0;
                    state = (long)SDL_APPFOCUSMOUSE;
                    break;
                case SDL_WINDOWEVENT_FOCUS_GAINED:
                    gain = 1;
                    state = (long)SDL_APPINPUTFOCUS;
                    break;
                case SDL_WINDOWEVENT_FOCUS_LOST:
                    gain = 0;
                    state = (long)SDL_APPINPUTFOCUS;
                    break;
                case SDL_WINDOWEVENT_MINIMIZED:
                    gain = 0;
                    state = (long)SDL_APPACTIVE;
                    break;
                default:
                    assert(event->window.event == SDL_WINDOWEVENT_RESTORED);
                    gain = 1;
                    state = (long)SDL_APPACTIVE;
            }
            _pg_insobj(dict, "gain", PyInt_FromLong(gain));
            _pg_insobj(dict, "state", PyInt_FromLong(state));
            break;
#endif /* IS_SDLv2 */
        case SDL_KEYDOWN:
#if IS_SDLv1
            _pg_insobj(dict, "unicode", _pg_chr(event->key.keysym.unicode));
        case SDL_KEYUP:
#else /* IS_SDLv2 */
        case SDL_KEYUP:
            _pg_insobj(dict, "unicode", _pg_get_event_unicode(event));
#endif /* IS_SDLv2 */
            _pg_insobj(dict, "key", PyInt_FromLong(event->key.keysym.sym));
            _pg_insobj(dict, "mod", PyInt_FromLong(event->key.keysym.mod));
            _pg_insobj(dict, "scancode", PyInt_FromLong(event->key.keysym.scancode));
            break;
        case SDL_MOUSEMOTION:
            obj = Py_BuildValue("(ii)", event->motion.x, event->motion.y);
            _pg_insobj(dict, "pos", obj);
            obj =
                Py_BuildValue("(ii)", event->motion.xrel, event->motion.yrel);
            _pg_insobj(dict, "rel", obj);
            if ((tuple = PyTuple_New(3))) {
                PyTuple_SET_ITEM(tuple, 0,
                                 PyInt_FromLong((event->motion.state &
                                                 SDL_BUTTON(1)) != 0));
                PyTuple_SET_ITEM(tuple, 1,
                                 PyInt_FromLong((event->motion.state &
                                                 SDL_BUTTON(2)) != 0));
                PyTuple_SET_ITEM(tuple, 2,
                                 PyInt_FromLong((event->motion.state &
                                                 SDL_BUTTON(3)) != 0));
                _pg_insobj(dict, "buttons", tuple);
            }
            break;
        case SDL_MOUSEBUTTONDOWN:
        case SDL_MOUSEBUTTONUP:
            obj = Py_BuildValue("(ii)", event->button.x, event->button.y);
            _pg_insobj(dict, "pos", obj);
            _pg_insobj(dict, "button", PyInt_FromLong(event->button.button));
            break;
        case SDL_JOYAXISMOTION:
            _pg_insobj(dict, "joy", _joy_map_instance(event->jaxis.which));
            _pg_insobj(dict, "instance_id", PyInt_FromLong(event->jaxis.which));
            _pg_insobj(dict, "axis", PyInt_FromLong(event->jaxis.axis));
            _pg_insobj(dict, "value",
                   PyFloat_FromDouble(event->jaxis.value / 32767.0));
            break;
        case SDL_JOYBALLMOTION:
            _pg_insobj(dict, "joy", _joy_map_instance(event->jaxis.which));
            _pg_insobj(dict, "instance_id", PyInt_FromLong(event->jball.which));
            _pg_insobj(dict, "ball", PyInt_FromLong(event->jball.ball));
            obj = Py_BuildValue("(ii)", event->jball.xrel, event->jball.yrel);
            _pg_insobj(dict, "rel", obj);
            break;
        case SDL_JOYHATMOTION:
            _pg_insobj(dict, "joy", _joy_map_instance(event->jaxis.which));
            _pg_insobj(dict, "instance_id", PyInt_FromLong(event->jhat.which));
            _pg_insobj(dict, "hat", PyInt_FromLong(event->jhat.hat));
            hx = hy = 0;
            if (event->jhat.value & SDL_HAT_UP)
                hy = 1;
            else if (event->jhat.value & SDL_HAT_DOWN)
                hy = -1;
            if (event->jhat.value & SDL_HAT_RIGHT)
                hx = 1;
            else if (event->jhat.value & SDL_HAT_LEFT)
                hx = -1;
            _pg_insobj(dict, "value", Py_BuildValue("(ii)", hx, hy));
            break;
        case SDL_JOYBUTTONUP:
        case SDL_JOYBUTTONDOWN:
            _pg_insobj(dict, "joy", _joy_map_instance(event->jaxis.which));
            _pg_insobj(dict, "instance_id", PyInt_FromLong(event->jbutton.which));
            _pg_insobj(dict, "button", PyInt_FromLong(event->jbutton.button));
            break;
#if IS_SDLv2
        case PGE_WINDOWMOVED:
        case PGE_WINDOWRESIZED:
        case PGE_WINDOWSIZECHANGED:
            /*other PGE_WINDOW* events do not have attributes */
            _pg_insobj(dict, "x", PyInt_FromLong(event->window.data1));
            _pg_insobj(dict, "y", PyInt_FromLong(event->window.data2));
            break;
#ifdef SDL2_AUDIODEVICE_SUPPORTED
        case SDL_AUDIODEVICEADDED:
        case SDL_AUDIODEVICEREMOVED:
            _pg_insobj(dict, "which", PyInt_FromLong(event->adevice.which));
            _pg_insobj(dict, "iscapture", PyInt_FromLong(event->adevice.iscapture));
            break;
#endif /* SDL2_AUDIODEVICE_SUPPORTED */
        case SDL_FINGERMOTION:
        case SDL_FINGERDOWN:
        case SDL_FINGERUP:
            /* https://wiki.libsdl.org/SDL_TouchFingerEvent */
            _pg_insobj(dict, "touch_id", PyLong_FromLongLong(event->tfinger.touchId));
            _pg_insobj(dict, "finger_id", PyLong_FromLongLong(event->tfinger.fingerId));
            _pg_insobj(dict, "x", PyFloat_FromDouble(event->tfinger.x));
            _pg_insobj(dict, "y", PyFloat_FromDouble(event->tfinger.y));
            _pg_insobj(dict, "dx", PyFloat_FromDouble(event->tfinger.dx));
            _pg_insobj(dict, "dy", PyFloat_FromDouble(event->tfinger.dy));
            _pg_insobj(dict, "pressure", PyFloat_FromDouble(event->tfinger.dy));
            break;
        case SDL_MULTIGESTURE:
            /* https://wiki.libsdl.org/SDL_MultiGestureEvent */
            _pg_insobj(dict, "touch_id", PyLong_FromLongLong(event->mgesture.touchId));
            _pg_insobj(dict, "x", PyFloat_FromDouble(event->mgesture.x));
            _pg_insobj(dict, "y", PyFloat_FromDouble(event->mgesture.y));
            _pg_insobj(dict, "rotated", PyFloat_FromDouble(event->mgesture.dTheta));
            _pg_insobj(dict, "pinched", PyFloat_FromDouble(event->mgesture.dDist));
            _pg_insobj(dict, "num_fingers", PyInt_FromLong(event->mgesture.numFingers));
            break;
        case SDL_MOUSEWHEEL:
            /* https://wiki.libsdl.org/SDL_MouseWheelEvent */
#ifndef NO_SDL_MOUSEWHEEL_FLIPPED
            _pg_insobj(dict, "flipped", PyBool_FromLong(event->wheel.direction == SDL_MOUSEWHEEL_FLIPPED));
#else
            _pg_insobj(dict, "flipped", PyBool_FromLong(0));
#endif
            _pg_insobj(dict, "y", PyInt_FromLong(event->wheel.y));
            _pg_insobj(dict, "x", PyInt_FromLong(event->wheel.x));
            _pg_insobj(dict, "which", PyInt_FromLong(event->wheel.which));
            break;
        case SDL_TEXTINPUT:
            /* https://wiki.libsdl.org/SDL_TextInputEvent */
            _pg_insobj(dict, "text", Text_FromUTF8(event->text.text));
            break;
        case SDL_TEXTEDITING:
            /* https://wiki.libsdl.org/SDL_TextEditingEvent */
            _pg_insobj(dict, "text", Text_FromUTF8(event->edit.text));
            _pg_insobj(dict, "start", PyLong_FromLong(event->edit.start));
            _pg_insobj(dict, "length", PyLong_FromLong(event->edit.length));
            break;
        /*  https://wiki.libsdl.org/SDL_DropEvent */
        case SDL_DROPFILE:
            _pg_insobj(dict, "file", Text_FromUTF8(event->drop.file));
            SDL_free(event->drop.file);
            break;

#if SDL_VERSION_ATLEAST(2, 0, 5)
        case SDL_DROPTEXT:
            _pg_insobj(dict, "text", Text_FromUTF8(event->drop.file));
            SDL_free(event->drop.file);
            break;
        case SDL_DROPBEGIN:
        case SDL_DROPCOMPLETE:
            break;
#endif /* SDL_VERSION_ATLEAST(2, 0, 5) */

        case SDL_CONTROLLERAXISMOTION:
            /* https://wiki.libsdl.org/SDL_ControllerAxisEvent */
            _pg_insobj(dict, "instance_id", PyLong_FromLong(event->caxis.which));
            _pg_insobj(dict, "axis", PyLong_FromLong(event->caxis.axis));
            _pg_insobj(dict, "value", PyLong_FromLong(event->caxis.value));
            break;
        case SDL_CONTROLLERBUTTONDOWN:
        case SDL_CONTROLLERBUTTONUP:
            /* https://wiki.libsdl.org/SDL_ControllerButtonEvent */
            _pg_insobj(dict, "instance_id", PyLong_FromLong(event->cbutton.which));
            _pg_insobj(dict, "button", PyLong_FromLong(event->cbutton.button));
            break;
        case SDL_CONTROLLERDEVICEADDED:
            _pg_insobj(dict, "device_index", PyLong_FromLong(event->cdevice.which));
            _pg_insobj(dict, "guid", get_joy_guid(event->jdevice.which));
            break;
        case SDL_JOYDEVICEADDED:
            _joy_map_add(event->jdevice.which);
            _pg_insobj(dict, "device_index", PyLong_FromLong(event->jdevice.which));
            _pg_insobj(dict, "guid", get_joy_guid(event->jdevice.which));
            break;
        case SDL_CONTROLLERDEVICEREMOVED:
        case SDL_CONTROLLERDEVICEREMAPPED:
            /* https://wiki.libsdl.org/SDL_ControllerDeviceEvent */
            _pg_insobj(dict, "instance_id", PyLong_FromLong(event->cdevice.which));
            break;
        case SDL_JOYDEVICEREMOVED:
            _joy_map_discard(event->jdevice.which);
            _pg_insobj(dict, "instance_id", PyLong_FromLong(event->jdevice.which));
            break;
#endif

#ifdef WIN32
#if IS_SDLv1
        case SDL_SYSWMEVENT:
            _pg_insobj(dict, "hwnd",
                   PyInt_FromLong((long)(event->syswm.msg->hwnd)));
            _pg_insobj(dict, "msg", PyInt_FromLong(event->syswm.msg->msg));
            _pg_insobj(dict, "wparam", PyInt_FromLong(event->syswm.msg->wParam));
            _pg_insobj(dict, "lparam", PyInt_FromLong(event->syswm.msg->lParam));
            break;
#else /* IS_SDLv2 */
        case SDL_SYSWMEVENT:
            _pg_insobj(dict, "hwnd",
                   PyInt_FromLong((long)(event->syswm.msg->msg.win.hwnd)));
            _pg_insobj(dict, "msg", PyInt_FromLong(event->syswm.msg->msg.win.msg));
            _pg_insobj(dict, "wparam", PyInt_FromLong(event->syswm.msg->msg.win.wParam));
            _pg_insobj(dict, "lparam", PyInt_FromLong(event->syswm.msg->msg.win.lParam));
            break;
#endif /* IS_SDLv2 */
#endif /* WIN32 */

#if (defined(unix) || defined(__unix__) || defined(_AIX) ||     \
     defined(__OpenBSD__)) &&                                   \
    (defined(SDL_VIDEO_DRIVER_X11) && !defined(__CYGWIN32__) && \
     !defined(ENABLE_NANOX) && !defined(__QNXNTO__))
#if IS_SDLv1
        case SDL_SYSWMEVENT:
            _pg_insobj(dict, "event",
                   Bytes_FromStringAndSize(
                       (char *)&(event->syswm.msg->event.xevent),
                       sizeof(XEvent)));
            break;
#else  /* IS_SDLv2 */
        case SDL_SYSWMEVENT:
            if (event->syswm.msg->subsystem == SDL_SYSWM_X11) {
                XEvent *xevent = (XEvent *)&event->syswm.msg->msg.x11.event;
                obj = Bytes_FromStringAndSize((char *)xevent, sizeof(XEvent));
                _pg_insobj(dict, "event", obj);
            }
            break;
#endif /* IS_SDLv2 */
#endif /* (defined(unix) || ... */
    } /* switch (event->type) */
    /* Events that dont have any attributes are not handled in switch
     * statement */

    switch (event->type) {
#if IS_SDLv2
        case PGE_WINDOWSHOWN:
        case PGE_WINDOWHIDDEN:
        case PGE_WINDOWEXPOSED:
        case PGE_WINDOWMOVED:
        case PGE_WINDOWRESIZED:
        case PGE_WINDOWSIZECHANGED:
        case PGE_WINDOWMINIMIZED:
        case PGE_WINDOWMAXIMIZED:
        case PGE_WINDOWRESTORED:
        case PGE_WINDOWENTER:
        case PGE_WINDOWLEAVE:
        case PGE_WINDOWFOCUSGAINED:
        case PGE_WINDOWFOCUSLOST:
        case PGE_WINDOWCLOSE:
        case PGE_WINDOWTAKEFOCUS:
        case PGE_WINDOWHITTEST:
        case SDL_TEXTEDITING:
        case SDL_TEXTINPUT:
        case SDL_MOUSEWHEEL:
#endif /* IS_SDLv2 */
        case SDL_KEYDOWN:
        case SDL_KEYUP:
        case SDL_MOUSEMOTION:
        case SDL_MOUSEBUTTONDOWN:
        case SDL_MOUSEBUTTONUP:
        {
#if IS_SDLv2
            SDL_Window *window = SDL_GetWindowFromID(event->window.windowID);
            PyObject *pgWindow;
            if (!window || !(pgWindow=SDL_GetWindowData(window, "pg_window"))) {
                pgWindow = Py_None;
            }
            Py_INCREF(pgWindow);
            _pg_insobj(dict, "window", pgWindow);
#else /* IS_SDLv1 */
            Py_INCREF(Py_None);
            _pg_insobj(dict, "window", Py_None);
#endif /* IS_SDLv1 */
            break;
        }
    }
    return dict;
}

/* event object internals */

static void
pg_event_dealloc(PyObject *self)
{
    pgEventObject *e = (pgEventObject *)self;
    Py_XDECREF(e->dict);
    PyObject_Del(self);
}

#ifdef PYPY_VERSION
/* Because pypy does not work with the __dict__ tp_dictoffset. */
PyObject *
pg_EventGetAttr(PyObject *o, PyObject *attr_name)
{
    /* Try e->dict first, if not try the generic attribute. */
    PyObject *result = PyDict_GetItem(((pgEventObject *)o)->dict, attr_name);
    if (!result) {
        return PyObject_GenericGetAttr(o, attr_name);
    }
    return result;
}

int
pg_EventSetAttr(PyObject *o, PyObject *name, PyObject *value)
{
    /* if the variable is in the dict, deal with it there.
       else if it's a normal attribute set it there.
       else if it's not an attribute, or in the dict, set it in the dict.
    */
    int dictResult;
    int setInDict = 0;
    PyObject *result = PyDict_GetItem(((pgEventObject *)o)->dict, name);

    if (result) {
        setInDict = 1;
    }
    else {
        result = PyObject_GenericGetAttr(o, name);
        if (!result) {
            setInDict = 1;
        }
    }

    if (setInDict) {
        dictResult = PyDict_SetItem(((pgEventObject *)o)->dict, name, value);
        if (dictResult) {
            return -1;
        }
        return 0;
    }
    else {
        return PyObject_GenericSetAttr(o, name, value);
    }
}
#endif

PyObject *
pg_event_str(PyObject *self)
{
    pgEventObject *e = (pgEventObject *)self;
    char *str;
    PyObject *strobj;
    PyObject *pyobj;
    char *s;
    size_t size;
#if PY3
    PyObject *encodedobj;
#endif

    strobj = PyObject_Str(e->dict);
    if (strobj == NULL) {
        return NULL;
    }
#if PY3
    encodedobj = PyUnicode_AsUTF8String(strobj);
    Py_DECREF(strobj);
    strobj = encodedobj;
    encodedobj = NULL;
    if (strobj == NULL) {
        return NULL;
    }
    s = PyBytes_AsString(strobj);
#else
    s = PyString_AsString(strobj);
#endif
    size = (11 + strlen(_pg_name_from_eventtype(e->type)) + strlen(s) +
            sizeof(e->type) * 3 + 1);

    str = (char *)PyMem_Malloc(size);
    if (!str) {
        Py_DECREF(strobj);
        return PyErr_NoMemory();
    }
    sprintf(str, "<Event(%d-%s %s)>", e->type,
        _pg_name_from_eventtype(e->type), s);

    Py_DECREF(strobj);

    pyobj = Text_FromUTF8(str);
    PyMem_Free(str);

    return (pyobj);
}

static int
_pg_event_nonzero(pgEventObject *self)
{
    return self->type != SDL_NOEVENT;
}

static PyNumberMethods pg_event_as_number = {
    (binaryfunc)NULL, /*Add*/
    (binaryfunc)NULL, /*subtract*/
    (binaryfunc)NULL, /*multiply*/
#if !PY3
    (binaryfunc)NULL, /*divide*/
#endif
    (binaryfunc)NULL,       /*remainder*/
    (binaryfunc)NULL,       /*divmod*/
    (ternaryfunc)NULL,      /*power*/
    (unaryfunc)NULL,        /*negative*/
    (unaryfunc)NULL,        /*pos*/
    (unaryfunc)NULL,        /*abs*/
    (inquiry)_pg_event_nonzero, /*nonzero*/
    (unaryfunc)NULL,        /*invert*/
    (binaryfunc)NULL,       /*lshift*/
    (binaryfunc)NULL,       /*rshift*/
    (binaryfunc)NULL,       /*and*/
    (binaryfunc)NULL,       /*xor*/
    (binaryfunc)NULL,       /*or*/
#if !PY3
    (coercion)NULL, /*coerce*/
#endif
    (unaryfunc)NULL, /*int*/
#if !PY3
    (unaryfunc)NULL, /*long*/
#endif
    (unaryfunc)NULL, /*float*/
};


static PyTypeObject pgEvent_Type;
#define pgEvent_Check(x) ((x)->ob_type == &pgEvent_Type)
#define OFF(x) offsetof(pgEventObject, x)

static PyMemberDef pg_event_members[] = {
    {"__dict__", T_OBJECT, OFF(dict), READONLY},
    {"type", T_INT, OFF(type), READONLY},
    {"dict", T_OBJECT, OFF(dict), READONLY},
    {NULL} /* Sentinel */
};

/*
 * eventA == eventB
 * eventA != eventB
 */
static PyObject *
pg_event_richcompare(PyObject *o1, PyObject *o2, int opid)
{
    pgEventObject *e1, *e2;

    if (!pgEvent_Check(o1) || !pgEvent_Check(o2)) {
        goto Unimplemented;
    }

    e1 = (pgEventObject *)o1;
    e2 = (pgEventObject *)o2;
    switch (opid) {
        case Py_EQ:
            return PyBool_FromLong(
                e1->type == e2->type &&
                PyObject_RichCompareBool(e1->dict, e2->dict, Py_EQ) == 1);
        case Py_NE:
            return PyBool_FromLong(
                e1->type != e2->type ||
                PyObject_RichCompareBool(e1->dict, e2->dict, Py_NE) == 1);
        default:
            break;
    }

Unimplemented:
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

static PyTypeObject pgEvent_Type = {
    PyVarObject_HEAD_INIT(NULL,0)
    "Event",                    /*name*/
    sizeof(pgEventObject),      /*basic size*/
    0,                          /*itemsize*/
    pg_event_dealloc,              /*dealloc*/
    0,                          /*print*/
    0,                          /*getattr*/
    0,                          /*setattr*/
    0,                          /*compare*/
    pg_event_str,                  /*repr*/
    &pg_event_as_number,           /*as_number*/
    0,                          /*as_sequence*/
    0,                          /*as_mapping*/
    (hashfunc)NULL,             /*hash*/
    (ternaryfunc)NULL,          /*call*/
    (reprfunc)NULL,             /*str*/
#ifdef PYPY_VERSION
    pg_EventGetAttr, /* tp_getattro */
    pg_EventSetAttr, /* tp_setattro */
#else
    PyObject_GenericGetAttr, /* tp_getattro */
    PyObject_GenericSetAttr, /* tp_setattro */
#endif
    0, /* tp_as_buffer */
#if PY3
    0,
#else
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_RICHCOMPARE,
#endif
    DOC_PYGAMEEVENTEVENT,          /* Documentation string */
    0,                             /* tp_traverse */
    0,                             /* tp_clear */
    pg_event_richcompare,             /* tp_richcompare */
    0,                             /* tp_weaklistoffset */
    0,                             /* tp_iter */
    0,                             /* tp_iternext */
    0,                             /* tp_methods */
    pg_event_members,                 /* tp_members */
    0,                             /* tp_getset */
    0,                             /* tp_base */
    0,                             /* tp_dict */
    0,                             /* tp_descr_get */
    0,                             /* tp_descr_set */
    offsetof(pgEventObject, dict), /* tp_dictoffset */
    0,                             /* tp_init */
    0,                             /* tp_alloc */
    0,                             /* tp_new */
};

static PyObject *
pgEvent_New(SDL_Event *event)
{
    pgEventObject *e;
    e = PyObject_New(pgEventObject, &pgEvent_Type);
    if (!e)
        return PyErr_NoMemory();

    if (event) {
        e->type = _pg_pgevent_deproxify(event->type);
        e->dict = dict_from_event(event);
    }
    else {
        e->type = SDL_NOEVENT;
        e->dict = PyDict_New();
    }
    if (!e->dict) {
        PyObject_Del(e);
        return PyErr_NoMemory();
    }
    return (PyObject *)e;
}

static PyObject *
pgEvent_New2(int type, PyObject *dict)
{
    pgEventObject *e;
    e = PyObject_New(pgEventObject, &pgEvent_Type);
    if (!e)
        return PyErr_NoMemory();

    e->type = _pg_pgevent_deproxify(type);
    if (!dict) {
        dict = PyDict_New();
        if (!dict) {
            PyObject_Del(e);
            return PyErr_NoMemory();
        }
    }
    else {
        if (PyDict_GetItemString(dict, "type")) {
            PyObject_Del(e);
            return RAISE(PyExc_ValueError,
                "redundant type field in event dict");
        }
        Py_INCREF(dict);
    }
    e->dict = dict;
    return (PyObject *)e;
}

/* event module functions */
static PyObject *
pg_Event(PyObject *self, PyObject *arg, PyObject *keywords)
{
    PyObject *dict = NULL;
    PyObject *event;
    int type;
    if (!PyArg_ParseTuple(arg, "i|O!", &type, &PyDict_Type, &dict))
        return NULL;

    if (!dict) {
        dict = PyDict_New();
        if (!dict)
            return PyErr_NoMemory();
    }
    else
        Py_INCREF(dict);

    if (keywords) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(keywords, &pos, &key, &value)) {
            if (PyDict_SetItem(dict, key, value) < 0) {
                Py_DECREF(dict);
                return NULL; /* Exception already set. */
            }
        }
    }

    event = pgEvent_New2(type, dict);

    Py_DECREF(dict);
    return event;
}

static PyObject *
event_name(PyObject *self, PyObject *arg)
{
    int type;
    if (!PyArg_ParseTuple(arg, "i", &type))
        return NULL;

    return Text_FromUTF8(_pg_name_from_eventtype(type));
}

static PyObject *
set_grab(PyObject *self, PyObject *arg)
{
    int doit;
#if IS_SDLv2
    SDL_Window *win = NULL;
#endif /* IS_SDLv2 */

#if PY2
    if (!PyArg_ParseTuple(arg, "i", &doit))
        return NULL;
#else
    if (!PyArg_ParseTuple(arg, "p", &doit))
        return NULL;
#endif
    VIDEO_INIT_CHECK();

#if IS_SDLv1
    if (doit)
        SDL_WM_GrabInput(SDL_GRAB_ON);
    else
        SDL_WM_GrabInput(SDL_GRAB_OFF);
#else  /* IS_SDLv2 */
    win = pg_GetDefaultWindow();
    if (win) {
        if (doit) {
            SDL_SetWindowGrab(win, SDL_TRUE);
            if (SDL_ShowCursor(SDL_QUERY) == SDL_DISABLE)
                SDL_SetRelativeMouseMode(1);
            else
                SDL_SetRelativeMouseMode(0);
        }
        else {
            SDL_SetWindowGrab(win, SDL_FALSE);
            SDL_SetRelativeMouseMode(0);
        }
    }
#endif /* IS_SDLv2 */

    Py_RETURN_NONE;
}

static PyObject *
get_grab(PyObject *self)
{
#if IS_SDLv1
    VIDEO_INIT_CHECK();
    return PyInt_FromLong(SDL_WM_GrabInput(SDL_GRAB_QUERY) == SDL_GRAB_ON);
#else  /* IS_SDLv2 */
    SDL_Window *win;
    SDL_bool mode = SDL_FALSE;

    VIDEO_INIT_CHECK();
    win = pg_GetDefaultWindow();
    if (win)
        mode = SDL_GetWindowGrab(win);
    return PyInt_FromLong(mode);
#endif /* IS_SDLv2 */
}

static void
_pg_event_pump(int dopump)
{
    if (dopump) {
        SDL_PumpEvents();
    }
#if IS_SDLv2
    /* We need to translate WINDOWEVENTS. But if we do that from the
     * from event filter, internal SDL stuff that rely on WINDOWEVENT
     * might break. So after every event pump, we translate events from
     * here */
    SDL_FilterEvents(_pg_translate_windowevent, NULL);
#endif
}

static int
_pg_event_wait(SDL_Event *event, int timeout)
{
    /* Custom re-implementation of SDL_WaitEventTimeout, doing this has
     * many advantages. This is copied from SDL source code, with a few
     * minor modifications */
    Uint32 finish = 0;

    if (timeout > 0)
        finish = SDL_GetTicks() + timeout;

    while (1) {
        _pg_event_pump(1); /* Use our custom pump here */
        switch (PG_PEEP_EVENT_ALL(event, 1, SDL_GETEVENT)) {
            case -1:
                return 0; /* Because this never happens, SDL does it too*/
            case 1:
                return 1;

            default:
                if (timeout >= 0 && SDL_GetTicks() >= finish) {
                    /* no events */
                    return 0;
                }
                SDL_Delay(1);
        }
    }
}

static PyObject *
pg_event_pump(PyObject *self)
{
    VIDEO_INIT_CHECK();
    _pg_event_pump(1);
    Py_RETURN_NONE;
}

static PyObject *
pg_event_poll(PyObject *self)
{
    SDL_Event event;
    VIDEO_INIT_CHECK();

    /* polling is just waiting for 0 timeout */
    if (!_pg_event_wait(&event, 0))
        return pgEvent_New(NULL);
    return pgEvent_New(&event);
}

static PyObject *
pg_event_wait(PyObject *self, PyObject *args, PyObject *kwargs)
{
    SDL_Event event;
    int status, timeout = 0;
    static char *kwids[] = {
        "timeout",
        NULL
    };

    VIDEO_INIT_CHECK();

    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", kwids, &timeout)) {
        return NULL;
    }

    if (!timeout)
        timeout = -1;

    Py_BEGIN_ALLOW_THREADS;
    status = _pg_event_wait(&event, timeout);
    Py_END_ALLOW_THREADS;

    if (!status)
        return pgEvent_New(NULL);
    return pgEvent_New(&event);
}

static int
_pg_eventtype_from_seq(PyObject *seq, int ind)
{
    int val;
    if (!pg_IntFromObjIndex(seq, ind, &val)) {
        PyErr_SetString(PyExc_TypeError,
            "type sequence must contain valid event types");
        return -1;
    }
    if (val < 0 || val >= PG_NUMEVENTS) {
        PyErr_SetString(PyExc_ValueError, "event type out of range");
        return -1;
    }
    return val;
}

static PyObject *
_pg_eventtype_as_seq(PyObject *obj, int *len)
{
    *len = 1;
    if (PySequence_Check(obj)) {
        *len = PySequence_Size(obj);
        /* The returned object gets decref'd later, so incref now */
        Py_INCREF(obj);
        return obj;
    }
    else if (PyInt_Check(obj))
        return Py_BuildValue("(O)", obj);
    else
        return RAISE(PyExc_TypeError,
                         "event type must be numeric or a sequence");
}

static void
_pg_flush_events(Uint32 type)
{
#if IS_SDLv1
    SDL_Event event;
    if (type == MAX_UINT32)
        while (PG_PEEP_EVENT_ALL(&event, 1, SDL_GETEVENT) == 1);
    else
        while (PG_PEEP_EVENT(&event, 1, SDL_GETEVENT, type) == 1);
#else /* IS_SDLv2 */
    if (type == MAX_UINT32)
        SDL_FlushEvents(SDL_FIRSTEVENT, SDL_LASTEVENT);
    else {
        SDL_FlushEvent(type);
        SDL_FlushEvent(_pg_pgevent_proxify(type));
    }
#endif /* IS_SDLv2 */
}

static PyObject *
pg_event_clear(PyObject *self, PyObject *args, PyObject *kwargs)
{
    int loop, len, type;
    PyObject *seq, *obj = NULL;
    int dopump = 1;

    static char *kwids[] = {
        "eventtype",
        "pump",
        NULL
    };

#if PY3
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Op", kwids,
                                     &obj, &dopump))
        return NULL;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Oi", kwids,
                                     &obj, &dopump))
        return NULL;
#endif

    VIDEO_INIT_CHECK();
    _pg_event_pump(dopump);

    if (obj == NULL || obj == Py_None) {
        _pg_flush_events(MAX_UINT32);
    }
    else {
        seq = _pg_eventtype_as_seq(obj, &len);
        if (!seq) /* error aldready set */
            return NULL;

        for (loop = 0; loop < len; loop++) {
            type = _pg_eventtype_from_seq(seq, loop);
            if (type == -1) {
                Py_DECREF(seq);
                return NULL; /* PyErr aldready set */
            }
            _pg_flush_events(type);
        }
        Py_DECREF(seq);
    }
    Py_RETURN_NONE;
}

static int
_pg_event_append_to_list(PyObject *list, SDL_Event *event)
{
    /* The caller of this function must handle decref of list on error */
    PyObject *e = pgEvent_New(event);
    if (!e) /* Exception already set. */
        return 0;

    if (PyList_Append(list, e)) {
        Py_DECREF(e);
        return 0; /* Exception already set. */
    }
    Py_DECREF(e);
    return 1;
}

static PyObject *
_pg_get_all_events(void)
{
    SDL_Event eventbuf[PG_GET_LIST_LEN];
    PyObject *list;
    int loop, len = PG_GET_LIST_LEN;

    list = PyList_New(0);
    if (!list)
        return PyErr_NoMemory();

    while (len == PG_GET_LIST_LEN) {
        len = PG_PEEP_EVENT_ALL(eventbuf, PG_GET_LIST_LEN, SDL_GETEVENT);
        if (len == -1) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            goto error;
        }

        for (loop = 0; loop < len; loop++) {
            if (!_pg_event_append_to_list(list, &eventbuf[loop]))
                goto error;
        }
    }
    return list;

error:
    Py_DECREF(list);
    return NULL;
}

static PyObject *
_pg_get_seq_events(PyObject *obj)
{
    SDL_Event event;
    int loop, type, len, ret;
    PyObject *seq, *list;

    list = PyList_New(0);
    if (!list)
        return PyErr_NoMemory();

    seq = _pg_eventtype_as_seq(obj, &len);
    if (!seq)
        goto error;

    for (loop = 0; loop < len; loop++) {
        type = _pg_eventtype_from_seq(seq, loop);
        if (type == -1)
            goto error;

        do {
            ret = PG_PEEP_EVENT(&event, 1, SDL_GETEVENT, type);
            if (ret < 0) {
                PyErr_SetString(pgExc_SDLError, SDL_GetError());
                goto error;
            }
            else if (ret > 0) {
                if (!_pg_event_append_to_list(list, &event))
                    goto error;
            }
        } while (ret);
#if IS_SDLv2
        do {
            ret = PG_PEEP_EVENT(&event, 1, SDL_GETEVENT,
                _pg_pgevent_proxify(type));
            if (ret < 0) {
                PyErr_SetString(pgExc_SDLError, SDL_GetError());
                goto error;
            }
            else if (ret > 0) {
                if (!_pg_event_append_to_list(list, &event))
                    goto error;
            }
        } while (ret);
#endif /* IS_SDLv2 */
    }
    Py_DECREF(seq);
    return list;

error:
    /* While doing a goto here, PyErr must be set */
    Py_DECREF(list);
    Py_XDECREF(seq);
    return NULL;
}

static PyObject *
pg_event_get(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *obj = NULL;
    int dopump = 1;

    static char *kwids[] = {
        "eventtype",
        "pump",
        NULL
    };

#if PY3
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Op", kwids,
                                     &obj, &dopump))
        return NULL;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Oi", kwids,
                                     &obj, &dopump))
        return NULL;
#endif

    VIDEO_INIT_CHECK();

    _pg_event_pump(dopump);

    if (obj == NULL || obj == Py_None)
        return _pg_get_all_events();
    else
        return _pg_get_seq_events(obj);
}

static PyObject *
pg_event_peek(PyObject *self, PyObject *args, PyObject *kwargs)
{
    SDL_Event event;
    int len, type, loop, res;
    PyObject *seq, *obj = NULL;
    int dopump = 1;

    static char *kwids[] = {
        "eventtype",
        "pump",
        NULL
    };

#if PY3
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Op", kwids,
                                     &obj, &dopump))
        return NULL;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Oi", kwids,
                                     &obj, &dopump))
        return NULL;
#endif

    VIDEO_INIT_CHECK();

    _pg_event_pump(dopump);

    if (obj == NULL || obj == Py_None) {
        res = PG_PEEP_EVENT_ALL(&event, 1, SDL_PEEKEVENT);
        if (res < 0)
            return RAISE(pgExc_SDLError, SDL_GetError());
        return pgEvent_New(res ? &event : NULL);
    }
    else {
        seq = _pg_eventtype_as_seq(obj, &len);
        if (!seq)
            return NULL;

        for (loop = 0; loop < len; loop++) {
            type = _pg_eventtype_from_seq(seq, loop);
            if (type == -1) {
                Py_DECREF(seq);
                return NULL;
            }
            res = PG_PEEP_EVENT(&event, 1, SDL_PEEKEVENT, type);
            if (res) {
                Py_DECREF(seq);

                if (res < 0)
                    return RAISE(pgExc_SDLError, SDL_GetError());
                return PyInt_FromLong(1);
            }
#if IS_SDLv2
            res = PG_PEEP_EVENT(&event, 1, SDL_PEEKEVENT,
                                _pg_pgevent_proxify(type));
            if (res) {
                Py_DECREF(seq);

                if (res < 0)
                    return RAISE(pgExc_SDLError, SDL_GetError());
                return PyInt_FromLong(1);
            }
#endif /* IS_SDLv2 */
        }
        Py_DECREF(seq);
        return PyInt_FromLong(0); /* No event type match. */
    }
}

/* You might notice how we do event blocking stuff on proxy events and
 * not the real SDL events. We do this because we want SDL events to pass
 * through our event filter, to do emulation stuff correctly. Then the
 * event is filtered after that */

static PyObject *
pg_event_post(PyObject *self, PyObject *obj)
{
    SDL_Event event;
    pgEventObject *e;
    int ret;

    VIDEO_INIT_CHECK();
    if (!pgEvent_Check(obj))
        return RAISE(PyExc_TypeError, "argument must be an Event object");

    e = (pgEventObject *)obj;
    if (SDL_EventState(_pg_pgevent_proxify(e->type), SDL_QUERY) == SDL_IGNORE)
        Py_RETURN_FALSE;

    pgEvent_FillUserEvent(e, &event);

    ret = SDL_PushEvent(&event);
#if IS_SDLv1
    if (ret == -1) {
        Py_DECREF(e->dict);
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    else {
        Py_RETURN_TRUE;
    }
#else /* IS_SDLv2 */
    if (ret == 1)
        Py_RETURN_TRUE;
    else {
        Py_DECREF(e->dict);
        if (ret == 0)
            Py_RETURN_FALSE;
        else
            return RAISE(pgExc_SDLError, SDL_GetError());
    }
#endif /* IS_SDLv2 */
}

static PyObject *
pg_event_set_allowed(PyObject *self, PyObject *obj)
{
    int len, loop, type;
    PyObject *seq;
    VIDEO_INIT_CHECK();

    if (obj == Py_None) {
#if IS_SDLv2
        int i;
        for (i=SDL_FIRSTEVENT; i<SDL_LASTEVENT; i++) {
            SDL_EventState(i, SDL_ENABLE);
        }
#else
        SDL_EventState(0xFF, SDL_ENABLE);
#endif /* IS_SDLv2 */
    }
    else {
        seq = _pg_eventtype_as_seq(obj, &len);
        if (!seq)
            return NULL;

        for (loop = 0; loop < len; loop++) {
            type = _pg_eventtype_from_seq(seq, loop);
            if (type == -1) {
                Py_DECREF(seq);
                return NULL;
            }
            SDL_EventState(_pg_pgevent_proxify(type), SDL_ENABLE);
        }
        Py_DECREF(seq);
    }
    Py_RETURN_NONE;
}

static PyObject *
pg_event_set_blocked(PyObject *self, PyObject *obj)
{
    int len, loop, type;
    PyObject *seq;
    VIDEO_INIT_CHECK();

    if (obj == Py_None) {
#if IS_SDLv2
        int i;
        /* Start at PGPOST_EVENTBEGIN */
        for (i=PGPOST_EVENTBEGIN; i<SDL_LASTEVENT; i++) {
            SDL_EventState(i, SDL_IGNORE);
        }
#else
        SDL_EventState(0xFF, SDL_IGNORE);
#endif /* IS_SDLv2 */
    }
    else {
        seq = _pg_eventtype_as_seq(obj, &len);
        if (!seq)
            return NULL;

        for (loop = 0; loop < len; loop++) {
            type = _pg_eventtype_from_seq(seq, loop);
            if (type == -1) {
                Py_DECREF(seq);
                return NULL;
            }
            SDL_EventState(_pg_pgevent_proxify(type), SDL_IGNORE);
        }
        Py_DECREF(seq);
    }
#if IS_SDLv2
    /* Never block SDL_WINDOWEVENT, we need them for translation */
    SDL_EventState(SDL_WINDOWEVENT, SDL_ENABLE);
    /* Never block PGE_KEYREPEAT too, its needed for pygame internal use */
    SDL_EventState(PGE_KEYREPEAT, SDL_ENABLE);
#endif /* IS_SDLv2 */
    Py_RETURN_NONE;
}

static PyObject *
pg_event_get_blocked(PyObject *self, PyObject *obj)
{
    int loop, type, len, isblocked = 0;
    PyObject *seq;

    VIDEO_INIT_CHECK();

    seq = _pg_eventtype_as_seq(obj, &len);
    if (!seq)
        return NULL;

    for (loop = 0; loop < len; loop++) {
        type = _pg_eventtype_from_seq(seq, loop);
        if (type == -1) {
            Py_DECREF(seq);
            return NULL;
        }
        if (SDL_EventState(_pg_pgevent_proxify(type), SDL_QUERY) ==
            SDL_IGNORE) {
            isblocked = 1;
            break;
        }
    }

    Py_DECREF(seq);
    return PyInt_FromLong(isblocked);
}


static PyObject *
pg_event_custom_type(PyObject *self)
{
    if (_custom_event < PG_NUMEVENTS)
        return PyInt_FromLong(_custom_event++);
    else
        return RAISE(pgExc_SDLError, "pygame.event.custom_type made too many event types.");
}

static PyMethodDef _event_methods[] = {
    {"__PYGAMEinit__", (PyCFunction)pgEvent_AutoInit, METH_NOARGS,
     "auto initialize for event module"},

    {"Event", (PyCFunction)pg_Event, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEEVENTEVENT},
    {"event_name", event_name, METH_VARARGS, DOC_PYGAMEEVENTEVENTNAME},

    {"set_grab", set_grab, METH_VARARGS, DOC_PYGAMEEVENTSETGRAB},
    {"get_grab", (PyCFunction)get_grab, METH_NOARGS, DOC_PYGAMEEVENTGETGRAB},

    {"pump", (PyCFunction)pg_event_pump, METH_NOARGS, DOC_PYGAMEEVENTPUMP},
    {"wait", (PyCFunction)pg_event_wait, METH_VARARGS | METH_KEYWORDS, DOC_PYGAMEEVENTWAIT},
    {"poll", (PyCFunction)pg_event_poll, METH_NOARGS, DOC_PYGAMEEVENTPOLL},
    {"clear", (PyCFunction)pg_event_clear, METH_VARARGS | METH_KEYWORDS, DOC_PYGAMEEVENTCLEAR},
    {"get", (PyCFunction)pg_event_get, METH_VARARGS | METH_KEYWORDS, DOC_PYGAMEEVENTGET},
    {"peek", (PyCFunction)pg_event_peek, METH_VARARGS | METH_KEYWORDS, DOC_PYGAMEEVENTPEEK},
    {"post", (PyCFunction)pg_event_post, METH_O, DOC_PYGAMEEVENTPOST},

    {"set_allowed", (PyCFunction)pg_event_set_allowed, METH_O, DOC_PYGAMEEVENTSETALLOWED},
    {"set_blocked", (PyCFunction)pg_event_set_blocked, METH_O, DOC_PYGAMEEVENTSETBLOCKED},
    {"get_blocked", (PyCFunction)pg_event_get_blocked, METH_O, DOC_PYGAMEEVENTGETBLOCKED},
    {"custom_type", (PyCFunction)pg_event_custom_type, METH_NOARGS, DOC_PYGAMEEVENTCUSTOMTYPE},


    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(event)
{
    PyObject *module, *dict, *apiobj;
    int ecode;
    static void *c_api[PYGAMEAPI_EVENT_NUMSLOTS];

#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "event",
                                         DOC_PYGAMEEVENT,
                                         -1,
                                         _event_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};
#endif

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    /* type preparation */
    if (PyType_Ready(&pgEvent_Type) < 0) {
        MODINIT_ERROR;
    }

    /* create the module */
#if PY3
    module = PyModule_Create(&_module);
#else
    module =
        Py_InitModule3(MODPREFIX "event", _event_methods, DOC_PYGAMEEVENT);
#endif
    dict = PyModule_GetDict(module);

    if (NULL == (joy_instance_map = PyDict_New())) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    if (-1 == PyDict_SetItemString(dict, "_joy_instance_map", joy_instance_map)) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    if (PyDict_SetItemString(dict, "EventType", (PyObject *)&pgEvent_Type) ==
        -1) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

#if IS_SDLv2
    if (!have_registered_events) {
        int numevents = PG_NUMEVENTS - SDL_USEREVENT;
        Uint32 user_event = SDL_RegisterEvents(numevents);

        if (user_event != SDL_USEREVENT) {
            PyErr_SetString(PyExc_ImportError,
                            "Unable to create another module instance");
            DECREF_MOD(module);
            MODINIT_ERROR;
        }

        have_registered_events = 1;
    }

    SDL_SetEventFilter(pg_event_filter, NULL);
#endif /* IS_SDLv2 */

    /* export the c api */
#if IS_SDLv2
    assert(PYGAMEAPI_EVENT_NUMSLOTS == 6);
#endif /* IS_SDLv2 */
    c_api[0] = &pgEvent_Type;
    c_api[1] = pgEvent_New;
    c_api[2] = pgEvent_New2;
    c_api[3] = pgEvent_FillUserEvent;
#if IS_SDLv2
    c_api[4] = pg_EnableKeyRepeat;
    c_api[5] = pg_GetKeyRepeat;
#endif /* IS_SDLv2 */
    apiobj = encapsulate_api(c_api, "event");
    if (apiobj == NULL) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF(apiobj);
    if (ecode) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    MODINIT_RETURN(module);
}
