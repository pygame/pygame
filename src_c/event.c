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

#define JOYEVENT_INSTANCE_ID "instance_id"
#define JOYEVENT_DEVICE_INDEX "device_index"

/* Define custom functions for peep events, for SDL1/2 compat */
#define PG_PEEP_EVENT(a, b, c, d) SDL_PeepEvents(a, b, c, d, d)
#define PG_PEEP_EVENT_ALL(x, y, z) \
    SDL_PeepEvents(x, y, z, SDL_FIRSTEVENT, SDL_LASTEVENT)

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

/* This defines the maximum values of key-press and unicode values we
 * can store at a time, it is used for determining the unicode attribute
 * for KEYUP events. Now that its set to 15, it means that a user can
 * simultaneously hold 15 keys (who would do that?) and on release, all
 * KEYUP events will have unicode attribute. Why 15? You can set any
 * arbitrary number you like ;) */
#define MAX_SCAN_UNICODE 15

/* SDL mutex to be held in the event filter when global state is modified.
 * This mutex is intentionally immortalised (never freed during the entire
 * duration of the program) because its cleanup can be messy with multiple
 * threads trying to use it. Since it's a singleton we don't need to worry
 * about memory leaks */
#ifndef __EMSCRIPTEN__
/* emscripten does not allow multithreading for now and SDL_CreateMutex fails.
 * Don't bother with mutexes on emscripten for now */
static SDL_mutex *pg_evfilter_mutex = NULL;
#endif

static struct ScanAndUnicode {
    SDL_Scancode key;
    char unicode[UNICODE_LEN];
} scanunicode[MAX_SCAN_UNICODE] = {{0}};

static int pg_key_repeat_delay = 0;
static int pg_key_repeat_interval = 0;

static SDL_TimerID _pg_repeat_timer = 0;
static SDL_Event _pg_repeat_event;
static SDL_Event _pg_last_keydown_event = {0};

#ifdef __EMSCRIPTEN__
/* these macros are no-op here */
#define PG_LOCK_EVFILTER_MUTEX
#define PG_UNLOCK_EVFILTER_MUTEX
#else /* not on emscripten */

#define PG_LOCK_EVFILTER_MUTEX                                             \
    if (pg_evfilter_mutex) {                                               \
        if (SDL_LockMutex(pg_evfilter_mutex) < 0) {                        \
            /* TODO: better error handling with future error-event API */  \
            /* since this error is very rare, we can completely give up if \
             * this happens for now */                                     \
            printf("Fatal pygame error in SDL_LockMutex: %s",              \
                   SDL_GetError());                                        \
            PG_EXIT(1);                                                    \
        }                                                                  \
    }

#define PG_UNLOCK_EVFILTER_MUTEX                                           \
    if (pg_evfilter_mutex) {                                               \
        if (SDL_UnlockMutex(pg_evfilter_mutex) < 0) {                      \
            /* TODO: handle errors with future error-event API */          \
            /* since this error is very rare, we can completely give up if \
             * this happens for now */                                     \
            printf("Fatal pygame error in SDL_UnlockMutex: %s",            \
                   SDL_GetError());                                        \
            PG_EXIT(1);                                                    \
        }                                                                  \
    }
#endif /* not on emscripten */

static Uint32
_pg_repeat_callback(Uint32 interval, void *param)
{
    /* This function is called in a SDL Timer thread */
    PG_LOCK_EVFILTER_MUTEX
    /* This assignment only shallow-copies, but SDL_KeyboardEvent does not have
     * any pointer values so it's safe to do */
    SDL_Event repeat_event_copy = _pg_repeat_event;
    int repeat_interval_copy = pg_key_repeat_interval;
    PG_UNLOCK_EVFILTER_MUTEX

    repeat_event_copy.type = PGE_KEYREPEAT;
    repeat_event_copy.key.state = SDL_PRESSED;
    repeat_event_copy.key.repeat = 1;
    SDL_PushEvent(&repeat_event_copy);
    return repeat_interval_copy;
}

/* This function attempts to determine the unicode attribute from
 * the keydown/keyup event. This is used as a last-resort, in case we
 * could not determine the unicode from TEXTINPUT field. Why?
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
        /* Control Key held, send control-key related unicode. */
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
 * did not support unicode characters that took up 4 bytes. In case this
 * bit of code is not clear, here is a python equivalent
def _pg_strip_utf8(string):
    if chr(string[0]) <= 0xFFFF:
        return string[0]
    else:
        return ""
*/
static void
_pg_strip_utf8(char *str, char *ret)
{
    Uint8 firstbyte = (Uint8)*str;

    /* Zero unicode buffer */
    memset(ret, 0, UNICODE_LEN);

    /* 1111 0000 is 0xF0 */
    if (firstbyte >= 0xF0) {
        /* Too large UTF8 string, do nothing */
        return;
    }

    /* 1110 0000 is 0xE0 */
    if (firstbyte >= 0xE0) {
        /* Copy first 3 bytes */
        memcpy(ret, str, 3);
    }
    /* 1100 0000 is 0xC0 */
    else if (firstbyte >= 0xC0) {
        /* Copy first 2 bytes */
        memcpy(ret, str, 2);
    }
    /* 1000 0000 is 0x80 */
    else if (firstbyte < 0x80) {
        /* Copy first byte */
        memcpy(ret, str, 1);
    }
}

static int
_pg_put_event_unicode(SDL_Event *event, char *uni)
{
    int i;
    for (i = 0; i < MAX_SCAN_UNICODE; i++) {
        if (!scanunicode[i].key) {
            scanunicode[i].key = event->key.keysym.scancode;
            _pg_strip_utf8(uni, scanunicode[i].unicode);
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
    for (i = 0; i < MAX_SCAN_UNICODE; i++) {
        if (scanunicode[i].key == event->key.keysym.scancode) {
            if (event->type == SDL_KEYUP) {
                /* mark the position as free real estate for other
                 * events to occupy. */
                scanunicode[i].key = 0;
            }
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

#define _PG_HANDLE_PROXIFY(name) \
    case SDL_##name:             \
    case PGPOST_##name:          \
        return proxify ? PGPOST_##name : SDL_##name

#define _PG_HANDLE_PROXIFY_PGE(name) \
    case PGE_##name:                 \
    case PGPOST_##name:              \
        return proxify ? PGPOST_##name : PGE_##name

/* The next three functions are used for proxying SDL events to and from
 * PGPOST_* events.
 *
 * Some SDL1 events (SDL_ACTIVEEVENT, SDL_VIDEORESIZE and SDL_VIDEOEXPOSE)
 * are redefined with SDL2, they HAVE to be proxied.
 *
 * SDL_USEREVENT is not proxied, because with SDL2, pygame assigns a
 * different event in place of SDL_USEREVENT, and users use PGE_USEREVENT
 *
 * Each WINDOW_* event must be defined twice, once as an event, and also
 * again, as a proxy event. WINDOW_* events MUST be proxied.
 */

static Uint32
_pg_pgevent_proxify_helper(Uint32 type, Uint8 proxify)
{
    switch (type) {
        _PG_HANDLE_PROXIFY(ACTIVEEVENT);
        _PG_HANDLE_PROXIFY(APP_TERMINATING);
        _PG_HANDLE_PROXIFY(APP_LOWMEMORY);
        _PG_HANDLE_PROXIFY(APP_WILLENTERBACKGROUND);
        _PG_HANDLE_PROXIFY(APP_DIDENTERBACKGROUND);
        _PG_HANDLE_PROXIFY(APP_WILLENTERFOREGROUND);
        _PG_HANDLE_PROXIFY(APP_DIDENTERFOREGROUND);
        _PG_HANDLE_PROXIFY(AUDIODEVICEADDED);
        _PG_HANDLE_PROXIFY(AUDIODEVICEREMOVED);
        _PG_HANDLE_PROXIFY(CLIPBOARDUPDATE);
        _PG_HANDLE_PROXIFY(CONTROLLERAXISMOTION);
        _PG_HANDLE_PROXIFY(CONTROLLERBUTTONDOWN);
        _PG_HANDLE_PROXIFY(CONTROLLERBUTTONUP);
        _PG_HANDLE_PROXIFY(CONTROLLERDEVICEADDED);
        _PG_HANDLE_PROXIFY(CONTROLLERDEVICEREMOVED);
        _PG_HANDLE_PROXIFY(CONTROLLERDEVICEREMAPPED);
#if SDL_VERSION_ATLEAST(2, 0, 14)
        _PG_HANDLE_PROXIFY(CONTROLLERTOUCHPADDOWN);
        _PG_HANDLE_PROXIFY(CONTROLLERTOUCHPADMOTION);
        _PG_HANDLE_PROXIFY(CONTROLLERTOUCHPADUP);
        _PG_HANDLE_PROXIFY(CONTROLLERSENSORUPDATE);
#endif
        _PG_HANDLE_PROXIFY(DOLLARGESTURE);
        _PG_HANDLE_PROXIFY(DOLLARRECORD);
        _PG_HANDLE_PROXIFY(DROPFILE);
        _PG_HANDLE_PROXIFY(DROPTEXT);
        _PG_HANDLE_PROXIFY(DROPBEGIN);
        _PG_HANDLE_PROXIFY(DROPCOMPLETE);
        _PG_HANDLE_PROXIFY(FINGERMOTION);
        _PG_HANDLE_PROXIFY(FINGERDOWN);
        _PG_HANDLE_PROXIFY(FINGERUP);
        _PG_HANDLE_PROXIFY(KEYDOWN);
        _PG_HANDLE_PROXIFY(KEYUP);
        _PG_HANDLE_PROXIFY(KEYMAPCHANGED);
        _PG_HANDLE_PROXIFY(JOYAXISMOTION);
        _PG_HANDLE_PROXIFY(JOYBALLMOTION);
        _PG_HANDLE_PROXIFY(JOYHATMOTION);
        _PG_HANDLE_PROXIFY(JOYBUTTONDOWN);
        _PG_HANDLE_PROXIFY(JOYBUTTONUP);
        _PG_HANDLE_PROXIFY(JOYDEVICEADDED);
        _PG_HANDLE_PROXIFY(JOYDEVICEREMOVED);
#if SDL_VERSION_ATLEAST(2, 0, 14)
        _PG_HANDLE_PROXIFY(LOCALECHANGED);
#endif
        _PG_HANDLE_PROXIFY(MOUSEMOTION);
        _PG_HANDLE_PROXIFY(MOUSEBUTTONDOWN);
        _PG_HANDLE_PROXIFY(MOUSEBUTTONUP);
        _PG_HANDLE_PROXIFY(MOUSEWHEEL);
        _PG_HANDLE_PROXIFY(MULTIGESTURE);
        _PG_HANDLE_PROXIFY(NOEVENT);
        _PG_HANDLE_PROXIFY(QUIT);
        _PG_HANDLE_PROXIFY(RENDER_TARGETS_RESET);
        _PG_HANDLE_PROXIFY(RENDER_DEVICE_RESET);
        _PG_HANDLE_PROXIFY(SYSWMEVENT);
        _PG_HANDLE_PROXIFY(TEXTEDITING);
        _PG_HANDLE_PROXIFY(TEXTINPUT);
        _PG_HANDLE_PROXIFY(VIDEORESIZE);
        _PG_HANDLE_PROXIFY(VIDEOEXPOSE);
        _PG_HANDLE_PROXIFY_PGE(MIDIIN);
        _PG_HANDLE_PROXIFY_PGE(MIDIOUT);
        _PG_HANDLE_PROXIFY_PGE(WINDOWSHOWN);
        _PG_HANDLE_PROXIFY_PGE(WINDOWHIDDEN);
        _PG_HANDLE_PROXIFY_PGE(WINDOWEXPOSED);
        _PG_HANDLE_PROXIFY_PGE(WINDOWMOVED);
        _PG_HANDLE_PROXIFY_PGE(WINDOWRESIZED);
        _PG_HANDLE_PROXIFY_PGE(WINDOWSIZECHANGED);
        _PG_HANDLE_PROXIFY_PGE(WINDOWMINIMIZED);
        _PG_HANDLE_PROXIFY_PGE(WINDOWMAXIMIZED);
        _PG_HANDLE_PROXIFY_PGE(WINDOWRESTORED);
        _PG_HANDLE_PROXIFY_PGE(WINDOWENTER);
        _PG_HANDLE_PROXIFY_PGE(WINDOWLEAVE);
        _PG_HANDLE_PROXIFY_PGE(WINDOWFOCUSGAINED);
        _PG_HANDLE_PROXIFY_PGE(WINDOWFOCUSLOST);
        _PG_HANDLE_PROXIFY_PGE(WINDOWCLOSE);
        _PG_HANDLE_PROXIFY_PGE(WINDOWTAKEFOCUS);
        _PG_HANDLE_PROXIFY_PGE(WINDOWHITTEST);
        _PG_HANDLE_PROXIFY_PGE(WINDOWICCPROFCHANGED);
        _PG_HANDLE_PROXIFY_PGE(WINDOWDISPLAYCHANGED);
        default:
            return type;
    }
}

static Uint32
_pg_pgevent_proxify(Uint32 type)
{
    return _pg_pgevent_proxify_helper(type, 1);
}

static Uint32
_pg_pgevent_deproxify(Uint32 type)
{
    return _pg_pgevent_proxify_helper(type, 0);
}

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
_pg_remove_pending_VIDEORESIZE(void *userdata, SDL_Event *event)
{
    SDL_Event *new_event = (SDL_Event *)userdata;

    if (event->type == SDL_VIDEORESIZE &&
        event->window.windowID == new_event->window.windowID) {
        /* We're about to post a new size event, drop the old ones */
        return 0;
    }
    return 1;
}

static int SDLCALL
_pg_remove_pending_VIDEOEXPOSE(void *userdata, SDL_Event *event)
{
    SDL_Event *new_event = (SDL_Event *)userdata;

    if (event->type == SDL_VIDEOEXPOSE &&
        event->window.windowID == new_event->window.windowID) {
        /* We're about to post a new videoexpose event, drop the old ones */
        return 0;
    }
    return 1;
}

/* SDL 2 to SDL 1.2 event mapping and SDL 1.2 key repeat emulation,
 * this can alter events in-place.
 * This function can be called from multiple threads, so a mutex must be held
 * when this function tries to modify any global state (the mutex is not needed
 * on all branches of this function) */
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

        PG_LOCK_EVFILTER_MUTEX
        if (pg_key_repeat_delay > 0) {
            if (_pg_repeat_timer)
                SDL_RemoveTimer(_pg_repeat_timer);

            _pg_repeat_event = *event;
            _pg_repeat_timer =
                SDL_AddTimer(pg_key_repeat_delay, _pg_repeat_callback, NULL);
        }

        /* store the keydown event for later in the SDL_TEXTINPUT */
        _pg_last_keydown_event = *event;
        PG_UNLOCK_EVFILTER_MUTEX
    }

    else if (event->type == SDL_TEXTINPUT) {
        PG_LOCK_EVFILTER_MUTEX
        if (_pg_last_keydown_event.type) {
            _pg_put_event_unicode(&_pg_last_keydown_event, event->text.text);
            _pg_last_keydown_event.type = 0;
        }
        PG_UNLOCK_EVFILTER_MUTEX
    }

    else if (event->type == PGE_KEYREPEAT) {
        event->type = SDL_KEYDOWN;
    }

    else if (event->type == SDL_KEYUP) {
        PG_LOCK_EVFILTER_MUTEX
        if (_pg_repeat_timer && _pg_repeat_event.key.keysym.scancode ==
                                    event->key.keysym.scancode) {
            SDL_RemoveTimer(_pg_repeat_timer);
            _pg_repeat_timer = 0;
        }
        PG_UNLOCK_EVFILTER_MUTEX
    }

    else if (event->type == SDL_MOUSEBUTTONDOWN ||
             event->type == SDL_MOUSEBUTTONUP) {
        if (event->button.button & PGM_BUTTON_KEEP)
            event->button.button ^= PGM_BUTTON_KEEP;
        else if (event->button.button >= PGM_BUTTON_WHEELUP)
            event->button.button += (PGM_BUTTON_X1 - PGM_BUTTON_WHEELUP);
    }

    else if (event->type == SDL_MOUSEWHEEL) {
        // #691 We are not moving wheel!
        if (!event->wheel.y && !event->wheel.x)
            return 0;

        SDL_GetMouseState(&x, &y);
        /* Generate a MouseButtonDown event and MouseButtonUp for
         * compatibility. https://wiki.libsdl.org/SDL_MouseWheelEvent
         */
        newdownevent.type = SDL_MOUSEBUTTONDOWN;
        newdownevent.button.x = x;
        newdownevent.button.y = y;
        newdownevent.button.state = SDL_PRESSED;
        newdownevent.button.clicks = 1;
        newdownevent.button.which = event->button.which;

        newupevent.type = SDL_MOUSEBUTTONUP;
        newupevent.button.x = x;
        newupevent.button.y = y;
        newupevent.button.state = SDL_RELEASED;
        newupevent.button.clicks = 1;
        newupevent.button.which = event->button.which;

        /* Use a for loop to simulate multiple events, because SDL 1
         * works that way */
        for (i = 0; i < abs(event->wheel.y); i++) {
            /* Do this in the loop because button.button is mutated before it
             * is posted from this filter */
            if (event->wheel.y > 0) {
                newdownevent.button.button = newupevent.button.button =
                    PGM_BUTTON_WHEELUP | PGM_BUTTON_KEEP;
            }
            else {
                newdownevent.button.button = newupevent.button.button =
                    PGM_BUTTON_WHEELDOWN | PGM_BUTTON_KEEP;
            }
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

/* The two keyrepeat functions below modify state accessed by the event filter,
 * so they too need to hold the safety mutex */
static int
pg_EnableKeyRepeat(int delay, int interval)
{
    if (delay < 0 || interval < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "delay and interval must equal at least 0");
        return -1;
    }
    PG_LOCK_EVFILTER_MUTEX
    pg_key_repeat_delay = delay;
    pg_key_repeat_interval = interval;
    PG_UNLOCK_EVFILTER_MUTEX
    return 0;
}

static void
pg_GetKeyRepeat(int *delay, int *interval)
{
    PG_LOCK_EVFILTER_MUTEX
    *delay = pg_key_repeat_delay;
    *interval = pg_key_repeat_interval;
    PG_UNLOCK_EVFILTER_MUTEX
}

static PyObject *
pgEvent_AutoQuit(PyObject *self, PyObject *_null)
{
    if (_pg_event_is_init) {
        PG_LOCK_EVFILTER_MUTEX
        if (_pg_repeat_timer) {
            SDL_RemoveTimer(_pg_repeat_timer);
            _pg_repeat_timer = 0;
        }
        PG_UNLOCK_EVFILTER_MUTEX
        /* The main reason for _custom_event to be reset here is so we
         * can have a unit test that checks if pygame.event.custom_type()
         * stops returning new types when they are finished, without that
         * test preventing further tests from getting a custom event type.*/
        _custom_event = _PGE_CUSTOM_EVENT_INIT;
    }
    _pg_event_is_init = 0;
    Py_RETURN_NONE;
}

static PyObject *
pgEvent_AutoInit(PyObject *self, PyObject *_null)
{
    if (!_pg_event_is_init) {
        pg_key_repeat_delay = 0;
        pg_key_repeat_interval = 0;
#ifndef __EMSCRIPTEN__
        if (!pg_evfilter_mutex) {
            /* Create mutex only if it has not been created already */
            pg_evfilter_mutex = SDL_CreateMutex();
            if (!pg_evfilter_mutex)
                return RAISE(pgExc_SDLError, SDL_GetError());
        }
#endif
        SDL_SetEventFilter(pg_event_filter, NULL);
    }
    _pg_event_is_init = 1;
    Py_RETURN_NONE;
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
        case SDL_APP_TERMINATING:
            return "AppTerminating";
        case SDL_APP_LOWMEMORY:
            return "AppLowMemory";
        case SDL_APP_WILLENTERBACKGROUND:
            return "AppWillEnterBackground";
        case SDL_APP_DIDENTERBACKGROUND:
            return "AppDidEnterBackground";
        case SDL_APP_WILLENTERFOREGROUND:
            return "AppWillEnterForeground";
        case SDL_APP_DIDENTERFOREGROUND:
            return "AppDidEnterForeground";
        case SDL_CLIPBOARDUPDATE:
            return "ClipboardUpdate";
        case SDL_KEYDOWN:
            return "KeyDown";
        case SDL_KEYUP:
            return "KeyUp";
        case SDL_KEYMAPCHANGED:
            return "KeyMapChanged";
#if SDL_VERSION_ATLEAST(2, 0, 14)
        case SDL_LOCALECHANGED:
            return "LocaleChanged";
#endif
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
        case SDL_DROPTEXT:
            return "DropText";
        case SDL_DROPBEGIN:
            return "DropBegin";
        case SDL_DROPCOMPLETE:
            return "DropComplete";
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
#if SDL_VERSION_ATLEAST(2, 0, 14)
        case SDL_CONTROLLERTOUCHPADDOWN:
            return "ControllerTouchpadDown";
        case SDL_CONTROLLERTOUCHPADMOTION:
            return "ControllerTouchpadMotion";
        case SDL_CONTROLLERTOUCHPADUP:
            return "ControllerTouchpadUp";
        case SDL_CONTROLLERSENSORUPDATE:
            return "ControllerSensorUpdate";
#endif /*SDL_VERSION_ATLEAST(2, 0, 14)*/
        case SDL_AUDIODEVICEADDED:
            return "AudioDeviceAdded";
        case SDL_AUDIODEVICEREMOVED:
            return "AudioDeviceRemoved";
        case SDL_RENDER_TARGETS_RESET:
            return "RenderTargetsReset";
        case SDL_RENDER_DEVICE_RESET:
            return "RenderDeviceReset";
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
        case PGE_WINDOWICCPROFCHANGED:
            return "WindowICCProfChanged";
        case PGE_WINDOWDISPLAYCHANGED:
            return "WindowDisplayChanged";
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

static PyObject *
get_joy_guid(int device_index)
{
    char strguid[33];
    SDL_JoystickGUID guid = SDL_JoystickGetDeviceGUID(device_index);

    SDL_JoystickGetGUIDString(guid, strguid, 33);
    return PyUnicode_FromString(strguid);
}

/** Try to insert the instance ID for a new device into the joystick mapping.
 */
void
_joy_map_add(int device_index)
{
    int instance_id = (int)SDL_JoystickGetDeviceInstanceID(device_index);
    PyObject *k, *v;
    if (instance_id != -1) {
        k = PyLong_FromLong(instance_id);
        v = PyLong_FromLong(device_index);
        if (k != NULL && v != NULL) {
            PyDict_SetItem(joy_instance_map, k, v);
        }
        Py_XDECREF(k);
        Py_XDECREF(v);
    }
}

/** Look up a device ID for an instance ID. */
PyObject *
_joy_map_instance(int instance_id)
{
    PyObject *v, *k = PyLong_FromLong(instance_id);
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
_joy_map_discard(int instance_id)
{
    PyObject *k = PyLong_FromLong(instance_id);

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
    long gain;
    long state;

    /* check if a proxy event or userevent was posted */
    if (event->type >= PGPOST_EVENTBEGIN && event->user.code == USEROBJ_CHECK)
        return (PyObject *)event->user.data1;

    dict = PyDict_New();
    if (!dict)
        return NULL;

    switch (event->type) {
        case SDL_VIDEORESIZE:
            obj = Py_BuildValue("(ii)", event->window.data1,
                                event->window.data2);
            _pg_insobj(dict, "size", obj);
            _pg_insobj(dict, "w", PyLong_FromLong(event->window.data1));
            _pg_insobj(dict, "h", PyLong_FromLong(event->window.data2));
            break;
        case SDL_ACTIVEEVENT:
            switch (event->window.event) {
                case SDL_WINDOWEVENT_ENTER:
                    gain = 1;
                    state = SDL_APPMOUSEFOCUS;
                    break;
                case SDL_WINDOWEVENT_LEAVE:
                    gain = 0;
                    state = SDL_APPMOUSEFOCUS;
                    break;
                case SDL_WINDOWEVENT_FOCUS_GAINED:
                    gain = 1;
                    state = SDL_APPINPUTFOCUS;
                    break;
                case SDL_WINDOWEVENT_FOCUS_LOST:
                    gain = 0;
                    state = SDL_APPINPUTFOCUS;
                    break;
                case SDL_WINDOWEVENT_MINIMIZED:
                    gain = 0;
                    state = SDL_APPACTIVE;
                    break;
                default:
                    assert(event->window.event == SDL_WINDOWEVENT_RESTORED);
                    gain = 1;
                    state = SDL_APPACTIVE;
            }
            _pg_insobj(dict, "gain", PyLong_FromLong(gain));
            _pg_insobj(dict, "state", PyLong_FromLong(state));
            break;
        case SDL_KEYDOWN:
        case SDL_KEYUP:
            PG_LOCK_EVFILTER_MUTEX
            /* this accesses state also accessed the event filter, so lock */
            _pg_insobj(dict, "unicode", _pg_get_event_unicode(event));
            PG_UNLOCK_EVFILTER_MUTEX
            _pg_insobj(dict, "key", PyLong_FromLong(event->key.keysym.sym));
            _pg_insobj(dict, "mod", PyLong_FromLong(event->key.keysym.mod));
            _pg_insobj(dict, "scancode",
                       PyLong_FromLong(event->key.keysym.scancode));
            break;
        case SDL_MOUSEMOTION:
            obj = Py_BuildValue("(ii)", event->motion.x, event->motion.y);
            _pg_insobj(dict, "pos", obj);
            obj =
                Py_BuildValue("(ii)", event->motion.xrel, event->motion.yrel);
            _pg_insobj(dict, "rel", obj);
            if ((tuple = PyTuple_New(3))) {
                PyTuple_SET_ITEM(tuple, 0,
                                 PyLong_FromLong((event->motion.state &
                                                  SDL_BUTTON(1)) != 0));
                PyTuple_SET_ITEM(tuple, 1,
                                 PyLong_FromLong((event->motion.state &
                                                  SDL_BUTTON(2)) != 0));
                PyTuple_SET_ITEM(tuple, 2,
                                 PyLong_FromLong((event->motion.state &
                                                  SDL_BUTTON(3)) != 0));
                _pg_insobj(dict, "buttons", tuple);
            }
            _pg_insobj(
                dict, "touch",
                PyBool_FromLong((event->motion.which == SDL_TOUCH_MOUSEID)));
            break;
        case SDL_MOUSEBUTTONDOWN:
        case SDL_MOUSEBUTTONUP:
            obj = Py_BuildValue("(ii)", event->button.x, event->button.y);
            _pg_insobj(dict, "pos", obj);
            _pg_insobj(dict, "button", PyLong_FromLong(event->button.button));
            _pg_insobj(
                dict, "touch",
                PyBool_FromLong((event->button.which == SDL_TOUCH_MOUSEID)));
            break;
        case SDL_JOYAXISMOTION:
            _pg_insobj(dict, "joy", _joy_map_instance(event->jaxis.which));
            _pg_insobj(dict, "instance_id",
                       PyLong_FromLong(event->jaxis.which));
            _pg_insobj(dict, "axis", PyLong_FromLong(event->jaxis.axis));
            _pg_insobj(dict, "value",
                       PyFloat_FromDouble(event->jaxis.value / 32767.0));
            break;
        case SDL_JOYBALLMOTION:
            _pg_insobj(dict, "joy", _joy_map_instance(event->jaxis.which));
            _pg_insobj(dict, "instance_id",
                       PyLong_FromLong(event->jball.which));
            _pg_insobj(dict, "ball", PyLong_FromLong(event->jball.ball));
            obj = Py_BuildValue("(ii)", event->jball.xrel, event->jball.yrel);
            _pg_insobj(dict, "rel", obj);
            break;
        case SDL_JOYHATMOTION:
            _pg_insobj(dict, "joy", _joy_map_instance(event->jaxis.which));
            _pg_insobj(dict, "instance_id",
                       PyLong_FromLong(event->jhat.which));
            _pg_insobj(dict, "hat", PyLong_FromLong(event->jhat.hat));
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
            _pg_insobj(dict, "instance_id",
                       PyLong_FromLong(event->jbutton.which));
            _pg_insobj(dict, "button", PyLong_FromLong(event->jbutton.button));
            break;
        case PGE_WINDOWDISPLAYCHANGED:
            _pg_insobj(dict, "display_index",
                       PyLong_FromLong(event->window.data1));
        case PGE_WINDOWMOVED:
        case PGE_WINDOWRESIZED:
        case PGE_WINDOWSIZECHANGED:
            /*other PGE_WINDOW* events do not have attributes */
            _pg_insobj(dict, "x", PyLong_FromLong(event->window.data1));
            _pg_insobj(dict, "y", PyLong_FromLong(event->window.data2));
            break;
        case SDL_AUDIODEVICEADDED:
        case SDL_AUDIODEVICEREMOVED:
            _pg_insobj(
                dict, "which",
                PyLong_FromLong(
                    event->adevice
                        .which));  // The audio device index for the ADDED
                                   // event (valid until next
                                   // SDL_GetNumAudioDevices() call),
                                   // SDL_AudioDeviceID for the REMOVED event
            _pg_insobj(dict, "iscapture",
                       PyLong_FromLong(event->adevice.iscapture));
            break;
        case SDL_FINGERMOTION:
        case SDL_FINGERDOWN:
        case SDL_FINGERUP:
            /* https://wiki.libsdl.org/SDL_TouchFingerEvent */
            _pg_insobj(dict, "touch_id",
                       PyLong_FromLongLong(event->tfinger.touchId));
            _pg_insobj(dict, "finger_id",
                       PyLong_FromLongLong(event->tfinger.fingerId));
            _pg_insobj(dict, "x", PyFloat_FromDouble(event->tfinger.x));
            _pg_insobj(dict, "y", PyFloat_FromDouble(event->tfinger.y));
            _pg_insobj(dict, "dx", PyFloat_FromDouble(event->tfinger.dx));
            _pg_insobj(dict, "dy", PyFloat_FromDouble(event->tfinger.dy));
            _pg_insobj(dict, "pressure",
                       PyFloat_FromDouble(event->tfinger.dy));
            break;
        case SDL_MULTIGESTURE:
            /* https://wiki.libsdl.org/SDL_MultiGestureEvent */
            _pg_insobj(dict, "touch_id",
                       PyLong_FromLongLong(event->mgesture.touchId));
            _pg_insobj(dict, "x", PyFloat_FromDouble(event->mgesture.x));
            _pg_insobj(dict, "y", PyFloat_FromDouble(event->mgesture.y));
            _pg_insobj(dict, "rotated",
                       PyFloat_FromDouble(event->mgesture.dTheta));
            _pg_insobj(dict, "pinched",
                       PyFloat_FromDouble(event->mgesture.dDist));
            _pg_insobj(dict, "num_fingers",
                       PyLong_FromLong(event->mgesture.numFingers));
            break;
        case SDL_MOUSEWHEEL:
            /* https://wiki.libsdl.org/SDL_MouseWheelEvent */
#ifndef NO_SDL_MOUSEWHEEL_FLIPPED
            _pg_insobj(dict, "flipped",
                       PyBool_FromLong(event->wheel.direction ==
                                       SDL_MOUSEWHEEL_FLIPPED));
#else
            _pg_insobj(dict, "flipped", PyBool_FromLong(0));
#endif
            _pg_insobj(dict, "x", PyLong_FromLong(event->wheel.x));
            _pg_insobj(dict, "y", PyLong_FromLong(event->wheel.y));

#if SDL_VERSION_ATLEAST(2, 0, 18)
            _pg_insobj(dict, "precise_x",
                       PyFloat_FromDouble((double)event->wheel.preciseX));
            _pg_insobj(dict, "precise_y",
                       PyFloat_FromDouble((double)event->wheel.preciseY));

#else /* ~SDL_VERSION_ATLEAST(2, 0, 18) */
            /* fallback to regular x and y when SDL version used does not
             * support precise fields */
            _pg_insobj(dict, "precise_x",
                       PyFloat_FromDouble((double)event->wheel.x));
            _pg_insobj(dict, "precise_y",
                       PyFloat_FromDouble((double)event->wheel.y));

#endif /* ~SDL_VERSION_ATLEAST(2, 0, 18) */
            _pg_insobj(
                dict, "touch",
                PyBool_FromLong((event->wheel.which == SDL_TOUCH_MOUSEID)));

            break;
        case SDL_TEXTINPUT:
            /* https://wiki.libsdl.org/SDL_TextInputEvent */
            _pg_insobj(dict, "text", PyUnicode_FromString(event->text.text));
            break;
        case SDL_TEXTEDITING:
            /* https://wiki.libsdl.org/SDL_TextEditingEvent */
            _pg_insobj(dict, "text", PyUnicode_FromString(event->edit.text));
            _pg_insobj(dict, "start", PyLong_FromLong(event->edit.start));
            _pg_insobj(dict, "length", PyLong_FromLong(event->edit.length));
            break;
        /*  https://wiki.libsdl.org/SDL_DropEvent */
        case SDL_DROPFILE:
            _pg_insobj(dict, "file", PyUnicode_FromString(event->drop.file));
            SDL_free(event->drop.file);
            break;
        case SDL_DROPTEXT:
            _pg_insobj(dict, "text", PyUnicode_FromString(event->drop.file));
            SDL_free(event->drop.file);
            break;
        case SDL_DROPBEGIN:
        case SDL_DROPCOMPLETE:
            break;
        case SDL_CONTROLLERAXISMOTION:
            /* https://wiki.libsdl.org/SDL_ControllerAxisEvent */
            _pg_insobj(dict, "instance_id",
                       PyLong_FromLong(event->caxis.which));
            _pg_insobj(dict, "axis", PyLong_FromLong(event->caxis.axis));
            _pg_insobj(dict, "value", PyLong_FromLong(event->caxis.value));
            break;
        case SDL_CONTROLLERBUTTONDOWN:
        case SDL_CONTROLLERBUTTONUP:
            /* https://wiki.libsdl.org/SDL_ControllerButtonEvent */
            _pg_insobj(dict, "instance_id",
                       PyLong_FromLong(event->cbutton.which));
            _pg_insobj(dict, "button", PyLong_FromLong(event->cbutton.button));
            break;
        case SDL_CONTROLLERDEVICEADDED:
            _pg_insobj(dict, "device_index",
                       PyLong_FromLong(event->cdevice.which));
            _pg_insobj(dict, "guid", get_joy_guid(event->jdevice.which));
            break;
        case SDL_JOYDEVICEADDED:
            _joy_map_add(event->jdevice.which);
            _pg_insobj(dict, "device_index",
                       PyLong_FromLong(event->jdevice.which));
            _pg_insobj(dict, "guid", get_joy_guid(event->jdevice.which));
            break;
        case SDL_CONTROLLERDEVICEREMOVED:
        case SDL_CONTROLLERDEVICEREMAPPED:
            /* https://wiki.libsdl.org/SDL_ControllerDeviceEvent */
            _pg_insobj(dict, "instance_id",
                       PyLong_FromLong(event->cdevice.which));
            break;
        case SDL_JOYDEVICEREMOVED:
            _joy_map_discard(event->jdevice.which);
            _pg_insobj(dict, "instance_id",
                       PyLong_FromLong(event->jdevice.which));
            break;
#if SDL_VERSION_ATLEAST(2, 0, 14)
        case SDL_CONTROLLERTOUCHPADDOWN:
        case SDL_CONTROLLERTOUCHPADMOTION:
        case SDL_CONTROLLERTOUCHPADUP:
            _pg_insobj(dict, "instance_id",
                       PyLong_FromLong(event->ctouchpad.which));
            _pg_insobj(dict, "touch_id",
                       PyLong_FromLongLong(event->ctouchpad.touchpad));
            _pg_insobj(dict, "finger_id",
                       PyLong_FromLongLong(event->ctouchpad.finger));
            _pg_insobj(dict, "x", PyFloat_FromDouble(event->ctouchpad.x));
            _pg_insobj(dict, "y", PyFloat_FromDouble(event->ctouchpad.y));
            _pg_insobj(dict, "pressure",
                       PyFloat_FromDouble(event->ctouchpad.pressure));
            break;
#endif /*SDL_VERSION_ATLEAST(2, 0, 14)*/

#ifdef WIN32
        case SDL_SYSWMEVENT:
            _pg_insobj(dict, "hwnd",
                       PyLong_FromLongLong(
                           (long long)(event->syswm.msg->msg.win.hwnd)));
            _pg_insobj(dict, "msg",
                       PyLong_FromLong(event->syswm.msg->msg.win.msg));
            _pg_insobj(dict, "wparam",
                       PyLong_FromLongLong(event->syswm.msg->msg.win.wParam));
            _pg_insobj(dict, "lparam",
                       PyLong_FromLongLong(event->syswm.msg->msg.win.lParam));
            break;
#endif /* WIN32 */

#if (defined(unix) || defined(__unix__) || defined(_AIX) ||     \
     defined(__OpenBSD__)) &&                                   \
    (defined(SDL_VIDEO_DRIVER_X11) && !defined(__CYGWIN32__) && \
     !defined(ENABLE_NANOX) && !defined(__QNXNTO__))
        case SDL_SYSWMEVENT:
            if (event->syswm.msg->subsystem == SDL_SYSWM_X11) {
                XEvent *xevent = (XEvent *)&event->syswm.msg->msg.x11.event;
                obj =
                    PyBytes_FromStringAndSize((char *)xevent, sizeof(XEvent));
                _pg_insobj(dict, "event", obj);
            }
            break;
#endif /* (defined(unix) || ... */
    }  /* switch (event->type) */
    /* Events that dont have any attributes are not handled in switch
     * statement */
    SDL_Window *window;
    switch (event->type) {
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
        case PGE_WINDOWICCPROFCHANGED:
        case PGE_WINDOWDISPLAYCHANGED: {
            window = SDL_GetWindowFromID(event->window.windowID);
            break;
        }
        case SDL_TEXTEDITING: {
            window = SDL_GetWindowFromID(event->edit.windowID);
            break;
        }
        case SDL_TEXTINPUT: {
            window = SDL_GetWindowFromID(event->text.windowID);
            break;
        }
        case SDL_DROPBEGIN:
        case SDL_DROPCOMPLETE:
        case SDL_DROPTEXT:
        case SDL_DROPFILE: {
            window = SDL_GetWindowFromID(event->drop.windowID);
            break;
        }
        case SDL_KEYDOWN:
        case SDL_KEYUP: {
            window = SDL_GetWindowFromID(event->key.windowID);
            break;
        }
        case SDL_MOUSEWHEEL: {
            window = SDL_GetWindowFromID(event->wheel.windowID);
            break;
        }
        case SDL_MOUSEMOTION: {
            window = SDL_GetWindowFromID(event->motion.windowID);
            break;
        }
        case SDL_MOUSEBUTTONDOWN:
        case SDL_MOUSEBUTTONUP: {
            window = SDL_GetWindowFromID(event->button.windowID);
            break;
        }
#if SDL_VERSION_ATLEAST(2, 0, 14)
        case SDL_FINGERMOTION:
        case SDL_FINGERDOWN:
        case SDL_FINGERUP: {
            window = SDL_GetWindowFromID(event->tfinger.windowID);
            break;
        }
#endif
        default: {
            return dict;
        }
    }
    PyObject *pgWindow;
    if (!window || !(pgWindow = SDL_GetWindowData(window, "pg_window"))) {
        pgWindow = Py_None;
    }
    Py_INCREF(pgWindow);
    _pg_insobj(dict, "window", pgWindow);
    return dict;
}

/* event object internals */

static void
pg_event_dealloc(PyObject *self)
{
    pgEventObject *e = (pgEventObject *)self;
    Py_XDECREF(e->dict);
    PyObject_Free(self);
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
    return PyUnicode_FromFormat("<Event(%d-%s %S)>", e->type,
                                _pg_name_from_eventtype(e->type), e->dict);
}

static int
_pg_event_nonzero(pgEventObject *self)
{
    return self->type != SDL_NOEVENT;
}

static PyNumberMethods pg_event_as_number = {
    .nb_bool = (inquiry)_pg_event_nonzero,
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

static int
_pg_event_populate(pgEventObject *event, int type, PyObject *dict)
{
    event->type = _pg_pgevent_deproxify(type);
    if (!dict) {
        dict = PyDict_New();
        if (!dict) {
            PyErr_NoMemory();
            return -1;
        }
    }
    else {
        if (PyDict_GetItemString(dict, "type")) {
            PyErr_SetString(PyExc_ValueError,
                            "redundant type field in event dict");
            return -1;
        }
        Py_INCREF(dict);
    }
    event->dict = dict;
    return 0;
}

static int
pg_event_init(pgEventObject *self, PyObject *args, PyObject *kwargs)
{
    int type;
    PyObject *dict = NULL;

    if (!PyArg_ParseTuple(args, "i|O!", &type, &PyDict_Type, &dict)) {
        return -1;
    }

    if (!dict) {
        dict = PyDict_New();
        if (!dict) {
            PyErr_NoMemory();
            return -1;
        }
    }
    else {
        Py_INCREF(dict);
    }

    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyDict_SetItem(dict, key, value) < 0) {
                Py_DECREF(dict);
                return -1;
            }
        }
    }

    if (_pg_event_populate(self, type, dict) == -1) {
        return -1;
    }

    Py_DECREF(dict);
    return 0;
}

static PyTypeObject pgEvent_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pygame.event.Event",
    .tp_basicsize = sizeof(pgEventObject),
    .tp_dealloc = pg_event_dealloc,
    .tp_repr = pg_event_str,
    .tp_as_number = &pg_event_as_number,
#ifdef PYPY_VERSION
    .tp_getattro = pg_EventGetAttr,
    .tp_setattro = pg_EventSetAttr,
#else
    .tp_getattro = PyObject_GenericGetAttr,
    .tp_setattro = PyObject_GenericSetAttr,
#endif
    .tp_doc = DOC_PYGAMEEVENTEVENT,
    .tp_richcompare = pg_event_richcompare,
    .tp_members = pg_event_members,
    .tp_dictoffset = offsetof(pgEventObject, dict),
    .tp_init = (initproc)pg_event_init,
    .tp_new = PyType_GenericNew,
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
        PyObject_Free(e);
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

    if (_pg_event_populate(e, type, dict) == -1) {
        PyObject_Free(e);
        return NULL;
    }
    return (PyObject *)e;
}

/* event module functions */

static PyObject *
event_name(PyObject *self, PyObject *arg)
{
    int type;
    if (!PyArg_ParseTuple(arg, "i", &type))
        return NULL;

    return PyUnicode_FromString(_pg_name_from_eventtype(type));
}

static PyObject *
set_grab(PyObject *self, PyObject *arg)
{
    int doit = PyObject_IsTrue(arg);
    if (doit == -1)
        return NULL;

    VIDEO_INIT_CHECK();

    SDL_Window *win = pg_GetDefaultWindow();
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

    Py_RETURN_NONE;
}

static PyObject *
get_grab(PyObject *self, PyObject *_null)
{
    SDL_Window *win;
    SDL_bool mode = SDL_FALSE;

    VIDEO_INIT_CHECK();
    win = pg_GetDefaultWindow();
    if (win)
        mode = SDL_GetWindowGrab(win);
    return PyBool_FromLong(mode);
}

static void
_pg_event_pump(int dopump)
{
    if (dopump) {
        SDL_PumpEvents();
    }
    /* We need to translate WINDOWEVENTS. But if we do that from the
     * from event filter, internal SDL stuff that rely on WINDOWEVENT
     * might break. So after every event pump, we translate events from
     * here */
    SDL_FilterEvents(_pg_translate_windowevent, NULL);
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
pg_event_pump(PyObject *self, PyObject *_null)
{
    VIDEO_INIT_CHECK();
    _pg_event_pump(1);
    Py_RETURN_NONE;
}

static PyObject *
pg_event_poll(PyObject *self, PyObject *_null)
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
    static char *kwids[] = {"timeout", NULL};

    VIDEO_INIT_CHECK();

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", kwids, &timeout)) {
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
    int val = 0;
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
_pg_eventtype_as_seq(PyObject *obj, Py_ssize_t *len)
{
    *len = 1;
    if (PySequence_Check(obj)) {
        *len = PySequence_Size(obj);
        /* The returned object gets decref'd later, so incref now */
        Py_INCREF(obj);
        return obj;
    }
    else if (PyLong_Check(obj))
        return Py_BuildValue("(O)", obj);
    else
        return RAISE(PyExc_TypeError,
                     "event type must be numeric or a sequence");
}

static void
_pg_flush_events(Uint32 type)
{
    if (type == MAX_UINT32)
        SDL_FlushEvents(SDL_FIRSTEVENT, SDL_LASTEVENT);
    else {
        SDL_FlushEvent(type);
        SDL_FlushEvent(_pg_pgevent_proxify(type));
    }
}

static PyObject *
pg_event_clear(PyObject *self, PyObject *args, PyObject *kwargs)
{
    Py_ssize_t len;
    int loop, type;
    PyObject *seq, *obj = NULL;
    int dopump = 1;

    static char *kwids[] = {"eventtype", "pump", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Op", kwids, &obj,
                                     &dopump))
        return NULL;

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
_pg_get_all_events_except(PyObject *obj)
{
    SDL_Event event;
    Py_ssize_t len;
    int loop, type, ret;
    PyObject *seq, *list;

    SDL_Event *filtered_events;
    int filtered_index = 0;
    int filtered_events_len = 16;

    SDL_Event eventbuf[PG_GET_LIST_LEN];

    filtered_events = malloc(sizeof(SDL_Event) * filtered_events_len);
    if (!filtered_events)
        return PyErr_NoMemory();

    list = PyList_New(0);
    if (!list) {
        free(filtered_events);
        return PyErr_NoMemory();
    }

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
                if (filtered_index == filtered_events_len) {
                    SDL_Event *new_filtered_events =
                        malloc(sizeof(SDL_Event) * filtered_events_len * 4);
                    if (new_filtered_events == NULL) {
                        goto error;
                    }
                    memcpy(new_filtered_events, filtered_events,
                           sizeof(SDL_Event) * filtered_events_len);
                    filtered_events_len *= 4;
                    free(filtered_events);
                    filtered_events = new_filtered_events;
                }
                filtered_events[filtered_index] = event;
                filtered_index++;
            }
        } while (ret);
        do {
            ret = PG_PEEP_EVENT(&event, 1, SDL_GETEVENT,
                                _pg_pgevent_proxify(type));
            if (ret < 0) {
                PyErr_SetString(pgExc_SDLError, SDL_GetError());
                goto error;
            }
            else if (ret > 0) {
                if (filtered_index == filtered_events_len) {
                    SDL_Event *new_filtered_events =
                        malloc(sizeof(SDL_Event) * filtered_events_len * 4);
                    if (new_filtered_events == NULL) {
                        free(filtered_events);
                        goto error;
                    }
                    memcpy(new_filtered_events, filtered_events,
                           sizeof(SDL_Event) * filtered_events_len);
                    filtered_events_len *= 4;
                    free(filtered_events);
                    filtered_events = new_filtered_events;
                }
                filtered_events[filtered_index] = event;
                filtered_index++;
            }
        } while (ret);
    }

    do {
        len = PG_PEEP_EVENT_ALL(eventbuf, PG_GET_LIST_LEN, SDL_GETEVENT);
        if (len == -1) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            goto error;
        }

        for (loop = 0; loop < len; loop++) {
            if (!_pg_event_append_to_list(list, &eventbuf[loop]))
                goto error;
        }
    } while (len == PG_GET_LIST_LEN);

    PG_PEEP_EVENT_ALL(filtered_events, filtered_index, SDL_ADDEVENT);

    free(filtered_events);
    Py_DECREF(seq);
    return list;

error:
    /* While doing a goto here, PyErr must be set */
    free(filtered_events);
    Py_DECREF(list);
    Py_XDECREF(seq);
    return NULL;
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
    Py_ssize_t len;
    SDL_Event event;
    int loop, type, ret;
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
    PyObject *obj_evtype = NULL;
    PyObject *obj_exclude = NULL;
    int dopump = 1;

    static char *kwids[] = {"eventtype", "pump", "exclude", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OpO", kwids, &obj_evtype,
                                     &dopump, &obj_exclude))
        return NULL;

    VIDEO_INIT_CHECK();

    _pg_event_pump(dopump);

    if (obj_evtype == NULL || obj_evtype == Py_None) {
        if (obj_exclude != NULL && obj_exclude != Py_None) {
            return _pg_get_all_events_except(obj_exclude);
        }
        return _pg_get_all_events();
    }
    else {
        if (obj_exclude != NULL && obj_exclude != Py_None) {
            return RAISE(
                pgExc_SDLError,
                "Invalid combination of excluded and included event type");
        }
        return _pg_get_seq_events(obj_evtype);
    }
}

static PyObject *
pg_event_peek(PyObject *self, PyObject *args, PyObject *kwargs)
{
    SDL_Event event;
    Py_ssize_t len;
    int type, loop, res;
    PyObject *seq, *obj = NULL;
    int dopump = 1;

    static char *kwids[] = {"eventtype", "pump", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Op", kwids, &obj,
                                     &dopump))
        return NULL;

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
                Py_RETURN_TRUE;
            }
            res = PG_PEEP_EVENT(&event, 1, SDL_PEEKEVENT,
                                _pg_pgevent_proxify(type));
            if (res) {
                Py_DECREF(seq);

                if (res < 0)
                    return RAISE(pgExc_SDLError, SDL_GetError());
                Py_RETURN_TRUE;
            }
        }
        Py_DECREF(seq);
        Py_RETURN_FALSE; /* No event type match. */
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
    if (ret == 1)
        Py_RETURN_TRUE;
    else {
        Py_DECREF(e->dict);
        if (ret == 0)
            Py_RETURN_FALSE;
        else
            return RAISE(pgExc_SDLError, SDL_GetError());
    }
}

static PyObject *
pg_event_set_allowed(PyObject *self, PyObject *obj)
{
    Py_ssize_t len;
    int loop, type;
    PyObject *seq;
    VIDEO_INIT_CHECK();

    if (obj == Py_None) {
        int i;
        for (i = SDL_FIRSTEVENT; i < SDL_LASTEVENT; i++) {
            SDL_EventState(i, SDL_ENABLE);
        }
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
    Py_ssize_t len;
    int loop, type;
    PyObject *seq;
    VIDEO_INIT_CHECK();

    if (obj == Py_None) {
        int i;
        /* Start at PGPOST_EVENTBEGIN */
        for (i = PGPOST_EVENTBEGIN; i < SDL_LASTEVENT; i++) {
            SDL_EventState(i, SDL_IGNORE);
        }
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
    /* Never block SDL_WINDOWEVENT, we need them for translation */
    SDL_EventState(SDL_WINDOWEVENT, SDL_ENABLE);
    /* Never block PGE_KEYREPEAT too, its needed for pygame internal use */
    SDL_EventState(PGE_KEYREPEAT, SDL_ENABLE);
    Py_RETURN_NONE;
}

static PyObject *
pg_event_get_blocked(PyObject *self, PyObject *obj)
{
    Py_ssize_t len;
    int loop, type, isblocked = 0;
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
    return PyBool_FromLong(isblocked);
}

static PyObject *
pg_event_custom_type(PyObject *self, PyObject *_null)
{
    if (_custom_event < PG_NUMEVENTS)
        return PyLong_FromLong(_custom_event++);
    else
        return RAISE(pgExc_SDLError,
                     "pygame.event.custom_type made too many event types.");
}

static PyMethodDef _event_methods[] = {
    {"_internal_mod_init", (PyCFunction)pgEvent_AutoInit, METH_NOARGS,
     "auto initialize for event module"},
    {"_internal_mod_quit", (PyCFunction)pgEvent_AutoQuit, METH_NOARGS,
     "auto quit for event module"},

    {"event_name", event_name, METH_VARARGS, DOC_PYGAMEEVENTEVENTNAME},

    {"set_grab", set_grab, METH_O, DOC_PYGAMEEVENTSETGRAB},
    {"get_grab", (PyCFunction)get_grab, METH_NOARGS, DOC_PYGAMEEVENTGETGRAB},

    {"pump", (PyCFunction)pg_event_pump, METH_NOARGS, DOC_PYGAMEEVENTPUMP},
    {"wait", (PyCFunction)pg_event_wait, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEEVENTWAIT},
    {"poll", (PyCFunction)pg_event_poll, METH_NOARGS, DOC_PYGAMEEVENTPOLL},
    {"clear", (PyCFunction)pg_event_clear, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEEVENTCLEAR},
    {"get", (PyCFunction)pg_event_get, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEEVENTGET},
    {"peek", (PyCFunction)pg_event_peek, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEEVENTPEEK},
    {"post", (PyCFunction)pg_event_post, METH_O, DOC_PYGAMEEVENTPOST},

    {"set_allowed", (PyCFunction)pg_event_set_allowed, METH_O,
     DOC_PYGAMEEVENTSETALLOWED},
    {"set_blocked", (PyCFunction)pg_event_set_blocked, METH_O,
     DOC_PYGAMEEVENTSETBLOCKED},
    {"get_blocked", (PyCFunction)pg_event_get_blocked, METH_O,
     DOC_PYGAMEEVENTGETBLOCKED},
    {"custom_type", (PyCFunction)pg_event_custom_type, METH_NOARGS,
     DOC_PYGAMEEVENTCUSTOMTYPE},

    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(event)
{
    PyObject *module, *apiobj;
    static void *c_api[PYGAMEAPI_EVENT_NUMSLOTS];

    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "event",
                                         DOC_PYGAMEEVENT,
                                         -1,
                                         _event_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        return NULL;
    }

    /* type preparation */
    if (PyType_Ready(&pgEvent_Type) < 0) {
        return NULL;
    }

    /* create the module */
    module = PyModule_Create(&_module);
    if (!module) {
        return NULL;
    }

    joy_instance_map = PyDict_New();
    /* need to keep a reference for use in the module */
    Py_XINCREF(joy_instance_map);
    if (PyModule_AddObject(module, "_joy_instance_map", joy_instance_map)) {
        Py_XDECREF(joy_instance_map);
        Py_DECREF(module);
        return NULL;
    }

    Py_INCREF(&pgEvent_Type);
    if (PyModule_AddObject(module, "EventType", (PyObject *)&pgEvent_Type)) {
        Py_DECREF(&pgEvent_Type);
        Py_DECREF(module);
        return NULL;
    }
    Py_INCREF(&pgEvent_Type);
    if (PyModule_AddObject(module, "Event", (PyObject *)&pgEvent_Type)) {
        Py_DECREF(&pgEvent_Type);
        Py_DECREF(module);
        return NULL;
    }

    /* export the c api */
    assert(PYGAMEAPI_EVENT_NUMSLOTS == 6);
    c_api[0] = &pgEvent_Type;
    c_api[1] = pgEvent_New;
    c_api[2] = pgEvent_New2;
    c_api[3] = pgEvent_FillUserEvent;
    c_api[4] = pg_EnableKeyRepeat;
    c_api[5] = pg_GetKeyRepeat;

    apiobj = encapsulate_api(c_api, "event");
    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj)) {
        Py_XDECREF(apiobj);
        Py_DECREF(module);
        return NULL;
    }

    SDL_RegisterEvents(PG_NUMEVENTS - SDL_USEREVENT);
    return module;
}
