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

#if IS_SDLv2
/*only register one block of user events.*/
static int have_registered_events = 0;

#define JOYEVENT_INSTANCE_ID "instance_id"
#define JOYEVENT_DEVICE_INDEX "device_index"
#else /* IS_SDLv1 */
#define JOYEVENT_INSTANCE_ID "joy"
#define JOYEVENT_DEVICE_INDEX "joy"
#endif /* IS_SDLv2 */

// The system message code is only tested on windows, so only
//   include it there for now.
#include <SDL_syswm.h>

/*this user event object is for safely passing
 *objects through the event queue.
 */

#define USEROBJECT_CHECK1 (Sint32)0xDEADBEEF
#define USEROBJECT_CHECK2 (Sint32)0xFEEDF00D

typedef struct UserEventObject {
    struct UserEventObject *next;
    PyObject *object;
} UserEventObject;

static UserEventObject *user_event_objects = NULL;

// Map joystick instance IDs to device ids for partial backwards compatibility
static PyObject *joy_instance_map = NULL;

#if IS_SDLv2
static int pg_key_repeat_delay = 0;
static int pg_key_repeat_interval = 0;

static SDL_TimerID _pg_repeat_timer = 0;
static SDL_Event _pg_repeat_event;
static SDL_bool  _pg_event_generate_videoresize = SDL_TRUE;

static Uint32
_pg_repeat_callback(Uint32 interval, void *param)
{
    _pg_repeat_event.type = PGE_KEYREPEAT;
    _pg_repeat_event.key.state = SDL_PRESSED;
    _pg_repeat_event.key.repeat = 1;
    SDL_PushEvent(&_pg_repeat_event);

    return pg_key_repeat_interval;
}

#endif /* IS_SLDv2 */
/* _custom_event stores the next custom user event type that will be returned
 * by pygame.event.custom_type(). It was supposed to start at PGE_USEREVENT
 * but because of a clash with libraries that just use pygame.USEREVENT
 * directly to be backward compatible with pygame 1.9.x, it was changed to
 * start at one higher.*/
#define _PGE_CUSTOM_EVENT_INIT PGE_USEREVENT + 1
static int _custom_event = _PGE_CUSTOM_EVENT_INIT;

static int _pg_event_is_init = 0;

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

#if IS_SDLv2
static char _pg_last_unicode_char[32] = { 0 };
static SDL_Event *_pg_last_keydown_event = NULL;

static int SDLCALL
_pg_remove_pending_PGS_VIDEORESIZE(void * userdata, SDL_Event *event)
{
    SDL_Event *new_event = (SDL_Event *)userdata;

    if (event->type == SDL_VIDEORESIZE &&
        event->window.windowID == new_event->window.windowID) {
        /* We're about to post a new size event, drop the old one */
        return 0;
    }
    return 1;
}

static int SDLCALL
_pg_remove_pending_PGS_VIDEOEXPOSE(void * userdata, SDL_Event *event)
{
    SDL_Event *new_event = (SDL_Event *)userdata;

    if (event->type == SDL_VIDEOEXPOSE &&
        event->window.windowID == new_event->window.windowID) {
        /* We're about to post a new videoexpose event, drop the old one */
        return 0;
    }
    return 1;
}

/*SDL 2 to SDL 1.2 event mapping and SDL 1.2 key repeat emulation*/
static int SDLCALL
pg_event_filter(void *_, SDL_Event *event)
{
    /* This event filter alters events inplace.
     */
    Uint32 type = event->type;

    if (type == SDL_WINDOWEVENT) {
        switch (event->window.event) {
            case SDL_WINDOWEVENT_SIZE_CHANGED:
                return 1;
            case SDL_WINDOWEVENT_RESIZED:
                if(_pg_event_generate_videoresize) {
                    /* keep resized event around for SDL_RendererEventWatch
                       (SDL2-internal) and pygame-internal event watch in
                       display.c */
                    SDL_Event newevent = *event;
                    newevent.type = SDL_VIDEORESIZE;

                    /* all previous resize events are superseded.
                       SDL2 already does this for SDL_WINDOWEVENT_RESIZED,
                       so we only need to filter our own custom event before
                       we push the new one*/
                    SDL_FilterEvents(_pg_remove_pending_PGS_VIDEORESIZE, &newevent);
                    SDL_PushEvent(&newevent);
                    return 1;
                }
                else {
                    return 1;
                }
            case SDL_WINDOWEVENT_EXPOSED:
                {
                    SDL_Event newevent = *event;
                    newevent.type = SDL_VIDEOEXPOSE;
                    
                    SDL_FilterEvents(_pg_remove_pending_PGS_VIDEOEXPOSE, &newevent);
                    SDL_PushEvent(&newevent);
                    return 1;
                }
            case SDL_WINDOWEVENT_ENTER:
            case SDL_WINDOWEVENT_LEAVE:
            case SDL_WINDOWEVENT_FOCUS_GAINED:
            case SDL_WINDOWEVENT_FOCUS_LOST:
            case SDL_WINDOWEVENT_MINIMIZED:
            case SDL_WINDOWEVENT_RESTORED:
                {
                    SDL_Event newevent = *event;
                    newevent.type = SDL_ACTIVEEVENT;
                    
                    SDL_PushEvent(&newevent);
                    return 1;
                }
            case SDL_WINDOWEVENT_CLOSE:
                break;
            default:
                /* DON'T ignore other SDL_WINDOWEVENTs for now.
                   If we delete events here, they won't be available to
                   low-level SDL2 either. For the python side, it's better
                   to omit events in pygame.event.get(). */
                return 1;
        }
    }
#pragma PG_WARN(Add event blocking here.)

    else if (type == SDL_KEYDOWN) {

        if (event->key.repeat) {
            return 0;
        }
        else if (pg_key_repeat_delay > 0) {
            if (_pg_repeat_timer) {
                SDL_RemoveTimer(_pg_repeat_timer);
            }
            memcpy(&_pg_repeat_event, event, sizeof(SDL_Event));
            _pg_repeat_timer = SDL_AddTimer(pg_key_repeat_delay, _pg_repeat_callback,
                                            NULL);
        }
        _pg_last_unicode_char[0] = 0;
        /* store the keydown event for later in the SDL_TEXTINPUT */
        _pg_last_keydown_event = event;
    }
    else if (type == SDL_TEXTINPUT) {
        if (_pg_last_keydown_event != NULL) {
            strncpy(_pg_last_unicode_char, event->text.text,
                    sizeof(_pg_last_unicode_char) -1);
            _pg_last_keydown_event = NULL;
        }
    }
    else if (type == SDL_KEYUP) {
        if (_pg_repeat_timer &&
            _pg_repeat_event.key.keysym.scancode == event->key.keysym.scancode) {
            SDL_RemoveTimer(_pg_repeat_timer);
            _pg_repeat_timer = 0;
        }
    }
    else if (type == PGE_KEYREPEAT) {
        event->type = SDL_KEYDOWN;
    }
    else if (type == SDL_MOUSEBUTTONDOWN || type == SDL_MOUSEBUTTONUP) {
        if (event->button.button & PGM_BUTTON_KEEP) {
            event->button.button ^= PGM_BUTTON_KEEP;
        }
        else if (event->button.button >= PGM_BUTTON_WHEELUP) {
            event->button.button += (PGM_BUTTON_X1 - PGM_BUTTON_WHEELUP);
        }
    }
    else if (type == SDL_MOUSEWHEEL) {
        SDL_Event newevent;
        int x, y;

        if (event->wheel.x == 0 && event->wheel.y == 0) {
            //#691 We are not moving wheel!
            return 1;
        }
        // Generate a MouseButtonDown event for compatibility.
        // https://wiki.libsdl.org/SDL_MouseWheelEvent
        newevent.type = SDL_MOUSEBUTTONDOWN;

        SDL_GetMouseState(&x, &y);
        newevent.button.x = x;
        newevent.button.y = y;

        newevent.button.state = SDL_PRESSED;
        newevent.button.clicks = 1;

        if (event->wheel.y != 0) {
            newevent.button.button = (event->wheel.y > 0) ?
                                     PGM_BUTTON_WHEELUP : PGM_BUTTON_WHEELDOWN;
        }
        else if (event->wheel.x != 0) {
            newevent.button.button = (event->wheel.x > 0) ?
                                     PGM_BUTTON_WHEELUP : PGM_BUTTON_WHEELDOWN;
        }
        newevent.button.button |= PGM_BUTTON_KEEP;

        /* this doesn't work! This is called by SDL, not Python:*/
        /*
          if (SDL_PushEvent(&newevent) < 0)
            return RAISE(pgExc_SDLError, SDL_GetError()), 0;
        */
        SDL_PushEvent(&newevent);
    }
    return 1;
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

/*must pass dictionary as this object*/
static UserEventObject *
_pg_user_event_addobject(PyObject *obj)
{
    UserEventObject *userobj = PyMem_New(UserEventObject, 1);
    if (!userobj)
        return NULL;

    Py_INCREF(obj);
    userobj->next = user_event_objects;
    userobj->object = obj;
    user_event_objects = userobj;

    return userobj;
}

/*note, we doublecheck to make sure the pointer is in our list,
 *not just some random pointer. this will keep us safe(r).
 */
static PyObject *
_pg_user_pg_event_getobject(UserEventObject *userobj)
{
    PyObject *obj = NULL;
    if (!user_event_objects) /*fail in most common case*/
        return NULL;
    if (user_event_objects == userobj) {
        obj = userobj->object;
        user_event_objects = userobj->next;
    }
    else {
        UserEventObject *hunt = user_event_objects;
        while (hunt && hunt->next != userobj)
            hunt = hunt->next;
        if (hunt) {
            hunt->next = userobj->next;
            obj = userobj->object;
        }
    }
    if (obj)
        PyMem_Del(userobj);
    return obj;
}

static void
_pg_user_event_cleanup(void)
{
    if (user_event_objects) {
        UserEventObject *hunt, *kill;
        hunt = user_event_objects;
        while (hunt) {
            kill = hunt;
            hunt = hunt->next;
            Py_DECREF(kill->object);
            PyMem_Del(kill);
        }
        user_event_objects = NULL;
    }
}

static int
pgEvent_FillUserEvent(pgEventObject *e, SDL_Event *event)
{
    UserEventObject *userobj = _pg_user_event_addobject(e->dict);
    if (!userobj)
        return -1;

    event->type = e->type;
    event->user.code = USEROBJECT_CHECK1;
    event->user.data1 = (void *)USEROBJECT_CHECK2;
    event->user.data2 = userobj;
    return 0;
}

static PyTypeObject pgEvent_Type;
static PyObject *
pgEvent_New(SDL_Event *);
static PyObject *
pgEvent_New2(int, PyObject *);
#define pgEvent_Check(x) ((x)->ob_type == &pgEvent_Type)

static char *
_pg_name_from_eventtype(int type)
{
    switch (type) {
        case SDL_ACTIVEEVENT:
            return "ActiveEvent";
#ifdef SDL2_AUDIODEVICE_SUPPORTED
        case SDL_AUDIODEVICEADDED:
            return "AudioDeviceAdded";
        case SDL_AUDIODEVICEREMOVED:
            return "AudioDeviceRemoved";
#endif /* SDL2_AUDIODEVICE_SUPPORTED */
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
        case SDL_WINDOWEVENT:
            return "WindowEvent";
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
#endif

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

#if IS_SDLv1

#if defined(Py_USING_UNICODE)

static PyObject *
_pg_our_unichr(long uni)
{
    static PyObject *bltin_unichr = NULL;

    if (bltin_unichr == NULL) {
        PyObject *bltins;

        bltins = PyImport_ImportModule(BUILTINS_MODULE);
        bltin_unichr = PyObject_GetAttrString(bltins, BUILTINS_UNICHR);
        Py_DECREF(bltins);
    }
    return PyEval_CallFunction(bltin_unichr, "(l)", uni);
}

static PyObject *
_pg_our_empty_ustr(void)
{
    static PyObject *empty_ustr = NULL;

    if (empty_ustr == NULL) {
        PyObject *bltins;
        PyObject *bltin_unicode;

        bltins = PyImport_ImportModule(BUILTINS_MODULE);
        bltin_unicode = PyObject_GetAttrString(bltins, BUILTINS_UNICODE);
        empty_ustr = PyEval_CallFunction(bltin_unicode, "(s)", "");
        Py_DECREF(bltin_unicode);
        Py_DECREF(bltins);
    }

    Py_INCREF(empty_ustr);

    return empty_ustr;
}

#else

static PyObject *
_pg_our_unichr(long uni)
{
    return PyInt_FromLong(uni);
}

static PyObject *
_pg_our_empty_ustr(void)
{
    return PyInt_FromLong(0);
}

#endif /* Py_USING_UNICODE */

#endif /* IS_SDLv1 */

static PyObject *
dict_from_event(SDL_Event *event)
{
    PyObject *dict = NULL, *tuple, *obj;
    int hx, hy;
#if IS_SDLv2
    long gain;
    long state;
#endif /* IS_SDLv2 */

    /*check if it is an event the user posted*/
    if (event->user.code == USEROBJECT_CHECK1 &&
        event->user.data1 == (void *)USEROBJECT_CHECK2) {
        dict = _pg_user_pg_event_getobject((UserEventObject *)event->user.data2);
        if (dict)
            return dict;
    }

    if (!(dict = PyDict_New()))
        return NULL;
    switch (event->type) {
#if IS_SDLv1
        case SDL_ACTIVEEVENT:
            _pg_insobj(dict, "gain", PyInt_FromLong(event->active.gain));
            _pg_insobj(dict, "state", PyInt_FromLong(event->active.state));
            break;
        case SDL_KEYDOWN:
            if (event->key.keysym.unicode)
                _pg_insobj(dict, "unicode", _pg_our_unichr(event->key.keysym.unicode));
            else
                _pg_insobj(dict, "unicode", _pg_our_empty_ustr());
        case SDL_KEYUP:
            _pg_insobj(dict, "key", PyInt_FromLong(event->key.keysym.sym));
            _pg_insobj(dict, "mod", PyInt_FromLong(event->key.keysym.mod));
            _pg_insobj(dict, "scancode",
                   PyInt_FromLong(event->key.keysym.scancode));
            break;
#else  /* IS_SDLv2 */
        case SDL_WINDOWEVENT:
            _pg_insobj(dict, "event", PyInt_FromLong(event->window.event));
            switch (event->window.event) {
                case SDL_WINDOWEVENT_CLOSE:
                    break;
            }
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
#ifdef SDL2_AUDIODEVICE_SUPPORTED
        case SDL_AUDIODEVICEADDED:
        case SDL_AUDIODEVICEREMOVED:
            _pg_insobj(dict, "which", PyInt_FromLong(event->adevice.which));
            _pg_insobj(dict, "iscapture", PyInt_FromLong(event->adevice.iscapture));
#endif /* SDL2_AUDIODEVICE_SUPPORTED */
            break;
        case SDL_KEYDOWN:
            _pg_insobj(dict, "unicode", Text_FromUTF8(_pg_last_unicode_char));
            /* fall through */
        case SDL_KEYUP:
            _pg_insobj(dict, "key", PyInt_FromLong(event->key.keysym.sym));
            _pg_insobj(dict, "mod", PyInt_FromLong(event->key.keysym.mod));
            _pg_insobj(dict, "scancode", PyInt_FromLong(event->key.keysym.scancode));
            break;
#endif /* IS_SDLv2 */
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


#if IS_SDLv1
        case SDL_VIDEORESIZE:
            obj = Py_BuildValue("(ii)", event->resize.w, event->resize.h);
            _pg_insobj(dict, "size", obj);
            _pg_insobj(dict, "w", PyInt_FromLong(event->resize.w));
            _pg_insobj(dict, "h", PyInt_FromLong(event->resize.h));
            break;
#else /* IS_SDLv2 */
        case SDL_VIDEORESIZE:
            obj = Py_BuildValue("(ii)", event->window.data1,
                                event->window.data2);
            _pg_insobj(dict, "size", obj);
            _pg_insobj(dict, "w", PyInt_FromLong(event->window.data1));
            _pg_insobj(dict, "h", PyInt_FromLong(event->window.data2));
            break;
#endif /* IS_SDLv2 */
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
            /* SDL_VIDEOEXPOSE and SDL_QUIT have no attributes */
    } /* switch (event->type) */
    if (event->type == PGE_USEREVENT && event->user.code == 0x1000) {
        _pg_insobj(dict, "filename", Text_FromUTF8(event->user.data1));
        free(event->user.data1);
        event->user.data1 = NULL;
    }
    if (event->type >= PGE_USEREVENT && event->type < PG_NUMEVENTS)
        _pg_insobj(dict, "code", PyInt_FromLong(event->user.code));

    switch (event->type) {
#if IS_SDLv2
        case SDL_WINDOWEVENT:
        case SDL_TEXTEDITING:
        case SDL_TEXTINPUT:
        case SDL_MOUSEWHEEL:
#endif /* IS_SDLv2 */
        case SDL_KEYDOWN:
        case SDL_KEYUP:
        case SDL_MOUSEMOTION:
        case SDL_MOUSEBUTTONDOWN:
        case SDL_MOUSEBUTTONUP:
        case SDL_USEREVENT:
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
    PyObject_DEL(self);
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
    sprintf(str, "<Event(%d-%s %s)>", e->type, _pg_name_from_eventtype(e->type),
            s);

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
    TYPE_HEAD(NULL, 0) "Event", /*name*/
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
    e = PyObject_NEW(pgEventObject, &pgEvent_Type);
    if (!e)
        return NULL;

    if (event) {
        e->type = event->type;
        e->dict = dict_from_event(event);
    }
    else {
        e->type = SDL_NOEVENT;
        e->dict = PyDict_New();
    }
    return (PyObject *)e;
}

static PyObject *
pgEvent_New2(int type, PyObject *dict)
{
    pgEventObject *e;
    e = PyObject_NEW(pgEventObject, &pgEvent_Type);
    if (e) {
        e->type = type;
        if (!dict)
            dict = PyDict_New();
        else
            Py_INCREF(dict);
        e->dict = dict;
    }
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

    if (!dict)
        dict = PyDict_New();
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
    if (!PyArg_ParseTuple(arg, "i", &doit))
        return NULL;
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
            else SDL_SetRelativeMouseMode(0);
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
get_grab(PyObject *self, PyObject *arg)
{
#if IS_SDLv1
    int mode;
#else  /* IS_SDLv2 */
    SDL_Window *win;
    SDL_bool mode = SDL_FALSE;
#endif /* IS_SDLv2 */

    VIDEO_INIT_CHECK();
#if IS_SDLv1
    mode = SDL_WM_GrabInput(SDL_GRAB_QUERY);
    return PyInt_FromLong(mode == SDL_GRAB_ON);
#else  /* IS_SDLv2 */
    win = pg_GetDefaultWindow();
    if (win)
        mode = SDL_GetWindowGrab(win);
    return PyInt_FromLong(mode);
#endif /* IS_SDLv2 */
}

static PyObject *
pg_event_pump(PyObject *self, PyObject *args)
{
    VIDEO_INIT_CHECK();
    SDL_PumpEvents();
    Py_RETURN_NONE;
}

static PyObject *
pg_event_wait(PyObject *self, PyObject *args, PyObject *kwargs)
{
    SDL_Event event;
    int status;
    int timeout = 0;
    static char *kwids[] = {
        "timeout",
        NULL
    };

    VIDEO_INIT_CHECK();
    
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", kwids, &timeout)) {
        return NULL;
    }

    #if IS_SDLv1
        if (timeout)
            return RAISE(PyExc_TypeError, "The timeout argument is unavailable in SDL1");
    #endif

    Py_BEGIN_ALLOW_THREADS;
    #if IS_SDLv1
        status = SDL_WaitEvent(&event);
    #else /* IS_SDLv2 */
        if (!timeout)
            status = SDL_WaitEvent(&event);
        else
            status = SDL_WaitEventTimeout(&event, timeout);
    #endif /* IS_SDLv2 */
    Py_END_ALLOW_THREADS;

    if (!status && !timeout) //status 0 means an error normally
        return RAISE(pgExc_SDLError, SDL_GetError());

    if (!status && timeout) //status 0 means WaitEventTimeout timed out
        return pgEvent_New(NULL);

    return pgEvent_New(&event);
}

static PyObject *
pg_event_poll(PyObject *self, PyObject *args)
{
    SDL_Event event;

    VIDEO_INIT_CHECK();

    if (SDL_PollEvent(&event))
        return pgEvent_New(&event);
    return pgEvent_New(NULL);
}

#if IS_SDLv1
static PyObject *
pg_event_clear(PyObject *self, PyObject *args, PyObject *kwargs)
{
    SDL_Event event;
    int mask = 0;
    int loop, num;
    PyObject *type = NULL;
    int dopump = 1;
    int val;

    static char *kwids[] = {
        "eventtype",
        "pump",
        NULL
    };

#if PY3
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Op", kwids,
                                     &type, &dopump))
        return NULL;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Oi", kwids,
                                     &type, &dopump))
        return NULL;
#endif

    VIDEO_INIT_CHECK();

    if (type == NULL || type == Py_None)
        mask = SDL_ALLEVENTS;
    else {
        if (PySequence_Check(type)) {
            num = PySequence_Size(type);
            for (loop = 0; loop < num; ++loop) {
                if (!pg_IntFromObjIndex(type, loop, &val))
                    return RAISE(
                        PyExc_TypeError,
                        "type sequence must contain valid event types");
                mask |= SDL_EVENTMASK(val);
            }
        }
        else if (pg_IntFromObj(type, &val))
            mask = SDL_EVENTMASK(val);
        else
            return RAISE(PyExc_TypeError,
                         "get type must be numeric or a sequence");
    }

    if (dopump)
        SDL_PumpEvents();

    while (SDL_PeepEvents(&event, 1, SDL_GETEVENT, mask) == 1)
    {
    }

    Py_RETURN_NONE;
}
#else /* IS_SDLv2 */
static PyObject *
pg_event_clear(PyObject *self, PyObject *args, PyObject *kwargs)
{
    Py_ssize_t num;
    int loop;
    PyObject *type = NULL;
    int dopump = 1;
    int val;

    static char *kwids[] = {
        "eventtype",
        "pump",
        NULL
    };

#if PY3
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Op", kwids,
                                     &type, &dopump))
        return NULL;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Oi", kwids,
                                     &type, &dopump))
        return NULL;
#endif

    VIDEO_INIT_CHECK();

    if (dopump)
        SDL_PumpEvents();

    if (type == NULL || type == Py_None) {
        SDL_FlushEvents(SDL_FIRSTEVENT, SDL_LASTEVENT);
    } else {
        if (PySequence_Check(type)) {
            num = PySequence_Size(type);
            for (loop = 0; loop < num; ++loop) {
                if (!pg_IntFromObjIndex(type, loop, &val))
                    return RAISE(
                        PyExc_TypeError,
                        "type sequence must contain valid event types");
                SDL_FlushEvent(val);
            }
        }
        else if (pg_IntFromObj(type, &val))
            SDL_FlushEvent(val);
        else
            return RAISE(PyExc_TypeError,
                         "get type must be numeric or a sequence");
    }

    Py_RETURN_NONE;
}
#endif /* IS_SDLv2 */

#if IS_SDLv1
static PyObject *
pg_event_get(PyObject *self, PyObject *args, PyObject *kwargs)
{
    SDL_Event event;
    int mask = 0;
    int loop, num;
    PyObject *type = NULL, *list, *e;
    int dopump = 1;
    int val;

    static char *kwids[] = {
        "eventtype",
        "pump",
        NULL
    };

#if PY3
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Op", kwids,
                                     &type, &dopump))
        return NULL;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Oi", kwids,
                                     &type, &dopump))
        return NULL;
#endif

    VIDEO_INIT_CHECK();

    if (type == NULL || type == Py_None)
        mask = SDL_ALLEVENTS;
    else {
        if (PySequence_Check(type)) {
            num = PySequence_Size(type);
            for (loop = 0; loop < num; ++loop) {
                if (!pg_IntFromObjIndex(type, loop, &val))
                    return RAISE(
                        PyExc_TypeError,
                        "type sequence must contain valid event types");
                mask |= SDL_EVENTMASK(val);
            }
        }
        else if (pg_IntFromObj(type, &val))
            mask = SDL_EVENTMASK(val);
        else
            return RAISE(PyExc_TypeError,
                         "eventtype must be numeric or a sequence");
    }

    list = PyList_New(0);
    if (!list)
        return NULL;

    if (dopump)
        SDL_PumpEvents();

    while (SDL_PeepEvents(&event, 1, SDL_GETEVENT, mask) == 1)
    {
        e = pgEvent_New(&event);
        if (!e) {
            Py_DECREF(list);
            return NULL;
        }

        if (0 != PyList_Append(list, e)) {
            Py_DECREF(list);
            Py_DECREF(e);
            return NULL; /* Exception already set. */
        }
        Py_DECREF(e);
    }
    return list;
}
#else /* IS_SDLv2 */
static PG_INLINE int
_pg_event_append_to_list(PyObject *list, SDL_Event *event)
{
    PyObject *e = pgEvent_New(event);
    if (!e) {
        Py_DECREF(list);
        return 0; /* Exception already set. */
    }
    if (0 != PyList_Append(list, e)) {
        Py_DECREF(e);
        Py_DECREF(list);
        return 0; /* Exception already set. */
    }
    Py_DECREF(e);
    return 1;
}

static PyObject *
pg_event_set_gen_videoresize(PyObject *self, PyObject *args)
{
    SDL_bool do_generate;

#if PY3
    if (!PyArg_ParseTuple(args, "p", &do_generate))
        return NULL;
#else
    if (!PyArg_ParseTuple(args, "i", &do_generate))
        return NULL;
#endif
    _pg_event_generate_videoresize=do_generate;

    if(do_generate) {
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}


static PyObject *
pg_event_get(PyObject *self, PyObject *args, PyObject *kwargs)
{
    SDL_Event event;
    Py_ssize_t num;
    int loop;
    PyObject *type = NULL, *list;
    int dopump = 1;
    int val, ret;

    static char *kwids[] = {
        "eventtype",
        "pump",
        NULL
    };

#if PY3
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Op", kwids,
                                     &type, &dopump))
        return NULL;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Oi", kwids,
                                     &type, &dopump))
        return NULL;
#endif

    VIDEO_INIT_CHECK();

    list = PyList_New(0);
    if (!list)
        return NULL;
    if (dopump)
        SDL_PumpEvents();

    if (type == NULL || type == Py_None) {
        while (SDL_PeepEvents(&event, 1, SDL_GETEVENT,
                              SDL_FIRSTEVENT, SDL_LASTEVENT) == 1) {
            if(!_pg_event_append_to_list(list, &event))
                return NULL;
        }
        return list;
    }

    if (PySequence_Check(type)) {
        num = PySequence_Size(type);
        for (loop = 0; loop < num; ++loop) {
            if (!pg_IntFromObjIndex(type, loop, &val)) {
                Py_DECREF(list);
                return RAISE(
                    PyExc_TypeError,
                    "type sequence must contain valid event types");
            }

            ret = SDL_PeepEvents(&event, 1, SDL_GETEVENT, val, val);

            if (ret < 0) {
                Py_DECREF(list);
                return RAISE(pgExc_SDLError, SDL_GetError());
            }
            else if (ret > 0) {
                if (!_pg_event_append_to_list(list, &event)) {
                    return NULL;
                }
            }
        }
    }
    else if (pg_IntFromObj(type, &val)) {
        ret = SDL_PeepEvents(&event, 1, SDL_GETEVENT, val, val);

        if (ret < 0) {
            Py_DECREF(list);
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
        else if (ret > 0) {
            if (!_pg_event_append_to_list(list, &event)) {
                return NULL;
            }
        }
    }
    else {
        Py_DECREF(list);
        return RAISE(PyExc_TypeError,
                     "get type must be numeric or a sequence");
    }
    return list;
}
#endif /* IS_SDLv2 */

#if IS_SDLv1
static PyObject *
pg_event_peek(PyObject *self, PyObject *args, PyObject *kwargs)
{
    SDL_Event event;
    int result;
    int mask = 0;
    int loop, num, noargs = 0;
    PyObject *type = NULL;
    int val;
    int dopump = 1;

    static char *kwids[] = {
        "eventtype",
        "pump",
        NULL
    };

#if PY3
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Op", kwids,
                                     &type, &dopump))
        return NULL;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Oi", kwids,
                                     &type, &dopump))
        return NULL;
#endif

    VIDEO_INIT_CHECK();

    if (!type || type == Py_None) {
        mask = SDL_ALLEVENTS;
        noargs = 1;
    }
    else {
        if (PySequence_Check(type)) {
            num = PySequence_Size(type);
            for (loop = 0; loop < num; ++loop) {
                if (!pg_IntFromObjIndex(type, loop, &val))
                    return RAISE(
                        PyExc_TypeError,
                        "type sequence must contain valid event types");
                mask |= SDL_EVENTMASK(val);
            }
        }
        else if (pg_IntFromObj(type, &val))
            mask = SDL_EVENTMASK(val);
        else
            return RAISE(PyExc_TypeError,
                         "peek type must be numeric or a sequence");
    }

    if (dopump)
        SDL_PumpEvents();
    result = SDL_PeepEvents(&event, 1, SDL_PEEKEVENT, mask);
    if (result < 0)
        return RAISE(pgExc_SDLError, SDL_GetError());

    if (noargs)
        return pgEvent_New(result ? &event : NULL);
    return PyInt_FromLong(result == 1);
}
#else /* IS_SDLv2 */
static PyObject *
pg_event_peek(PyObject *self, PyObject *args, PyObject *kwargs)
{
    SDL_Event event;
    Py_ssize_t num;
    int result;
    int loop;
    PyObject *type = NULL;
    int val;
    int dopump = 1;

    static char *kwids[] = {
        "eventtype",
        "pump",
        NULL
    };

#if PY3
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Op", kwids,
                                     &type, &dopump))
        return NULL;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Oi", kwids,
                                     &type, &dopump))
        return NULL;
#endif

    VIDEO_INIT_CHECK();

    if (dopump)
        SDL_PumpEvents();

    if (type == NULL || type == Py_None) {
        result = SDL_PeepEvents(&event, 1, SDL_PEEKEVENT, SDL_FIRSTEVENT, SDL_LASTEVENT);
        if (result < 0)
            return RAISE(pgExc_SDLError, SDL_GetError());
        return pgEvent_New(result ? &event : NULL);
    }

    if (PySequence_Check(type)) {
        num = PySequence_Size(type);
        for (loop = 0; loop < num; ++loop) {
            if (!pg_IntFromObjIndex(type, loop, &val))
                return RAISE(
                    PyExc_TypeError,
                    "type sequence must contain valid event types");
            result = SDL_PeepEvents(&event, 1, SDL_PEEKEVENT, val, val);
            if (result < 0) {
                return RAISE(pgExc_SDLError, SDL_GetError());
            } else if (result == 1) {
                return PyInt_FromLong(1);
            }
        }

        return PyInt_FromLong(0); /* No event type match. */
    }
    else if (pg_IntFromObj(type, &val)) {
        result = SDL_PeepEvents(&event, 1, SDL_PEEKEVENT, val, val);
        if (result < 0)
            return RAISE(pgExc_SDLError, SDL_GetError());
        return PyInt_FromLong(result == 1);
    }
    return RAISE(PyExc_TypeError,
                 "peek type must be numeric or a sequence");
}
#endif /* IS_SDLv2 */

static PyObject *
pg_event_post(PyObject *self, PyObject *args)
{
    pgEventObject *e;
    SDL_Event event;
    int isblocked = 0;

    if (!PyArg_ParseTuple(args, "O!", &pgEvent_Type, &e))
        return NULL;

    VIDEO_INIT_CHECK();

    /* see if the event is blocked before posting it. */
    isblocked = SDL_EventState(e->type, SDL_QUERY) == SDL_IGNORE;

    if (isblocked) {
        /* event is blocked, so we do not post it. */
        Py_RETURN_NONE;
    }

    if (e->type == SDL_KEYDOWN || e->type == SDL_KEYUP){
        PyObject *event_key      = PyDict_GetItemString(e->dict, "key");
        PyObject *event_scancode = PyDict_GetItemString(e->dict, "scancode");
        PyObject *event_mod      = PyDict_GetItemString(e->dict, "mod");
#if IS_SDLv1
        PyObject *event_unicode  = PyDict_GetItemString(e->dict, "unicode");
#else  /* IS_SDLv2 */
        PyObject *event_window_ID= PyDict_GetItemString(e->dict, "window");
#endif /* IS_SDLv2 */
        event.type =  e->type;

        if (event_key == NULL){
            return RAISE(pgExc_SDLError, "key event posted without keycode");
        }
        if (!PyInt_Check(event_key)){
            return RAISE(pgExc_SDLError, "posted event keycode must be int");
        }
        event.key.keysym.sym = PyLong_AsLong(event_key);

        if (event_scancode != NULL){
            if (!PyInt_Check(event_scancode)){
                return RAISE(pgExc_SDLError, "posted event scancode must be int");
            }
            event.key.keysym.scancode = PyLong_AsLong(event_scancode);
        }

        if (event_mod != NULL && event_mod != Py_None){
            if (!PyInt_Check(event_scancode)){
                return RAISE(pgExc_SDLError, "posted event modifiers must be int");
            }
            if (PyLong_AsLong(event_mod) > 65535 || PyLong_AsLong(event_mod) < 0) {
                return RAISE(pgExc_SDLError, "mods must be 16-bit int");
            }
            event.key.keysym.mod = (Uint16) PyLong_AsLong(event_mod);
        }

#if IS_SDLv1
        /*ignore unicode property*/
#else  /* IS_SDLv2 */
        if (event_window_ID != NULL && event_window_ID != Py_None){
            if (!PyInt_Check(event_window_ID)){
                return RAISE(pgExc_SDLError, "posted event window id must be int");
            }
            event.key.windowID = PyLong_AsLong(event_window_ID);
        }
#endif /* IS_SDLv2 */
    }
    else if (e->type >= PGE_USEREVENT && e->type < PG_NUMEVENTS) {
        if (pgEvent_FillUserEvent(e, &event))
            return NULL;
    }
    else {
        /* HACK:
           A non-USEREVENT type is treated like a USEREVENT union in the SDL2
           event queue. This needs to be decoded again. */
         if (pgEvent_FillUserEvent(e, &event))
            return NULL;
    }
#if IS_SDLv1
    if (SDL_PushEvent(&event) == -1)
#else  /* IS_SDLv2 */
    if (SDL_PushEvent(&event) < 0)
#endif /* IS_SDLv2 */
        return RAISE(pgExc_SDLError, SDL_GetError());

    Py_RETURN_NONE;
}

static int
_pg_check_event_in_range(int evt)
{
// #if IS_SDLv1
//     return evt >= 0 && evt < PG_NUMEVENTS;
// #else /* IS_SDLv2 */
//     return evt >= 0 && evt < PGE_EVENTEND; /* needed for extras */
// #endif /* IS_S*DLv2 */
    return evt >= 0 && evt < PG_NUMEVENTS;
}

static PyObject *
pg_event_set_allowed(PyObject *self, PyObject *args)
{
    PyObject *type;
    int val;

    if (PyTuple_Size(args) != 1)
        return RAISE(PyExc_ValueError, "set_allowed requires 1 argument");

    VIDEO_INIT_CHECK();

    type = PyTuple_GET_ITEM(args, 0);
    if (PySequence_Check(type)) {
        Py_ssize_t num = PySequence_Length(type);
        int loop;

        for (loop = 0; loop < num; ++loop) {
            if (!pg_IntFromObjIndex(type, loop, &val))
                return RAISE(PyExc_TypeError,
                             "type sequence must contain valid event types");
            if (!_pg_check_event_in_range(val))
                return RAISE(PyExc_ValueError, "Invalid event in sequence");
            SDL_EventState(val, SDL_ENABLE);
        }
    }
    else if (type == Py_None) {
#if IS_SDLv2
        int i;
        for (i=SDL_FIRSTEVENT; i<SDL_LASTEVENT; i++) {
            SDL_EventState(i, SDL_ENABLE);
        }
#else
        SDL_EventState(0xFF, SDL_ENABLE);
#endif /* IS_SDLv2 */
    } else if (pg_IntFromObj(type, &val)) {
        if (!_pg_check_event_in_range(val))
            return RAISE(PyExc_ValueError, "Invalid event");
        SDL_EventState(val, SDL_ENABLE);
    }
    else
        return RAISE(PyExc_TypeError, "type must be numeric or a sequence");

    Py_RETURN_NONE;
}

static PyObject *
pg_event_set_blocked(PyObject *self, PyObject *args)
{
    PyObject *type;
    int val;

    if (PyTuple_Size(args) != 1)
        return RAISE(PyExc_ValueError, "set_blocked requires 1 argument");

    VIDEO_INIT_CHECK();

    type = PyTuple_GET_ITEM(args, 0);
    if (PySequence_Check(type)) {
        Py_ssize_t num = PySequence_Length(type);
        int loop;

        for (loop = 0; loop < num; ++loop) {
            if (!pg_IntFromObjIndex(type, loop, &val))
                return RAISE(PyExc_TypeError,
                             "type sequence must contain valid event types");
            if (!_pg_check_event_in_range(val))
                return RAISE(PyExc_ValueError, "Invalid event in sequence");
            SDL_EventState(val, SDL_IGNORE);
        }
    }
    else if (type == Py_None) {
#if IS_SDLv2
        int i;
        for (i=SDL_FIRSTEVENT; i<SDL_LASTEVENT; i++) {
            SDL_EventState(i, SDL_IGNORE);
        }
#else
        SDL_EventState(0xFF, SDL_IGNORE);
#endif /* IS_SDLv2 */
    } else if (pg_IntFromObj(type, &val)) {
        if (!_pg_check_event_in_range(val))
            return RAISE(PyExc_ValueError, "Invalid event");
        SDL_EventState(val, SDL_IGNORE);
    }
    else
        return RAISE(PyExc_TypeError, "type must be numeric or a sequence");

    Py_RETURN_NONE;
}

static PyObject *
pg_event_get_blocked(PyObject *self, PyObject *args)
{
    Py_ssize_t num;
    int loop;
    PyObject *type;
    int val;
    int isblocked = 0;

    if (PyTuple_Size(args) != 1)
        return RAISE(PyExc_ValueError, "get_blocked requires 1 argument");

    VIDEO_INIT_CHECK();

    type = PyTuple_GET_ITEM(args, 0);
    if (PySequence_Check(type)) {
        num = PySequence_Length(type);
        for (loop = 0; loop < num; ++loop) {
            if (!pg_IntFromObjIndex(type, loop, &val))
                return RAISE(PyExc_TypeError,
                             "type sequence must contain valid event types");
            if (!_pg_check_event_in_range(val))
                return RAISE(PyExc_ValueError, "Invalid event in sequence");
            isblocked |= SDL_EventState(val, SDL_QUERY) == SDL_IGNORE;
        }
    }
    else if (pg_IntFromObj(type, &val)) {
        if (!_pg_check_event_in_range(val))
            return RAISE(PyExc_ValueError, "Invalid event");
        isblocked = SDL_EventState(val, SDL_QUERY) == SDL_IGNORE;
    }
    else
        return RAISE(PyExc_TypeError, "type must be numeric or a sequence");

    return PyInt_FromLong(isblocked);
}


static PyObject *
pg_event_custom_type(PyObject *self, PyObject *args)
{
    if (_custom_event < PG_NUMEVENTS) {
        return PyInt_FromLong(_custom_event++);
    }
    else
        return RAISE(pgExc_SDLError, "pygame.event.custom_type made too many event types.");
}

static PyMethodDef _event_methods[] = {
    {"__PYGAMEinit__", pgEvent_AutoInit, METH_NOARGS,
     "auto initialize for event module"},
#if IS_SDLv2
    {"_set_gen_videoresize", pg_event_set_gen_videoresize, METH_VARARGS, "enable or disable legacy VIDEORESIZE events"},
#endif /* IS_SDLv2 */

    {"Event", (PyCFunction)pg_Event, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEEVENTEVENT},
    {"event_name", event_name, METH_VARARGS, DOC_PYGAMEEVENTEVENTNAME},

    {"set_grab", set_grab, METH_VARARGS, DOC_PYGAMEEVENTSETGRAB},
    {"get_grab", get_grab, METH_NOARGS, DOC_PYGAMEEVENTGETGRAB},

    {"pump", pg_event_pump, METH_NOARGS, DOC_PYGAMEEVENTPUMP},
    {"wait", (PyCFunction)pg_event_wait, METH_VARARGS | METH_KEYWORDS, DOC_PYGAMEEVENTWAIT},
    {"poll", pg_event_poll, METH_NOARGS, DOC_PYGAMEEVENTPOLL},
    {"clear", (PyCFunction)pg_event_clear, METH_VARARGS | METH_KEYWORDS, DOC_PYGAMEEVENTCLEAR},
    {"get", (PyCFunction)pg_event_get, METH_VARARGS | METH_KEYWORDS, DOC_PYGAMEEVENTGET},
    {"peek", (PyCFunction)pg_event_peek, METH_VARARGS | METH_KEYWORDS, DOC_PYGAMEEVENTPEEK},
    {"post", pg_event_post, METH_VARARGS, DOC_PYGAMEEVENTPOST},

    {"set_allowed", pg_event_set_allowed, METH_VARARGS, DOC_PYGAMEEVENTSETALLOWED},
    {"set_blocked", pg_event_set_blocked, METH_VARARGS, DOC_PYGAMEEVENTSETBLOCKED},
    {"get_blocked", pg_event_get_blocked, METH_VARARGS, DOC_PYGAMEEVENTGETBLOCKED},
    {"custom_type", pg_event_custom_type, METH_NOARGS, DOC_PYGAMEEVENTCUSTOMTYPE},


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

        if (user_event == (Uint32)-1) {
            PyErr_SetString(pgExc_SDLError, "unable to register user events");
            DECREF_MOD(module);
            MODINIT_ERROR;
        }
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

    /* Assume if there are events in the user events list
     * there is also a registered cleanup callback for them.
     */
    if (user_event_objects == NULL) {
        pg_RegisterQuit(_pg_user_event_cleanup);
    }

    MODINIT_RETURN(module);
}
