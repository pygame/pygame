# cython: language_level=2
#

from sdl2 cimport *

#https://wiki.libsdl.org/CategoryGameController

cdef extern from "SDL.h" nogil:
    ctypedef enum SDL_GameControllerAxis:
        SDL_CONTROLLER_AXIS_INVALID = -1,    
        SDL_CONTROLLER_AXIS_LEFTX,
        SDL_CONTROLLER_AXIS_LEFTY,
        SDL_CONTROLLER_AXIS_RIGHTX,
        SDL_CONTROLLER_AXIS_RIGHTY,
        SDL_CONTROLLER_AXIS_TRIGGERLEFT,
        SDL_CONTROLLER_AXIS_TRIGGERRIGHT,
        SDL_CONTROLLER_AXIS_MAX
    
    ctypedef enum SDL_GameControllerButton:
        SDL_CONTROLLER_BUTTON_INVALID = -1,
        SDL_CONTROLLER_BUTTON_A,
        SDL_CONTROLLER_BUTTON_B,
        SDL_CONTROLLER_BUTTON_X,
        SDL_CONTROLLER_BUTTON_Y,
        SDL_CONTROLLER_BUTTON_BACK,
        SDL_CONTROLLER_BUTTON_GUIDE,
        SDL_CONTROLLER_BUTTON_START,
        SDL_CONTROLLER_BUTTON_LEFTSTICK,
        SDL_CONTROLLER_BUTTON_RIGHTSTICK,
        SDL_CONTROLLER_BUTTON_LEFTSHOULDER,
        SDL_CONTROLLER_BUTTON_RIGHTSHOULDER,
        SDL_CONTROLLER_BUTTON_DPAD_UP,
        SDL_CONTROLLER_BUTTON_DPAD_DOWN,
        SDL_CONTROLLER_BUTTON_DPAD_LEFT,
        SDL_CONTROLLER_BUTTON_DPAD_RIGHT,
        SDL_CONTROLLER_BUTTON_MAX
 

    ctypedef struct SDL_GameController
    ctypedef enum SDL_GameControllerBindType:
        SDL_CONTROLLER_BINDTYPE_NONE = 0,
        SDL_CONTROLLER_BINDTYPE_BUTTON,
        SDL_CONTROLLER_BINDTYPE_AXIS,
        SDL_CONTROLLER_BINDTYPE_HAT

    ctypedef struct _hat:
        int hat
        int hat_mask
        
    cdef union _value:
        int button
        int axis

        _hat hat
            
    ctypedef struct SDL_GameControllerButtonBind:
        _value value
        SDL_GameControllerBindType bindType
        
    ctypedef struct SDL_Joystick
    ctypedef Sint32 SDL_JoystickID
    ctypedef struct SDL_JoystickGUID:
        Uint8 data[16]

    int SDL_GameControllerAddMapping(const char* mappingString)
    int SDL_GameControllerEventState(int state)
    int SDL_NumJoysticks()

    Uint8 SDL_GameControllerGetButton(SDL_GameController*      gamecontroller,
                                      SDL_GameControllerButton button)
                                      
    void SDL_GameControllerClose(SDL_GameController* gamecontroller)
    void SDL_GameControllerUpdate()
    void SDL_JoystickGetGUIDString(SDL_JoystickGUID guid,
                                   char*            pszGUID,
                                   int              cbGUID)
    SDL_GameController* SDL_GameControllerFromInstanceID(SDL_JoystickID joyid)
    SDL_GameController* SDL_GameControllerOpen(int joystick_index)
    SDL_bool SDL_GameControllerGetAttached(SDL_GameController* gamecontroller)
    Sint16 SDL_GameControllerGetAxis(SDL_GameController*    gamecontroller,
                                     SDL_GameControllerAxis axis)
    SDL_GameControllerButtonBind SDL_GameControllerGetBindForAxis(SDL_GameController*    gamecontroller,
                                                                  SDL_GameControllerAxis axis)
    SDL_GameControllerButtonBind SDL_GameControllerGetBindForButton(SDL_GameController*      gamecontroller,
                                                                    SDL_GameControllerButton button)

    SDL_Joystick* SDL_GameControllerGetJoystick(SDL_GameController* gamecontroller)
    const char* SDL_GameControllerGetStringForAxis(SDL_GameControllerAxis axis)
    const char* SDL_GameControllerGetStringForButton(SDL_GameControllerButton button)
    const char* SDL_GameControllerName(SDL_GameController* gamecontroller)
    const char* SDL_GameControllerNameForIndex(int joystick_index)
    char* SDL_GameControllerMapping(SDL_GameController* gamecontroller)
    char* SDL_GameControllerMappingForGUID(SDL_JoystickGUID guid)

    SDL_bool SDL_IsGameController(int joystick_index)
    SDL_JoystickGUID SDL_JoystickGetGUID(SDL_Joystick* joystick)
    

cdef class Controller:
    cdef SDL_GameController* _controller