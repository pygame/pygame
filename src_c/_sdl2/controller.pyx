from . import error

cdef extern from "../pygame.h" nogil:
    int pgJoystick_Check(object joy)
    object pgJoystick_New(int);
    void import_pygame_joystick()
    
import_pygame_joystick()

cdef extern from "SDL.h" nogil:
    void SDL_free(void *mem)

def set_eventstate(state):
    return SDL_GameControllerEventState(state)

def get_count():
    """ Returns the number of attached joysticks.
    """
    num = SDL_NumJoysticks()
    if num < 0:
        raise error()
    return num

def update():
    """ Will automatically called by the event loop,
        not necessary to call this function.
    """
    
    # https://wiki.libsdl.org/SDL_GameControllerUpdate
    
    SDL_GameControllerUpdate()

def is_controller(index):
    """ Check if the given joystick is supported by the game controller interface.

    :param int index: Index of the joystick.

    :return: 1 if supported, 0 if unsupported or invalid index.
    """
    # https://wiki.libsdl.org/SDL_IsGameController
    
    return SDL_IsGameController(index)

def name_forindex(index):
    """ Returns the name of controller,
        or NULL if there's no name or the index is invalid.
    """
    # https://wiki.libsdl.org/SDL_GameControllerNameForIndex
    return SDL_GameControllerNameForIndex(index)

cdef class Controller:
    def __init__(self, int index):
        """ Create a controller object and open it by given index.
        
        :param int index: Index of the joystick.
        """

        if not SDL_IsGameController(index):
            raise error('Index is invalid or not a supported joystick.')

        self._controller = SDL_GameControllerOpen(index)
        self._index = index
        if not self._controller:
            raise error('Could not open controller %d.' % index)

    """
    def __dealloc__(self):
        self.close()
    """
    
    def close(self):
        # https://wiki.libsdl.org/SDL_GameControllerClose
        if self._controller:
            SDL_GameControllerClose(self._controller)
            self._controller = NULL

    @staticmethod
    def from_joystick(joy):
        # https://wiki.libsdl.org/SDL_GameControllerFromInstanceID
        """ Create a controller object from pygame.joystick.Joystick object.
        
        """
        if not pgJoystick_Check(joy):
            raise TypeError('should be a pygame.joystick.Joystick object.')

        cdef Controller self = Controller.__new__(Controller)
        self.__init__(joy.get_id())
        return self
        
    @staticmethod
    def from_joystickid(SDL_JoystickID joyid):
        """ Create a controller object from JoystickID
        
        """
        # https://wiki.libsdl.org/SDL_GameControllerFromInstanceID
        cdef Controller self = Controller.__new__(Controller)
        self.__init__(joyid)
        return self
        
    @property
    def name(self):
        # https://wiki.libsdl.org/SDL_GameControllerName
        return SDL_GameControllerName(self._controller)
    
    def attached(self):
        # https://wiki.libsdl.org/SDL_GameControllerGetAttached
        return SDL_GameControllerGetAttached(self._controller)
        
    def as_joystick(self):
        # return a pygame.joystick.Joystick() object.
        joy = pgJoystick_New(self._index)
        return joy
        
    def get_axis(self, SDL_GameControllerAxis axis):
        # https://wiki.libsdl.org/SDL_GameControllerGetAxis
        return SDL_GameControllerGetAxis(self._controller, axis)
        
    def get_button(self, SDL_GameControllerButton button):
        # https://wiki.libsdl.org/SDL_GameControllerGetButton
        return SDL_GameControllerGetButton(self._controller, button)
        
    def get_mapping(self):
        # https://wiki.libsdl.org/SDL_GameControllerMapping
        # TODO: mapping should be a readable dict instead of a string.
        cdef char* mapping = SDL_GameControllerMapping(self._controller)
        
        try:
            #Convert the string to Python string?
            pyMapping = mapping.decode('UTF-8')
        except:
            #Returns bytes if decoding failed.
            pyMapping = mapping
        finally:
            SDL_free(mapping)
	        
        return pyMapping
        
    def add_mapping(self, mapping):
        # https://wiki.libsdl.org/SDL_GameControllerAddMapping
        # TODO: mapping should be a readable dict instead of a string.
        cdef SDL_Joystick *joy
        cdef SDL_JoystickGUID guid
        cdef char[64] pszGUID
        
        joy = SDL_GameControllerGetJoystick(self._controller)
        guid = SDL_JoystickGetGUID(joy)
        name = SDL_GameControllerName(self._controller)
        SDL_JoystickGetGUIDString(guid, pszGUID, 63)
        
        mapstring = "%s,%s,%s" % (pszGUID, name, mapping)
        res = SDL_GameControllerAddMapping(mapstring)
        if res < 0:
            raise()
        
        return res