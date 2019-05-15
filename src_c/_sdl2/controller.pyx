from . import error

cdef extern from "../pygame.h" nogil:
    int pgJoystick_Check(object joy)
    object pgJoystick_New(int);
    void import_pygame_joystick()
    void pg_RegisterQuit(object)
    void JOYSTICK_INIT_CHECK()

cdef extern from "SDL.h" nogil:
    void SDL_free(void *mem)
 
import_pygame_joystick()

def GAMECONTROLLER_INIT_CHECK():
    if not SDL_WasInit(_SDL_INIT_GAMECONTROLLER):
        raise error("gamecontroller system not initialized")
    
cdef bint _controller_autoinit():
    if not SDL_WasInit(_SDL_INIT_GAMECONTROLLER):
        if SDL_InitSubSystem(_SDL_INIT_GAMECONTROLLER):
            return False
        #pg_RegisterQuit(_controller_autoquit)

    return True
    
cdef void _controller_autoquit():
    cdef Controller controller
    for c in Controller._controllers:
        controller = c
        controller.quit()
        controller._controller = NULL
        
    Controller._controllers.clear()
    
    if SDL_WasInit(_SDL_INIT_GAMECONTROLLER):
        SDL_QuitSubSystem(_SDL_INIT_GAMECONTROLLER)

    
# not automatically initialize controller at this moment.

def __PYGAMEinit__(**kwargs):
    _controller_autoinit()

def init():
    if not _controller_autoinit():
        raise error()

def get_init():
    return not SDL_WasInit(_SDL_INIT_GAMECONTROLLER) == 0

def quit():
    if SDL_WasInit(_SDL_INIT_GAMECONTROLLER):
        SDL_QuitSubSystem(_SDL_INIT_GAMECONTROLLER)
        
def set_eventstate(state):
    GAMECONTROLLER_INIT_CHECK()
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
    GAMECONTROLLER_INIT_CHECK()
    SDL_GameControllerUpdate()

def is_controller(index):
    """ Check if the given joystick is supported by the game controller interface.

    :param int index: Index of the joystick.

    :return: 1 if supported, 0 if unsupported or invalid index.
    """
    GAMECONTROLLER_INIT_CHECK()
    return SDL_IsGameController(index)

def name_forindex(index):
    """ Returns the name of controller,
        or NULL if there's no name or the index is invalid.
    """
    GAMECONTROLLER_INIT_CHECK()
    return SDL_GameControllerNameForIndex(index)

cdef class Controller:
    _controllers = []
        
    def __init__(self, int index):
        """ Create a controller object and open it by given index.
        
        :param int index: Index of the joystick.
        """
        GAMECONTROLLER_INIT_CHECK()
        if not SDL_IsGameController(index):
            raise error('Index is invalid or not a supported joystick.')

        self._controller = SDL_GameControllerOpen(index)
        self._index = index
        if not self._controller:
            raise error('Could not open controller %d.' % index)
        
        Controller._controllers.append(self)
        
    def __dealloc__(self):
        Controller._controllers.remove(self)
        self.quit()
    
    def _CLOSEDCHECK(self):
        if not self._controller:
            raise error('called on a closed controller')
    def init(self):
        self.__init__(self._index)

    def get_init(self):
        return not self._controller == NULL
        
    def quit(self):
        if self._controller:
            SDL_GameControllerClose(self._controller)
            self._controller = NULL
            
    @staticmethod
    def from_joystick(joy):
        """ Create a controller object from pygame.joystick.Joystick object.
        
        """
        # https://wiki.libsdl.org/SDL_GameControllerFromInstanceID
        JOYSTICK_INIT_CHECK()
        if not pgJoystick_Check(joy):
            raise TypeError('should be a pygame.joystick.Joystick object.')
    
        cdef Controller self = Controller.__new__(Controller)
        self.__init__(joy.get_id())
        return self

    @property
    def id(self):
        return self._index
        
    @property
    def name(self):
        # https://wiki.libsdl.org/SDL_GameControllerName
        GAMECONTROLLER_INIT_CHECK()
        self._CLOSEDCHECK()
        return SDL_GameControllerName(self._controller)
            
    def attached(self):
        # https://wiki.libsdl.org/SDL_GameControllerGetAttached
        GAMECONTROLLER_INIT_CHECK()
        self._CLOSEDCHECK()
        return SDL_GameControllerGetAttached(self._controller)
        
    def as_joystick(self):
        # create a pygame.joystick.Joystick() object by using index.
        JOYSTICK_INIT_CHECK()
        GAMECONTROLLER_INIT_CHECK()
        joy = pgJoystick_New(self._index)
        return joy
        
    def get_axis(self, SDL_GameControllerAxis axis):
        # https://wiki.libsdl.org/SDL_GameControllerGetAxis
        GAMECONTROLLER_INIT_CHECK()
        self._CLOSEDCHECK()
        return SDL_GameControllerGetAxis(self._controller, axis)
        
    def get_button(self, SDL_GameControllerButton button):
        # https://wiki.libsdl.org/SDL_GameControllerGetButton
        GAMECONTROLLER_INIT_CHECK()
        self._CLOSEDCHECK()
        return SDL_GameControllerGetButton(self._controller, button)
        
    def get_mapping(self):
        #https://wiki.libsdl.org/SDL_GameControllerMapping
        # TODO: mapping should be a readable dict instead of a string.
        GAMECONTROLLER_INIT_CHECK()
        self._CLOSEDCHECK()
        mapping = SDL_GameControllerMapping(self._controller)
        SDL_free(mapping)
        return mapping
        
    def add_mapping(self, mapping):
        # https://wiki.libsdl.org/SDL_GameControllerAddMapping
        # TODO: mapping should be a readable dict instead of a string.
        GAMECONTROLLER_INIT_CHECK()
        self._CLOSEDCHECK()
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
            raise error()
        
        return res