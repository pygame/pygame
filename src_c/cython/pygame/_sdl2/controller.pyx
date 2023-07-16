from pygame._sdl2.sdl2 import error

cdef extern from "../pygame.h" nogil:
    int pgJoystick_Check(object joy)
    object pgJoystick_New(int);
    void import_pygame_joystick()
    void pg_RegisterQuit(object)
    void JOYSTICK_INIT_CHECK()

cdef extern from "SDL.h" nogil:
    void SDL_free(void *mem)
    int SDL_VERSION_ATLEAST(int major, int minor, int patch) 

import_pygame_joystick()

def _gamecontroller_init_check():
    if not SDL_WasInit(_SDL_INIT_GAMECONTROLLER):
        raise error("gamecontroller system not initialized")

cdef bint _controller_autoinit() noexcept:
    if not SDL_WasInit(_SDL_INIT_GAMECONTROLLER):
        if SDL_InitSubSystem(_SDL_INIT_GAMECONTROLLER):
            return False
        #pg_RegisterQuit(_controller_autoquit)
    return True

cdef void _controller_autoquit() noexcept:
    cdef Controller controller
    for c in Controller._controllers:
        controller = c
        controller.quit()
        controller._controller = NULL

    Controller._controllers.clear()

    if SDL_WasInit(_SDL_INIT_GAMECONTROLLER):
        SDL_QuitSubSystem(_SDL_INIT_GAMECONTROLLER)

# not automatically initialize controller at this moment.

def _internal_mod_init(**kwargs):
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
    _gamecontroller_init_check()
    SDL_GameControllerEventState(int(state))

def get_eventstate():
    _gamecontroller_init_check()
    return SDL_GameControllerEventState(-1) == 1

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
    _gamecontroller_init_check()
    SDL_GameControllerUpdate()

def is_controller(index):
    """ Check if the given joystick is supported by the game controller interface.

    :param int index: Index of the joystick.

    :return: 1 if supported, 0 if unsupported or invalid index.
    """
    _gamecontroller_init_check()
    return SDL_IsGameController(index) == 1

def name_forindex(index):
    """ Returns the name of controller,
        or NULL if there's no name or the index is invalid.
    """
    _gamecontroller_init_check()
    max_controllers = SDL_NumJoysticks()
    if max_controllers < 0:
        raise error()

    if 0 <= index < max_controllers:
        return SDL_GameControllerNameForIndex(index).decode('utf-8')

    return None

cdef class Controller:
    _controllers = []

    def __init__(self, int index):
        """ Create a controller object and open it by given index.

        :param int index: Index of the joystick.
        """
        _gamecontroller_init_check()
        if not SDL_IsGameController(index):
            raise error('Index is invalid or not a supported joystick.')

        self._controller = SDL_GameControllerOpen(index)
        self._index = index
        if not self._controller:
            raise error('Could not open controller %d.' % index)

        Controller._controllers.append(self)

    def __dealloc__(self):
        try:
            Controller._controllers.remove(self)
        except ValueError:
            pass # Controller is not in list.

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
        _gamecontroller_init_check()
        self._CLOSEDCHECK()
        return SDL_GameControllerName(self._controller).decode('utf-8')

    def attached(self):
        # https://wiki.libsdl.org/SDL_GameControllerGetAttached
        _gamecontroller_init_check()
        self._CLOSEDCHECK()
        return SDL_GameControllerGetAttached(self._controller)

    def as_joystick(self):
        # create a pygame.joystick.Joystick() object by using index.
        JOYSTICK_INIT_CHECK()
        _gamecontroller_init_check()
        joy = pgJoystick_New(self._index)
        return joy

    def get_axis(self, SDL_GameControllerAxis axis):
        # https://wiki.libsdl.org/SDL_GameControllerGetAxis
        _gamecontroller_init_check()
        self._CLOSEDCHECK()
        return SDL_GameControllerGetAxis(self._controller, axis)

    def get_button(self, SDL_GameControllerButton button):
        # https://wiki.libsdl.org/SDL_GameControllerGetButton
        _gamecontroller_init_check()
        self._CLOSEDCHECK()
        return SDL_GameControllerGetButton(self._controller, button) == 1

    def get_mapping(self):
        #https://wiki.libsdl.org/SDL_GameControllerMapping
        # TODO: mapping should be a readable dict instead of a string.
        _gamecontroller_init_check()
        self._CLOSEDCHECK()
        raw_mapping = SDL_GameControllerMapping(self._controller)
        mapping = raw_mapping.decode('utf-8')
        SDL_free(raw_mapping)

        # split mapping, cut off guid, name and last (empty) comma
        mapping = mapping.split(",")[2:-1]
        keys = []
        values = []

        for obj in mapping:
            a = obj.split(':')
            keys.append(a[0])
            values.append(a[1])

        #create and return the dict
        mapping = dict(zip(keys, values))
        return mapping

    def set_mapping(self, mapping):
        # https://wiki.libsdl.org/SDL_GameControllerAddMapping
        # TODO: mapping should be a readable dict instead of a string.
        _gamecontroller_init_check()
        self._CLOSEDCHECK()
        cdef SDL_Joystick *joy
        cdef SDL_JoystickGUID guid
        cdef char[64] pszGUID

        joy = SDL_GameControllerGetJoystick(self._controller)
        guid = SDL_JoystickGetGUID(joy)
        name = SDL_GameControllerName(self._controller)
        SDL_JoystickGetGUIDString(guid, pszGUID, 63)

        str_map = ""
        for key, value in mapping.items():
            str_map += "{}:{},".format(key, value)

        mapstring = b"%s,%s,%s" % (pszGUID, name, str_map.encode('utf-8'))
        res = SDL_GameControllerAddMapping(mapstring)
        if res < 0:
            raise error()

        return res

    def rumble(self, low_frequency, high_frequency, duration):
        """
        Play a rumble effect on the controller, with set power (0-1 range) and
        duration (in ms). Returns True if the effect was played successfully,
        False otherwise.
        """
        _gamecontroller_init_check()
        self._CLOSEDCHECK()
        
        duration = max(duration, 0)
        low = min(max(low_frequency, 0.0), 1.0)
        high = min(max(high_frequency, 0.0), 1.0)

        return not PG_GameControllerRumble(
            self._controller, low * 0xFFFF, high * 0xFFFF, duration
        )

    def stop_rumble(self):
        """
        Stop any rumble effect playing on the controller.
        """
        _gamecontroller_init_check()
        self._CLOSEDCHECK()
        PG_GameControllerRumble(self._controller, 0, 0, 1)
