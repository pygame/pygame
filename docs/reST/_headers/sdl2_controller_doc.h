/* Auto generated file: with makeref.py .  Docs go in docs/reST/ref/ . */
#define DOC_PYGAMESDL2CONTROLLER "Pygame module to work with controllers."
#define DOC_PYGAMESDL2CONTROLLERINIT "init() -> None\ninitialize the controller module"
#define DOC_PYGAMESDL2CONTROLLERQUIT "quit() -> None\nUninitialize the controller module."
#define DOC_PYGAMESDL2CONTROLLERGETINIT "get_init() -> bool\nReturns True if the controller module is initialized."
#define DOC_PYGAMESDL2CONTROLLERSETEVENTSTATE "set_eventstate(state) -> None\nSets the current state of events related to controllers"
#define DOC_PYGAMESDL2CONTROLLERGETEVENTSTATE "get_eventstate() -> bool\nGets the current state of events related to controllers"
#define DOC_PYGAMESDL2CONTROLLERGETCOUNT "get_count() -> int\nGet the number of joysticks connected"
#define DOC_PYGAMESDL2CONTROLLERISCONTROLLER "is_controller(index) -> bool\nCheck if the given joystick is supported by the game controller interface"
#define DOC_PYGAMESDL2CONTROLLERNAMEFORINDEX "name_forindex(index) -> name or None\nGet the name of the controller"
#define DOC_PYGAMESDL2CONTROLLERCONTROLLER "Controller(index) -> Controller\nCreate a new Controller object."
#define DOC_CONTROLLERQUIT "quit() -> None\nuninitialize the Controller"
#define DOC_CONTROLLERGETINIT "get_init() -> bool\ncheck if the Controller is initialized"
#define DOC_CONTROLLERFROMJOYSTICK "from_joystick(joystick) -> Controller\nCreate a Controller from a pygame.joystick.Joystick object"
#define DOC_CONTROLLERATTACHED "attached() -> bool\nCheck if the Controller has been opened and is currently connected."
#define DOC_CONTROLLERASJOYSTICK "as_joystick() -> Joystick object\nReturns a pygame.joystick.Joystick() object"
#define DOC_CONTROLLERGETAXIS "get_axis(axis) -> int\nGet the current state of a joystick axis"
#define DOC_CONTROLLERGETBUTTON "get_button(button) -> bool\nGet the current state of a button"
#define DOC_CONTROLLERGETMAPPING "get_mapping() -> mapping\nGet the mapping assigned to the controller"
#define DOC_CONTROLLERSETMAPPING "set_mapping(mapping) -> int\nAssign a mapping to the controller"
#define DOC_CONTROLLERRUMBLE "rumble(low_frequency, high_frequency, duration) -> bool\nStart a rumbling effect"
#define DOC_CONTROLLERSTOPRUMBLE "stop_rumble() -> None\nStop any rumble effect playing"


/* Docs in a comment... slightly easier to read. */

/*

pygame._sdl2.controller
Pygame module to work with controllers.

pygame._sdl2.controller.init
 init() -> None
initialize the controller module

pygame._sdl2.controller.quit
 quit() -> None
Uninitialize the controller module.

pygame._sdl2.controller.get_init
 get_init() -> bool
Returns True if the controller module is initialized.

pygame._sdl2.controller.set_eventstate
 set_eventstate(state) -> None
Sets the current state of events related to controllers

pygame._sdl2.controller.get_eventstate
 get_eventstate() -> bool
Gets the current state of events related to controllers

pygame._sdl2.controller.get_count
 get_count() -> int
Get the number of joysticks connected

pygame._sdl2.controller.is_controller
 is_controller(index) -> bool
Check if the given joystick is supported by the game controller interface

pygame._sdl2.controller.name_forindex
 name_forindex(index) -> name or None
Get the name of the controller

pygame._sdl2.controller.Controller
 Controller(index) -> Controller
Create a new Controller object.

pygame._sdl2.controller.Controller.quit
 quit() -> None
uninitialize the Controller

pygame._sdl2.controller.Controller.get_init
 get_init() -> bool
check if the Controller is initialized

pygame._sdl2.controller.Controller.from_joystick
 from_joystick(joystick) -> Controller
Create a Controller from a pygame.joystick.Joystick object

pygame._sdl2.controller.Controller.attached
 attached() -> bool
Check if the Controller has been opened and is currently connected.

pygame._sdl2.controller.Controller.as_joystick
 as_joystick() -> Joystick object
Returns a pygame.joystick.Joystick() object

pygame._sdl2.controller.Controller.get_axis
 get_axis(axis) -> int
Get the current state of a joystick axis

pygame._sdl2.controller.Controller.get_button
 get_button(button) -> bool
Get the current state of a button

pygame._sdl2.controller.Controller.get_mapping
 get_mapping() -> mapping
Get the mapping assigned to the controller

pygame._sdl2.controller.Controller.set_mapping
 set_mapping(mapping) -> int
Assign a mapping to the controller

pygame._sdl2.controller.Controller.rumble
 rumble(low_frequency, high_frequency, duration) -> bool
Start a rumbling effect

pygame._sdl2.controller.Controller.stop_rumble
 stop_rumble() -> None
Stop any rumble effect playing

*/