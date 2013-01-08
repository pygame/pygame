/* Auto generated file: with makeref.py .  Docs go in src/ *.doc . */
#define DOC_PYGAMEDISPLAY "pygame module to control the display window and screen"

#define DOC_PYGAMEDISPLAYINIT "init() -> None\ninitialize the display module"

#define DOC_PYGAMEDISPLAYQUIT "quit() -> None\nuninitialize the display module"

#define DOC_PYGAMEDISPLAYGETINIT "get_init() -> bool\ntrue if the display module is initialized"

#define DOC_PYGAMEDISPLAYSETMODE "set_mode(resolution=(0,0), flags=0, depth=0) -> Surface\ninitialize a window or screen for display"

#define DOC_PYGAMEDISPLAYGETSURFACE "get_surface() -> Surface\nget a reference to the currently set display surface"

#define DOC_PYGAMEDISPLAYFLIP "flip() -> None\nupdate the full display Surface to the screen"

#define DOC_PYGAMEDISPLAYUPDATE "update(rectangle=None) -> None\nupdate(rectangle_list) -> None\nupdate portions of the screen for software displays"

#define DOC_PYGAMEDISPLAYGETDRIVER "get_driver() -> name\nget the name of the pygame display backend"

#define DOC_PYGAMEDISPLAYINFO "Info() -> VideoInfo\nCreate a video display information object"

#define DOC_PYGAMEDISPLAYGETWMINFO "get_wm_info() -> dict\nGet information about the current windowing system"

#define DOC_PYGAMEDISPLAYLISTMODES "list_modes(depth=0, flags=pygame.FULLSCREEN) -> list\nget list of available fullscreen modes"

#define DOC_PYGAMEDISPLAYMODEOK "mode_ok(size, flags=0, depth=0) -> depth\npick the best color depth for a display mode"

#define DOC_PYGAMEDISPLAYGLGETATTRIBUTE "gl_get_attribute(flag) -> value\nget the value for an opengl flag for the current display"

#define DOC_PYGAMEDISPLAYGLSETATTRIBUTE "gl_set_attribute(flag, value) -> None\nrequest an opengl display attribute for the display mode"

#define DOC_PYGAMEDISPLAYGETACTIVE "get_active() -> bool\ntrue when the display is active on the display"

#define DOC_PYGAMEDISPLAYICONIFY "iconify() -> bool\niconify the display surface"

#define DOC_PYGAMEDISPLAYTOGGLEFULLSCREEN "toggle_fullscreen() -> bool\nswitch between fullscreen and windowed displays"

#define DOC_PYGAMEDISPLAYSETGAMMA "set_gamma(red, green=None, blue=None) -> bool\nchange the hardware gamma ramps"

#define DOC_PYGAMEDISPLAYSETGAMMARAMP "set_gamma_ramp(red, green, blue) -> bool\nchange the hardware gamma ramps with a custom lookup"

#define DOC_PYGAMEDISPLAYSETICON "set_icon(Surface) -> None\nchange the system image for the display window"

#define DOC_PYGAMEDISPLAYSETCAPTION "set_caption(title, icontitle=None) -> None\nset the current window caption"

#define DOC_PYGAMEDISPLAYGETCAPTION "get_caption() -> (title, icontitle)\nget the current window caption"

#define DOC_PYGAMEDISPLAYSETPALETTE "set_palette(palette=None) -> None\nset the display color palette for indexed displays"



/* Docs in a comment... slightly easier to read. */

/*

pygame.display
pygame module to control the display window and screen

pygame.display.init
 init() -> None
initialize the display module

pygame.display.quit
 quit() -> None
uninitialize the display module

pygame.display.get_init
 get_init() -> bool
true if the display module is initialized

pygame.display.set_mode
 set_mode(resolution=(0,0), flags=0, depth=0) -> Surface
initialize a window or screen for display

pygame.display.get_surface
 get_surface() -> Surface
get a reference to the currently set display surface

pygame.display.flip
 flip() -> None
update the full display Surface to the screen

pygame.display.update
 update(rectangle=None) -> None
 update(rectangle_list) -> None
update portions of the screen for software displays

pygame.display.get_driver
 get_driver() -> name
get the name of the pygame display backend

pygame.display.Info
 Info() -> VideoInfo
Create a video display information object

pygame.display.get_wm_info
 get_wm_info() -> dict
Get information about the current windowing system

pygame.display.list_modes
 list_modes(depth=0, flags=pygame.FULLSCREEN) -> list
get list of available fullscreen modes

pygame.display.mode_ok
 mode_ok(size, flags=0, depth=0) -> depth
pick the best color depth for a display mode

pygame.display.gl_get_attribute
 gl_get_attribute(flag) -> value
get the value for an opengl flag for the current display

pygame.display.gl_set_attribute
 gl_set_attribute(flag, value) -> None
request an opengl display attribute for the display mode

pygame.display.get_active
 get_active() -> bool
true when the display is active on the display

pygame.display.iconify
 iconify() -> bool
iconify the display surface

pygame.display.toggle_fullscreen
 toggle_fullscreen() -> bool
switch between fullscreen and windowed displays

pygame.display.set_gamma
 set_gamma(red, green=None, blue=None) -> bool
change the hardware gamma ramps

pygame.display.set_gamma_ramp
 set_gamma_ramp(red, green, blue) -> bool
change the hardware gamma ramps with a custom lookup

pygame.display.set_icon
 set_icon(Surface) -> None
change the system image for the display window

pygame.display.set_caption
 set_caption(title, icontitle=None) -> None
set the current window caption

pygame.display.get_caption
 get_caption() -> (title, icontitle)
get the current window caption

pygame.display.set_palette
 set_palette(palette=None) -> None
set the display color palette for indexed displays

*/