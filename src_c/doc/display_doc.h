/* Auto generated file: with makeref.py .  Docs go in src/ *.doc . */
#define DOC_PYGAMEDISPLAY "pygame module to control the display window and screen"

#define DOC_PYGAMEDISPLAYINIT "init() -> None\nInitialize the display module"

#define DOC_PYGAMEDISPLAYQUIT "quit() -> None\nUninitialize the display module"

#define DOC_PYGAMEDISPLAYGETINIT "get_init() -> bool\nReturns True if the display module has been initialized"

#define DOC_PYGAMEDISPLAYSETMODE "set_mode(resolution=(0,0), flags=0, depth=0) -> Surface\nInitialize a window or screen for display"

#define DOC_PYGAMEDISPLAYGETSURFACE "get_surface() -> Surface\nGet a reference to the currently set display surface"

#define DOC_PYGAMEDISPLAYFLIP "flip() -> None\nUpdate the full display Surface to the screen"

#define DOC_PYGAMEDISPLAYUPDATE "update(rectangle=None) -> None\nupdate(rectangle_list) -> None\nUpdate portions of the screen for software displays"

#define DOC_PYGAMEDISPLAYGETDRIVER "get_driver() -> name\nGet the name of the pygame display backend"

#define DOC_PYGAMEDISPLAYINFO "Info() -> VideoInfo\nCreate a video display information object"

#define DOC_PYGAMEDISPLAYGETWMINFO "get_wm_info() -> dict\nGet information about the current windowing system"

#define DOC_PYGAMEDISPLAYLISTMODES "list_modes(depth=0, flags=pygame.FULLSCREEN) -> list\nGet list of available fullscreen modes"

#define DOC_PYGAMEDISPLAYMODEOK "mode_ok(size, flags=0, depth=0) -> depth\nPick the best color depth for a display mode"

#define DOC_PYGAMEDISPLAYGLGETATTRIBUTE "gl_get_attribute(flag) -> value\nGet the value for an OpenGL flag for the current display"

#define DOC_PYGAMEDISPLAYGLSETATTRIBUTE "gl_set_attribute(flag, value) -> None\nRequest an OpenGL display attribute for the display mode"

#define DOC_PYGAMEDISPLAYGETACTIVE "get_active() -> bool\nReturns True when the display is active on the display"

#define DOC_PYGAMEDISPLAYICONIFY "iconify() -> bool\nIconify the display surface"

#define DOC_PYGAMEDISPLAYTOGGLEFULLSCREEN "toggle_fullscreen() -> bool\nSwitch between fullscreen and windowed displays"

#define DOC_PYGAMEDISPLAYSETGAMMA "set_gamma(red, green=None, blue=None) -> bool\nChange the hardware gamma ramps"

#define DOC_PYGAMEDISPLAYSETGAMMARAMP "set_gamma_ramp(red, green, blue) -> bool\nChange the hardware gamma ramps with a custom lookup"

#define DOC_PYGAMEDISPLAYSETICON "set_icon(Surface) -> None\nChange the system image for the display window"

#define DOC_PYGAMEDISPLAYSETCAPTION "set_caption(title, icontitle=None) -> None\nSet the current window caption"

#define DOC_PYGAMEDISPLAYGETCAPTION "get_caption() -> (title, icontitle)\nGet the current window caption"

#define DOC_PYGAMEDISPLAYSETPALETTE "set_palette(palette=None) -> None\nSet the display color palette for indexed displays"

#define DOC_PYGAMEDISPLAYGETNUMDISPLAYS "get_num_displays() -> int\nReturn the number of displays"



/* Docs in a comment... slightly easier to read. */

/*

pygame.display
pygame module to control the display window and screen

pygame.display.init
 init() -> None
Initialize the display module

pygame.display.quit
 quit() -> None
Uninitialize the display module

pygame.display.get_init
 get_init() -> bool
Returns True if the display module has been initialized

pygame.display.set_mode
 set_mode(resolution=(0,0), flags=0, depth=0) -> Surface
Initialize a window or screen for display

pygame.display.get_surface
 get_surface() -> Surface
Get a reference to the currently set display surface

pygame.display.flip
 flip() -> None
Update the full display Surface to the screen

pygame.display.update
 update(rectangle=None) -> None
 update(rectangle_list) -> None
Update portions of the screen for software displays

pygame.display.get_driver
 get_driver() -> name
Get the name of the pygame display backend

pygame.display.Info
 Info() -> VideoInfo
Create a video display information object

pygame.display.get_wm_info
 get_wm_info() -> dict
Get information about the current windowing system

pygame.display.list_modes
 list_modes(depth=0, flags=pygame.FULLSCREEN) -> list
Get list of available fullscreen modes

pygame.display.mode_ok
 mode_ok(size, flags=0, depth=0) -> depth
Pick the best color depth for a display mode

pygame.display.gl_get_attribute
 gl_get_attribute(flag) -> value
Get the value for an OpenGL flag for the current display

pygame.display.gl_set_attribute
 gl_set_attribute(flag, value) -> None
Request an OpenGL display attribute for the display mode

pygame.display.get_active
 get_active() -> bool
Returns True when the display is active on the display

pygame.display.iconify
 iconify() -> bool
Iconify the display surface

pygame.display.toggle_fullscreen
 toggle_fullscreen() -> bool
Switch between fullscreen and windowed displays

pygame.display.set_gamma
 set_gamma(red, green=None, blue=None) -> bool
Change the hardware gamma ramps

pygame.display.set_gamma_ramp
 set_gamma_ramp(red, green, blue) -> bool
Change the hardware gamma ramps with a custom lookup

pygame.display.set_icon
 set_icon(Surface) -> None
Change the system image for the display window

pygame.display.set_caption
 set_caption(title, icontitle=None) -> None
Set the current window caption

pygame.display.get_caption
 get_caption() -> (title, icontitle)
Get the current window caption

pygame.display.set_palette
 set_palette(palette=None) -> None
Set the display color palette for indexed displays

pygame.display.get_num_displays
 get_num_displays() -> int
Return the number of displays

*/