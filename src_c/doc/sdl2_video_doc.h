/* Auto generated file: with makeref.py .  Docs go in docs/reST/ref/ . */
#define DOC_PYGAMESDL2VIDEO "Experimental pygame module for porting new SDL video systems"
#define DOC_PYGAMESDL2VIDEOWINDOW "Window(title="pygame", size=(640, 480), position=None, fullscreen=False, fullscreen_desktop=False, keywords) -> Window\npygame object that represents a window"
#define DOC_WINDOWFROMDISPLAYMODULE "from_display_module() -> Window\nCreates window using window created by pygame.display.set_mode()."
#define DOC_WINDOWFROMWINDOW "from_window(other) -> Window\nCreate Window from another window. Could be from another UI toolkit."
#define DOC_WINDOWGRAB "grab -> bool\nGets or sets whether the mouse is confined to the window."
#define DOC_WINDOWRELATIVEMOUSE "relative_mouse -> bool\nGets or sets the window's relative mouse motion state."
#define DOC_WINDOWSETWINDOWED "set_windowed() -> None\nEnable windowed mode (exit fullscreen)."
#define DOC_WINDOWSETFULLSCREEN "set_fullscreen(desktop=False) -> None\nEnter fullscreen."
#define DOC_WINDOWTITLE "title -> string\nGets or sets whether the window title."
#define DOC_WINDOWDESTROY "destroy() -> None\nDestroys the window."
#define DOC_WINDOWHIDE "hide() -> None\nHide the window."
#define DOC_WINDOWSHOW "show() -> None\nShow the window."
#define DOC_WINDOWFOCUS "focus(input_only=False) -> None\nRaise the window above other windows and set the input focus. The "input_only" argument is only supported on X11."
#define DOC_WINDOWRESTORE "restore() -> None\nRestore the size and position of a minimized or maximized window."
#define DOC_WINDOWMAXIMIZE "maximize() -> None\nMaximize the window."
#define DOC_WINDOWMINIMIZE "maximize() -> None\nMinimize the window."
#define DOC_WINDOWRESIZABLE "resizable -> bool\nGets and sets whether the window is resizable."
#define DOC_WINDOWBORDERLESS "borderless -> bool\nAdd or remove the border from the window."
#define DOC_WINDOWSETICON "set_icon(surface) -> None\nSet the icon for the window."
#define DOC_WINDOWID "id -> int\nGet the unique window ID. *Read-only*"
#define DOC_WINDOWSIZE "size -> (int, int)\nGets and sets the window size."
#define DOC_WINDOWPOSITION "position -> (int, int) or WINDOWPOS_CENTERED or WINDOWPOS_UNDEFINED\nGets and sets the window position."
#define DOC_WINDOWOPACITY "opacity -> float\nGets and sets the window opacity. Between 0.0 (fully transparent) and 1.0 (fully opaque)."
#define DOC_WINDOWDISPLAYINDEX "display_index -> int\nGet the index of the display that owns the window. *Read-only*"
#define DOC_WINDOWSETMODALFOR "set_modal_for(Window) -> None\nSet the window as a modal for a parent window. This function is only supported on X11."
#define DOC_PYGAMESDL2VIDEOTEXTURE "Texture(renderer, size, depth=0, static=False, streaming=False, target=False) -> Texture\npygame object that representing a Texture."
#define DOC_TEXTUREFROMSURFACE "from_surface(renderer, surface) -> Texture\nCreate a texture from an existing surface."
#define DOC_TEXTURERENDERER "renderer -> Renderer\nGets the renderer associated with the Texture. *Read-only*"
#define DOC_TEXTUREWIDTH "width -> int\nGets the width of the Texture. *Read-only*"
#define DOC_TEXTUREHEIGHT "height -> int\nGets the height of the Texture. *Read-only*"
#define DOC_TEXTUREALPHA "alpha -> int\nGets and sets an additional alpha value multiplied into render copy operations."
#define DOC_TEXTUREBLENDMODE "blend_mode -> int\nGets and sets the blend mode for the Texture."
#define DOC_TEXTURECOLOR "color -> color\nGets and sets an additional color value multiplied into render copy operations."
#define DOC_TEXTUREGETRECT "get_rect(**kwargs) -> Rect\nGet the rectangular area of the texture."
#define DOC_TEXTUREDRAW "draw(srcrect=None, dstrect=None, angle=0, origin=None, flip_x=False, flip_y=False) -> None\nCopy a portion of the texture to the rendering target."
#define DOC_TEXTUREUPDATE "update(surface, area=None) -> None\nUpdate the texture with a Surface. WARNING: Slow operation, use sparingly."
#define DOC_PYGAMESDL2VIDEOIMAGE "Image(textureOrImage, srcrect=None) -> Image\nEasy way to use a portion of a Texture without worrying about srcrect all the time."
#define DOC_IMAGEGETRECT "get_rect() -> Rect\nGet the rectangular area of the Image."
#define DOC_IMAGEDRAW "draw(srcrect=None, dstrect=None) -> None\nCopy a portion of the Image to the rendering target."
#define DOC_IMAGEANGLE "angle -> float\nGets and sets the angle the Image draws itself with."
#define DOC_IMAGEORIGIN "origin -> (float, float) or None.\nGets and sets the origin. Origin=None means the Image will be rotated around its center."
#define DOC_IMAGEFLIPX "flip_x -> bool\nGets and sets whether the Image is flipped on the x axis."
#define DOC_IMAGEFLIPY "flip_y -> bool\nGets and sets whether the Image is flipped on the y axis."
#define DOC_IMAGECOLOR "color -> Color\nGets and sets the Image color modifier."
#define DOC_IMAGEALPHA "alpha -> float\nGets and sets the Image alpha modifier."
#define DOC_IMAGEBLENDMODE "blend_mode -> int\nGets and sets the blend mode for the Image."
#define DOC_IMAGETEXTURE "texture -> Texture\nGets and sets the Texture the Image is based on."
#define DOC_IMAGESRCRECT "srcrect -> Rect\nGets and sets the Rect the Image is based on."
#define DOC_PYGAMESDL2VIDEORENDERER "Renderer(window, index=-1, accelerated=-1, vsync=False, target_texture=False) -> Renderer\nCreate a 2D rendering context for a window."
#define DOC_RENDERERFROMWINDOW "from_window(window) -> Renderer\nEasy way to create a Renderer."
#define DOC_RENDERERDRAWBLENDMODE "draw_blend_mode -> int\nGets and sets the blend mode used by the drawing functions."
#define DOC_RENDERERDRAWCOLOR "draw_color -> Color\nGets and sets the color used by the drawing functions."
#define DOC_RENDERERCLEAR "clear() -> None\nClear the current rendering target with the drawing color."
#define DOC_RENDERERPRESENT "present() -> None\nUpdates the screen with any new rendering since previous call."
#define DOC_RENDERERGETVIEWPORT "get_viewport() -> Rect\nReturns the drawing area on the target."
#define DOC_RENDERERSETVIEWPORT "set_viewport(area) -> None\nSet the drawing area on the target. If area is None, the entire target will be used."
#define DOC_RENDERERLOGICALSIZE "logical_size -> (int width, int height)\nGets and sets the logical size."
#define DOC_RENDERERSCALE "scale -> (float x_scale, float y_scale)\nGets and sets the scale."
#define DOC_RENDERERTARGET "target -> Texture or None\nGets and sets the render target. None represents the default target (the renderer)."
#define DOC_RENDERERBLIT "blit(source, dest, area=None, special_flags=0)-> Rect\nFor compatibility purposes. Textures created by different Renderers cannot be shared!"
#define DOC_RENDERERDRAWLINE "draw_line(p1, p2) -> None\nDraws a line."
#define DOC_RENDERERDRAWPOINT "draw_point(point) -> None\nDraws a point."
#define DOC_RENDERERDRAWRECT "draw_rect(rect)-> None\nDraws a rectangle."
#define DOC_RENDERERFILLRECT "fill_rect(rect)-> None\nFills a rectangle."
#define DOC_RENDERERTOSURFACE "to_surface(surface=None, area=None)-> Surface\nRead pixels from current render target and create a pygame.Surface. WARNING: Slow operation, use sparingly."


/* Docs in a comment... slightly easier to read. */

/*

pygame.sdl2_video
Experimental pygame module for porting new SDL video systems

pygame._sdl2.video.Window
 Window(title="pygame", size=(640, 480), position=None, fullscreen=False, fullscreen_desktop=False, keywords) -> Window
pygame object that represents a window

pygame._sdl2.video.Window.from_display_module
 from_display_module() -> Window
Creates window using window created by pygame.display.set_mode().

pygame._sdl2.video.Window.from_window
 from_window(other) -> Window
Create Window from another window. Could be from another UI toolkit.

pygame._sdl2.video.Window.grab
 grab -> bool
Gets or sets whether the mouse is confined to the window.

pygame._sdl2.video.Window.relative_mouse
 relative_mouse -> bool
Gets or sets the window's relative mouse motion state.

pygame._sdl2.video.Window.set_windowed
 set_windowed() -> None
Enable windowed mode (exit fullscreen).

pygame._sdl2.video.Window.set_fullscreen
 set_fullscreen(desktop=False) -> None
Enter fullscreen.

pygame._sdl2.video.Window.title
 title -> string
Gets or sets whether the window title.

pygame._sdl2.video.Window.destroy
 destroy() -> None
Destroys the window.

pygame._sdl2.video.Window.hide
 hide() -> None
Hide the window.

pygame._sdl2.video.Window.show
 show() -> None
Show the window.

pygame._sdl2.video.Window.focus
 focus(input_only=False) -> None
Raise the window above other windows and set the input focus. The "input_only" argument is only supported on X11.

pygame._sdl2.video.Window.restore
 restore() -> None
Restore the size and position of a minimized or maximized window.

pygame._sdl2.video.Window.maximize
 maximize() -> None
Maximize the window.

pygame._sdl2.video.Window.minimize
 maximize() -> None
Minimize the window.

pygame._sdl2.video.Window.resizable
 resizable -> bool
Gets and sets whether the window is resizable.

pygame._sdl2.video.Window.borderless
 borderless -> bool
Add or remove the border from the window.

pygame._sdl2.video.Window.set_icon
 set_icon(surface) -> None
Set the icon for the window.

pygame._sdl2.video.Window.id
 id -> int
Get the unique window ID. *Read-only*

pygame._sdl2.video.Window.size
 size -> (int, int)
Gets and sets the window size.

pygame._sdl2.video.Window.position
 position -> (int, int) or WINDOWPOS_CENTERED or WINDOWPOS_UNDEFINED
Gets and sets the window position.

pygame._sdl2.video.Window.opacity
 opacity -> float
Gets and sets the window opacity. Between 0.0 (fully transparent) and 1.0 (fully opaque).

pygame._sdl2.video.Window.display_index
 display_index -> int
Get the index of the display that owns the window. *Read-only*

pygame._sdl2.video.Window.set_modal_for
 set_modal_for(Window) -> None
Set the window as a modal for a parent window. This function is only supported on X11.

pygame._sdl2.video.Texture
 Texture(renderer, size, depth=0, static=False, streaming=False, target=False) -> Texture
pygame object that representing a Texture.

pygame._sdl2.video.Texture.from_surface
 from_surface(renderer, surface) -> Texture
Create a texture from an existing surface.

pygame._sdl2.video.Texture.renderer
 renderer -> Renderer
Gets the renderer associated with the Texture. *Read-only*

pygame._sdl2.video.Texture.width
 width -> int
Gets the width of the Texture. *Read-only*

pygame._sdl2.video.Texture.height
 height -> int
Gets the height of the Texture. *Read-only*

pygame._sdl2.video.Texture.alpha
 alpha -> int
Gets and sets an additional alpha value multiplied into render copy operations.

pygame._sdl2.video.Texture.blend_mode
 blend_mode -> int
Gets and sets the blend mode for the Texture.

pygame._sdl2.video.Texture.color
 color -> color
Gets and sets an additional color value multiplied into render copy operations.

pygame._sdl2.video.Texture.get_rect
 get_rect(**kwargs) -> Rect
Get the rectangular area of the texture.

pygame._sdl2.video.Texture.draw
 draw(srcrect=None, dstrect=None, angle=0, origin=None, flip_x=False, flip_y=False) -> None
Copy a portion of the texture to the rendering target.

pygame._sdl2.video.Texture.update
 update(surface, area=None) -> None
Update the texture with a Surface. WARNING: Slow operation, use sparingly.

pygame._sdl2.video.Image
 Image(textureOrImage, srcrect=None) -> Image
Easy way to use a portion of a Texture without worrying about srcrect all the time.

pygame._sdl2.video.Image.get_rect
 get_rect() -> Rect
Get the rectangular area of the Image.

pygame._sdl2.video.Image.draw
 draw(srcrect=None, dstrect=None) -> None
Copy a portion of the Image to the rendering target.

pygame._sdl2.video.Image.angle
 angle -> float
Gets and sets the angle the Image draws itself with.

pygame._sdl2.video.Image.origin
 origin -> (float, float) or None.
Gets and sets the origin. Origin=None means the Image will be rotated around its center.

pygame._sdl2.video.Image.flip_x
 flip_x -> bool
Gets and sets whether the Image is flipped on the x axis.

pygame._sdl2.video.Image.flip_y
 flip_y -> bool
Gets and sets whether the Image is flipped on the y axis.

pygame._sdl2.video.Image.color
 color -> Color
Gets and sets the Image color modifier.

pygame._sdl2.video.Image.alpha
 alpha -> float
Gets and sets the Image alpha modifier.

pygame._sdl2.video.Image.blend_mode
 blend_mode -> int
Gets and sets the blend mode for the Image.

pygame._sdl2.video.Image.texture
 texture -> Texture
Gets and sets the Texture the Image is based on.

pygame._sdl2.video.Image.srcrect
 srcrect -> Rect
Gets and sets the Rect the Image is based on.

pygame._sdl2.video.Renderer
 Renderer(window, index=-1, accelerated=-1, vsync=False, target_texture=False) -> Renderer
Create a 2D rendering context for a window.

pygame._sdl2.video.Renderer.from_window
 from_window(window) -> Renderer
Easy way to create a Renderer.

pygame._sdl2.video.Renderer.draw_blend_mode
 draw_blend_mode -> int
Gets and sets the blend mode used by the drawing functions.

pygame._sdl2.video.Renderer.draw_color
 draw_color -> Color
Gets and sets the color used by the drawing functions.

pygame._sdl2.video.Renderer.clear
 clear() -> None
Clear the current rendering target with the drawing color.

pygame._sdl2.video.Renderer.present
 present() -> None
Updates the screen with any new rendering since previous call.

pygame._sdl2.video.Renderer.get_viewport
 get_viewport() -> Rect
Returns the drawing area on the target.

pygame._sdl2.video.Renderer.set_viewport
 set_viewport(area) -> None
Set the drawing area on the target. If area is None, the entire target will be used.

pygame._sdl2.video.Renderer.logical_size
 logical_size -> (int width, int height)
Gets and sets the logical size.

pygame._sdl2.video.Renderer.scale
 scale -> (float x_scale, float y_scale)
Gets and sets the scale.

pygame._sdl2.video.Renderer.target
 target -> Texture or None
Gets and sets the render target. None represents the default target (the renderer).

pygame._sdl2.video.Renderer.blit
 blit(source, dest, area=None, special_flags=0)-> Rect
For compatibility purposes. Textures created by different Renderers cannot be shared!

pygame._sdl2.video.Renderer.draw_line
 draw_line(p1, p2) -> None
Draws a line.

pygame._sdl2.video.Renderer.draw_point
 draw_point(point) -> None
Draws a point.

pygame._sdl2.video.Renderer.draw_rect
 draw_rect(rect)-> None
Draws a rectangle.

pygame._sdl2.video.Renderer.fill_rect
 fill_rect(rect)-> None
Fills a rectangle.

pygame._sdl2.video.Renderer.to_surface
 to_surface(surface=None, area=None)-> Surface
Read pixels from current render target and create a pygame.Surface. WARNING: Slow operation, use sparingly.

*/