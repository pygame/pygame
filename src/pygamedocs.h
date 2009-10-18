/* Auto generated file: with makeref.py .  Docs go in src/ *.doc . */
#define DOC_PYGAME "the top level pygame package"

#define DOC_PYGAMEINIT "pygame.init(): return (numpass, numfail)\ninitialize all imported pygame modules"

#define DOC_PYGAMEQUIT "pygame.quit(): return None\nuninitialize all pygame modules"

#define DOC_PYGAMEERROR "raise pygame.error, message\nstandard pygame exception"

#define DOC_PYGAMEGETERROR "pygame.get_error(): return errorstr\nget the current error message"

#define DOC_PYGAMESETERROR "pygame.set_error(error_msg): return None\nset the current error message"

#define DOC_PYGAMEGETSDLVERSION "pygame.get_sdl_version(): return major, minor, patch\nget the version number of SDL"

#define DOC_PYGAMEGETSDLBYTEORDER "pygame.get_sdl_byteorder(): return int\nget the byte order of SDL"

#define DOC_PYGAMEREGISTERQUIT "register_quit(callable): return None\nregister a function to be called when pygame quits"

#define DOC_PYGAMEVERSION "module pygame.version\nsmall module containing version information"

#define DOC_PYGAMEVERSIONVER "pygame.version.ver = '1.2'\nversion number as a string"

#define DOC_PYGAMEVERSIONVERNUM "pygame.version.vernum = (1, 5, 3)\ntupled integers of the version"

#define DOC_PYGAMECAMERA "pygame module for camera use"

#define DOC_PYGAMECAMERACOLORSPACE "pygame.camera.colorspace(Surface, format, DestSurface = None): return Surface\nSurface colorspace conversion"

#define DOC_PYGAMECAMERALISTCAMERAS "pygame.camera.list_cameras(): return [cameras]\nreturns a list of available cameras"

#define DOC_PYGAMECAMERACAMERA "pygame.camera.Camera(device, (width, height), format): return Camera\nload a camera"

#define DOC_CAMERASTART "Camera.start(): return None\nopens, initializes, and starts capturing"

#define DOC_CAMERASTOP "Camera.stop(): return None\nstops, uninitializes, and closes the camera"

#define DOC_CAMERAGETCONTROLS "Camera.get_controls(): return (hflip = bool, vflip = bool, brightness)\ngets current values of user controls"

#define DOC_CAMERASETCONTROLS "Camera.set_controls(hflip = bool, vflip = bool, brightness): return (hflip = bool, vflip = bool, brightness)\nchanges camera settings if supported by the camera"

#define DOC_CAMERAGETSIZE "Camera.get_size(): return (width, height)\nreturns the dimensions of the images being recorded"

#define DOC_CAMERAQUERYIMAGE "Camera.query_image(): return bool\nchecks if a frame is ready"

#define DOC_CAMERAGETIMAGE "Camera.get_image(Surface = None): return Surface\ncaptures an image as a Surface"

#define DOC_CAMERAGETRAW "Camera.get_raw(): return string\nreturns an unmodified image as a string"

#define DOC_PYGAMECDROM "pygame module for audio cdrom control"

#define DOC_PYGAMECDROMINIT "pygame.cdrom.init(): return None\ninitialize the cdrom module"

#define DOC_PYGAMECDROMQUIT "pygame.cdrom.quit(): return None\nuninitialize the cdrom module"

#define DOC_PYGAMECDROMGETINIT "pygame.cdrom.get_init(): return bool\ntrue if the cdrom module is initialized"

#define DOC_PYGAMECDROMGETCOUNT "pygame.cdrom.get_count(): return count\nnumber of cd drives on the system"

#define DOC_PYGAMECDROMCD "pygame.cdrom.CD(id): return CD\nclass to manage a cdrom drive"

#define DOC_CDINIT "CD.init(): return None\ninitialize a cdrom drive for use"

#define DOC_CDQUIT "CD.quit(): return None\nuninitialize a cdrom drive for use"

#define DOC_CDGETINIT "CD.get_init(): return bool\ntrue if this cd device initialized"

#define DOC_CDPLAY "CD.play(track, start=None, end=None): return None\nstart playing audio"

#define DOC_CDSTOP "CD.stop(): return None\nstop audio playback"

#define DOC_CDPAUSE "CD.pause(): return None\ntemporarily stop audio playback"

#define DOC_CDRESUME "CD.resume(): return None\nunpause audio playback"

#define DOC_CDEJECT "CD.eject(): return None\neject or open the cdrom drive"

#define DOC_CDGETID "CD.get_id(): return id\nthe index of the cdrom drive"

#define DOC_CDGETNAME "CD.get_name(): return name\nthe system name of the cdrom drive"

#define DOC_CDGETBUSY "CD.get_busy(): return bool\ntrue if the drive is playing audio"

#define DOC_CDGETPAUSED "CD.get_paused(): return bool\ntrue if the drive is paused"

#define DOC_CDGETCURRENT "CD.get_current(): return track, seconds\nthe current audio playback position"

#define DOC_CDGETEMPTY "CD.get_empty(): return bool\nFalse if a cdrom is in the drive"

#define DOC_CDGETNUMTRACKS "CD.get_numtracks(): return count\nthe number of tracks on the cdrom"

#define DOC_CDGETTRACKAUDIO "CD.get_track_audio(track): return bool\ntrue if the cdrom track has audio data"

#define DOC_CDGETALL "CD.get_all(): return [(audio, start, end, lenth), ...]\nget all track information"

#define DOC_CDGETTRACKSTART "CD.get_track_start(track): return seconds\nstart time of a cdrom track"

#define DOC_CDGETTRACKLENGTH "CD.get_track_length(track): return seconds\nlength of a cdrom track"

#define DOC_PYGAMECOLOR "pygame.Color(name): Return Color\npygame.Color(r, g, b, a): Return Color\npygame.Color(rgbvalue): Return Color\npygame object for color representations"

#define DOC_COLORR "Color.r: Return int\nGets or sets the red value of the Color."

#define DOC_COLORG "Color.g: Return int\nGets or sets the green value of the Color."

#define DOC_COLORB "Color.b: Return int\nGets or sets the blue value of the Color."

#define DOC_COLORA "Color.a: Return int\nGets or sets the alpha value of the Color."

#define DOC_COLORCMY "Color.cmy: Return tuple\nGets or sets the CMY representation of the Color."

#define DOC_COLORHSVA "Color.hsva: Return tuple\nGets or sets the HSVA representation of the Color."

#define DOC_COLORHSLA "Color.hsla: Return tuple\nGets or sets the HSLA representation of the Color."

#define DOC_COLORI1I2I3 "Color.i1i2i3: Return tuple\nGets or sets the I1I2I3 representation of the Color."

#define DOC_COLORNORMALIZE "Color.normalize(): Return tuple\nReturns the normalized RGBA values of the Color."

#define DOC_COLORCORRECTGAMMA "Color.correct_gamma (gamma): Return Color\nApplies a certain gamma value to the Color."

#define DOC_COLORSETLENGTH "Color.set_length(len)\nSet the number of elements in the Color to 1,2,3, or 4."

#define DOC_PYGAMECURSORS "pygame module for cursor resources"

#define DOC_PYGAMECURSORSCOMPILE "pygame.cursor.compile(strings, black='X', white='.', xor='o'): return data, mask\ncreate binary cursor data from simple strings"

#define DOC_PYGAMECURSORSLOADXBM "pygame.cursors.load_xbm(cursorfile): return cursor_args\npygame.cursors.load_xbm(cursorfile, maskfile): return cursor_args\nload cursor data from an xbm file"

#define DOC_PYGAMEDISPLAY "pygame module to control the display window and screen"

#define DOC_PYGAMEDISPLAYINIT "pygame.display.init(): return None\ninitialize the display module"

#define DOC_PYGAMEDISPLAYQUIT "pygame.display.quit(): return None\nuninitialize the display module"

#define DOC_PYGAMEDISPLAYGETINIT "pygame.display.get_init(): return bool\ntrue if the display module is initialized"

#define DOC_PYGAMEDISPLAYSETMODE "pygame.display.set_mode(resolution=(0,0), flags=0, depth=0): return Surface\ninitialize a window or screen for display"

#define DOC_PYGAMEDISPLAYGETSURFACE "pygame.display.get_surface(): return Surface\nget a reference to the currently set display surface"

#define DOC_PYGAMEDISPLAYFLIP "pygame.display.flip(): return None\nupdate the full display Surface to the screen"

#define DOC_PYGAMEDISPLAYUPDATE "pygame.display.update(rectangle=None): return None\npygame.display.update(rectangle_list): return None\nupdate portions of the screen for software displays"

#define DOC_PYGAMEDISPLAYGETDRIVER "pygame.display.get_driver(): return name\nget the name of the pygame display backend"

#define DOC_PYGAMEDISPLAYINFO "pygame.display.Info(): return VideoInfo\nCreate a video display information object"

#define DOC_PYGAMEDISPLAYGETWMINFO "pygame.display.get_wm_info(): return dict\nGet information about the current windowing system"

#define DOC_PYGAMEDISPLAYLISTMODES "pygame.display.list_modes(depth=0, flags=pygame.FULLSCREEN): return list\nget list of available fullscreen modes"

#define DOC_PYGAMEDISPLAYMODEOK "pygame.display.mode_ok(size, flags=0, depth=0): return depth\npick the best color depth for a display mode"

#define DOC_PYGAMEDISPLAYGLGETATTRIBUTE "pygame.display.gl_get_attribute(flag): return value\nget the value for an opengl flag for the current display"

#define DOC_PYGAMEDISPLAYGLSETATTRIBUTE "pygame.display.gl_set_attribute(flag, value): return None\nrequest an opengl display attribute for the display mode"

#define DOC_PYGAMEDISPLAYGETACTIVE "pygame.display.get_active(): return bool\ntrue when the display is active on the display"

#define DOC_PYGAMEDISPLAYICONIFY "pygame.display.iconify(): return bool\niconify the display surface"

#define DOC_PYGAMEDISPLAYTOGGLEFULLSCREEN "pygame.display.toggle_fullscreen(): return bool\nswitch between fullscreen and windowed displays"

#define DOC_PYGAMEDISPLAYSETGAMMA "pygame.display.set_gamma(red, green=None, blue=None): return bool\nchange the hardware gamma ramps"

#define DOC_PYGAMEDISPLAYSETGAMMARAMP "change the hardware gamma ramps with a custom lookup\npygame.display.set_gamma_ramp(red, green, blue): return bool\nset_gamma_ramp(red, green, blue): return bool"

#define DOC_PYGAMEDISPLAYSETICON "pygame.display.set_icon(Surface): return None\nchange the system image for the display window"

#define DOC_PYGAMEDISPLAYSETCAPTION "pygame.display.set_caption(title, icontitle=None): return None\nset the current window caption"

#define DOC_PYGAMEDISPLAYGETCAPTION "pygame.display.get_caption(): return (title, icontitle)\nget the current window caption"

#define DOC_PYGAMEDISPLAYSETPALETTE "pygame.display.set_palette(palette=None): return None\nset the display color palette for indexed displays"

#define DOC_PYGAMEDRAW "pygame module for drawing shapes"

#define DOC_PYGAMEDRAWRECT "pygame.draw.rect(Surface, color, Rect, width=0): return Rect\ndraw a rectangle shape"

#define DOC_PYGAMEDRAWPOLYGON "pygame.draw.polygon(Surface, color, pointlist, width=0): return Rect\ndraw a shape with any number of sides"

#define DOC_PYGAMEDRAWCIRCLE "pygame.draw.circle(Surface, color, pos, radius, width=0): return Rect\ndraw a circle around a point"

#define DOC_PYGAMEDRAWELLIPSE "pygame.draw.ellipse(Surface, color, Rect, width=0): return Rect\ndraw a round shape inside a rectangle"

#define DOC_PYGAMEDRAWARC "pygame.draw.arc(Surface, color, Rect, start_angle, stop_angle, width=1): return Rect\ndraw a partial section of an ellipse"

#define DOC_PYGAMEDRAWLINE "pygame.draw.line(Surface, color, start_pos, end_pos, width=1): return Rect\ndraw a straight line segment"

#define DOC_PYGAMEDRAWLINES "pygame.draw.lines(Surface, color, closed, pointlist, width=1): return Rect\ndraw multiple contiguous line segments"

#define DOC_PYGAMEDRAWAALINE "pygame.draw.aaline(Surface, color, startpos, endpos, blend=1): return Rect\ndraw fine antialiased lines"

#define DOC_PYGAMEDRAWAALINES "pygame.draw.aalines(Surface, color, closed, pointlist, blend=1): return Rect"

#define DOC_PYGAMEEVENT "pygame module for interacting with events and queues"

#define DOC_PYGAMEEVENTPUMP "pygame.event.pump(): return None\ninternally process pygame event handlers"

#define DOC_PYGAMEEVENTGET "pygame.event.get(): return Eventlist\npygame.event.get(type): return Eventlist\npygame.event.get(typelist): return Eventlist\nget events from the queue"

#define DOC_PYGAMEEVENTPOLL "pygame.event.poll(): return Event\nget a single event from the queue"

#define DOC_PYGAMEEVENTWAIT "pygame.event.wait(): return Event\nwait for a single event from the queue"

#define DOC_PYGAMEEVENTPEEK "pygame.event.peek(type): return bool\npygame.event.peek(typelist): return bool\ntest if event types are waiting on the queue"

#define DOC_PYGAMEEVENTCLEAR "pygame.event.clear(): return None\npygame.event.clear(type): return None\npygame.event.clear(typelist): return None\nremove all events from the queue"

#define DOC_PYGAMEEVENTEVENTNAME "pygame.event.event_name(type): return string\nget the string name from and event id"

#define DOC_PYGAMEEVENTSETBLOCKED "pygame.event.set_blocked(type): return None\npygame.event.set_blocked(typelist): return None\npygame.event.set_blocked(None): return None\ncontrol which events are allowed on the queue"

#define DOC_PYGAMEEVENTSETALLOWED "pygame.event.set_allowed(type): return None\npygame.event.set_allowed(typelist): return None\npygame.event.set_allowed(None): return None\ncontrol which events are allowed on the queue"

#define DOC_PYGAMEEVENTGETBLOCKED "pygame.event.get_blocked(type): return bool\ntest if a type of event is blocked from the queue"

#define DOC_PYGAMEEVENTSETGRAB "pygame.event.set_grab(bool): return None\ncontrol the sharing of input devices with other applications"

#define DOC_PYGAMEEVENTGETGRAB "pygame.event.get_grab(): return bool\ntest if the program is sharing input devices"

#define DOC_PYGAMEEVENTPOST "pygame.event.post(Event): return None\nplace a new event on the queue"

#define DOC_PYGAMEEVENTEVENT "pygame.event.Event(type, dict): return Event\npygame.event.Event(type, **attributes): return Event\ncreate a new event object"

#define DOC_PYGAMEEXAMPLES "module of example programs"

#define DOC_PYGAMEEXAMPLESALIENSMAIN "pygame.aliens.main(): return None\nplay the full aliens example"

#define DOC_PYGAMEEXAMPLESOLDALIENMAIN "pygame.examples.oldalien.main(): return None\nplay the original aliens example"

#define DOC_PYGAMEEXAMPLESSTARSMAIN "pygame.examples.stars.main(): return None\nrun a simple starfield example"

#define DOC_PYGAMEEXAMPLESCHIMPMAIN "pygame.examples.chimp.main(): return None\nhit the moving chimp"

#define DOC_PYGAMEEXAMPLESMOVEITMAIN "pygame.examples.moveit.main(): return None\ndisplay animated objects on the screen"

#define DOC_PYGAMEEXAMPLESFONTYMAIN "pygame.examples.fonty.main(): return None\nrun a font rendering example"

#define DOC_PYGAMEEXAMPLESVGRADEMAIN "pygame.examples.vgrade.main(): return None\ndisplay a vertical gradient"

#define DOC_PYGAMEEXAMPLESEVENTLISTMAIN "pygame.examples.eventlist.main(): return None\ndisplay pygame events"

#define DOC_PYGAMEEXAMPLESARRAYDEMOMAIN "pygame.examples.arraydemo.main(arraytype=None): return None\nshow various surfarray effects"

#define DOC_PYGAMEEXAMPLESSOUNDMAIN "pygame.examples.sound.main(file_path=None): return None\nload and play a sound"

#define DOC_PYGAMEEXAMPLESSOUNDARRAYDEMOSMAIN "pygame.examples.sound_array_demos.main(arraytype=None): return None\nplay various sndarray effects"

#define DOC_PYGAMEEXAMPLESLIQUIDMAIN "pygame.examples.liquid.main(): return None\ndisplay an animated liquid effect"

#define DOC_PYGAMEEXAMPLESGLCUBEMAIN "pygame.examples.glcube.main(): return None\ndisplay an animated 3D cube using OpenGL"

#define DOC_PYGAMEEXAMPLESSCRAPCLIPBOARDMAIN "pygame.examples.scrap_clipboard.main(): return None\naccess the clipboard"

#define DOC_PYGAMEEXAMPLESMASKMAIN "pygame.examples.mask.main(*args): return None\ndisplay multiple images bounce off each other using collision detection"

#define DOC_PYGAMEEXAMPLESTESTSPRITEMAIN "pygame.examples.testsprite.main(update_rects = True, use_static = False, use_FastRenderGroup = False, screen_dims = [640, 480], use_alpha = False, flags = 0): return None\nshow lots of sprites moving around"

#define DOC_PYGAMEEXAMPLESHEADLESSNOWINDOWSNEEDEDMAIN "pygame.examples.headless_no_windows_needed.main(fin, fout, w, h): return None\nwrite an image file that is smoothscaled copy of an input file"

#define DOC_PYGAMEEXAMPLESFASTEVENTSMAIN "pygame.examples.fastevents.main(): return None\nstress test the fastevents module"

#define DOC_PYGAMEEXAMPLESOVERLAYMAIN "pygame.examples.overlay.main(fname): return None\nplay a .pgm video using overlays"

#define DOC_PYGAMEEXAMPLESBLENDFILLMAIN "pygame.examples.blend_fill.main(): return None\ndemonstrate the various surface.fill method blend options"

#define DOC_PYGAMEEXAMPLESCURSORSMAIN "pygame.examples.cursors.main(): return None\ndisplay two different custom cursors"

#define DOC_PYGAMEEXAMPLESPIXELARRAYMAIN "pygame.examples.pixelarray.main(): return None\ndisplay various pixelarray generated effects"

#define DOC_PYGAMEEXAMPLESSCALETESTMAIN "pygame.examples.scaletest.main(imagefile, convert_alpha=False, run_speed_test=True): return None\ninteractively scale an image using smoothscale"

#define DOC_PYGAMEEXAMPLESMIDIMAIN "pygame.examples.midi.main(mode='output', device_id=None): return None\nrun a midi example"

#define DOC_PYGAMEEXAMPLESSCROLLMAIN "pygame.examples.scroll.main(image_file=None): return None\nrun a Surface.scroll example that shows a magnified image"

#define DOC_PYGAMEEXAMPLESMOVIEPLAYERMAIN "pygame.examples.moveplayer.main(filepath): return None\nplay an MPEG movie"

#define DOC_PYGAMEFONT "pygame module for loading and rendering fonts"

#define DOC_PYGAMEFONTINIT "pygame.font.init(): return None\ninitialize the font module"

#define DOC_PYGAMEFONTQUIT "pygame.font.quit(): return None\nuninitialize the font module"

#define DOC_PYGAMEFONTGETINIT "pygame.font.get_init(): return bool\ntrue if the font module is initialized"

#define DOC_PYGAMEFONTGETDEFAULTFONT "pygame.font.get_default_font(): return string\nget the filename of the default font"

#define DOC_PYGAMEFONTGETFONTS "pygame.font.get_fonts(): return list of strings\nget all available fonts"

#define DOC_PYGAMEFONTMATCHFONT "pygame.font.match_font(name, bold=False, italic=False): return path\nfind a specific font on the system"

#define DOC_PYGAMEFONTSYSFONT "pygame.font.SysFont(name, size, bold=False, italic=False): return Font\ncreate a Font object from the system fonts"

#define DOC_PYGAMEFONTFONT "pygame.font.Font(filename, size): return Font\npygame.font.Font(object, size): return Font\ncreate a new Font object from a file"

#define DOC_FONTRENDER "Font.render(text, antialias, color, background=None): return Surface\ndraw text on a new Surface"

#define DOC_FONTSIZE "Font.size(text): return (width, height)\ndetermine the amount of space needed to render text"

#define DOC_FONTSETUNDERLINE "Font.set_underline(bool): return None\ncontrol if text is rendered with an underline"

#define DOC_FONTGETUNDERLINE "Font.get_underline(): return bool\ncheck if text will be rendered with an underline"

#define DOC_FONTSETBOLD "Font.set_bold(bool): return None\nenable fake rendering of bold text"

#define DOC_FONTGETBOLD "Font.get_bold(): return bool\ncheck if text will be rendered bold"

#define DOC_FONTSETITALIC "Font.set_bold(bool): return None\nenable fake rendering of italic text"

#define DOC_FONTMETRICS "Font.metrics(text): return list\nGets the metrics for each character in the pased string."

#define DOC_FONTGETITALIC "Font.get_italic(): return bool\ncheck if the text will be rendered italic"

#define DOC_FONTGETLINESIZE "Font.get_linesize(): return int\nget the line space of the font text"

#define DOC_FONTGETHEIGHT "Font.get_height(): return int\nget the height of the font"

#define DOC_FONTGETASCENT "Font.get_ascent(): return int\nget the ascent of the font"

#define DOC_FONTGETDESCENT "Font.get_descent(): return int\nget the descent of the font"

#define DOC_PYGAMEGFXDRAW "pygame module for drawing shapes"

#define DOC_PYGAMEGFXDRAWPIXEL "pygame.gfxdraw.pixel(surface, x, y, color): return None\nplace a pixel"

#define DOC_PYGAMEGFXDRAWHLINE "pygame.gfxdraw.hline(surface, x1, x2, y, color): return None\ndraw a horizontal line"

#define DOC_PYGAMEGFXDRAWVLINE "pgyame.gfxdraw.vline(surface, x, y1, y2, color): return None\ndraw a vertical line"

#define DOC_PYGAMEGFXDRAWRECTANGLE "pgyame.gfxdraw.rectangle(surface, rect, color): return None\ndraw a rectangle"

#define DOC_PYGAMEGFXDRAWBOX "pgyame.gfxdraw.box(surface, rect, color): return None\ndraw a box"

#define DOC_PYGAMEGFXDRAWLINE "pgyame.gfxdraw.line(surface, x1, y1, x2, y2, color): return None\ndraw a line"

#define DOC_PYGAMEGFXDRAWCIRCLE "pgyame.gfxdraw.circle(surface, x, y, r, color): return None\ndraw a circle"

#define DOC_PYGAMEGFXDRAWARC "pgyame.gfxdraw.arc(surface, x, y, r, start, end, color): return None\ndraw an arc"

#define DOC_PYGAMEGFXDRAWAACIRCLE "pgyame.gfxdraw.aacircle(surface, x, y, r, color): return None\ndraw an anti-aliased circle"

#define DOC_PYGAMEGFXDRAWFILLEDCIRCLE "pgyame.gfxdraw.filled_circle(surface, x, y, r, color): return None\ndraw a filled circle"

#define DOC_PYGAMEGFXDRAWELLIPSE "pgyame.gfxdraw.ellipse(surface, x, y, rx, ry, color): return None\ndraw an ellipse"

#define DOC_PYGAMEGFXDRAWAAELLIPSE "pgyame.gfxdraw.aaellipse(surface, x, y, rx, ry, color): return None\ndraw an anti-aliased ellipse"

#define DOC_PYGAMEGFXDRAWFILLEDELLIPSE "pgyame.gfxdraw.filled_ellipse(surface, x, y, rx, ry, color): return None\ndraw a filled ellipse"

#define DOC_PYGAMEGFXDRAWPIE "pgyame.gfxdraw.pie(surface, x, y, r, start, end, color): return None\ndraw a pie"

#define DOC_PYGAMEGFXDRAWTRIGON "pgyame.gfxdraw.trigon(surface, x1, y1, x2, y2, x3, y3, color): return None\ndraw a triangle"

#define DOC_PYGAMEGFXDRAWAATRIGON "pgyame.gfxdraw.aatrigon(surface, x1, y1, x2, y2, x3, y3, color): return None\ndraw an anti-aliased triangle"

#define DOC_PYGAMEGFXDRAWFILLEDTRIGON "pgyame.gfxdraw.filled_trigon(surface, x1, y1, x3, y2, x3, y3, color): return None\ndraw a filled trigon"

#define DOC_PYGAMEGFXDRAWPOLYGON "pgyame.gfxdraw.polygon(surface, points, color): return None\ndraw a polygon"

#define DOC_PYGAMEGFXDRAWAAPOLYGON "pgyame.gfxdraw.aapolygon(surface, points, color): return None\ndraw an anti-aliased polygon"

#define DOC_PYGAMEGFXDRAWFILLEDPOLYGON "pgyame.gfxdraw.filled_polygon(surface, points, color): return None\ndraw a filled polygon"

#define DOC_PYGAMEGFXDRAWTEXTUREDPOLYGON "pgyame.gfxdraw.textured_polygon(surface, points, texture, tx, ty): return None\ndraw a textured polygon"

#define DOC_PYGAMEGFXDRAWBEZIER "pgyame.gfxdraw.bezier(surface, points, steps, color): return None\ndraw a bezier curve"

#define DOC_PYGAMEIMAGE "pygame module for image transfer"

#define DOC_PYGAMEIMAGELOAD "pygame.image.load(filename): return Surface\npygame.image.load(fileobj, namehint=""): return Surface\nload new image from a file"

#define DOC_PYGAMEIMAGESAVE "pygame.image.save(Surface, filename): return None\nsave an image to disk"

#define DOC_PYGAMEIMAGEGETEXTENDED "pygame.image.get_extended(): return bool\ntest if extended image formats can be loaded"

#define DOC_PYGAMEIMAGETOSTRING "pygame.image.tostring(Surface, format, flipped=False): return string\ntransfer image to string buffer"

#define DOC_PYGAMEIMAGEFROMSTRING "pygame.image.fromstring(string, size, format, flipped=False): return Surface\ncreate new Surface from a string buffer"

#define DOC_PYGAMEIMAGEFROMBUFFER "pygame.image.frombuffer(string, size, format): return Surface\ncreate a new Surface that shares data inside a string buffer"

#define DOC_PYGAMEJOYSTICK "pygame module for interacting with joystick devices"

#define DOC_PYGAMEJOYSTICKINIT "pygame.joystick.init(): return None\ninitialize the joystick module"

#define DOC_PYGAMEJOYSTICKQUIT "pygame.joystick.quit(): return None\nuninitialize the joystick module"

#define DOC_PYGAMEJOYSTICKGETINIT "pygame.joystick.get_init(): return bool\ntrue if the joystick module is initialized"

#define DOC_PYGAMEJOYSTICKGETCOUNT "pygame.joystick.get_count(): return count\nnumber of joysticks on the system"

#define DOC_PYGAMEJOYSTICKJOYSTICK "pygame.joystick.Joystick(id): return Joystick\ncreate a new Joystick object"

#define DOC_JOYSTICKINIT "Joystick.init(): return None\ninitialize the Joystick"

#define DOC_JOYSTICKQUIT "Joystick.quit(): return None\nuninitialize the Joystick"

#define DOC_JOYSTICKGETINIT "Joystick.get_init(): return bool\ncheck if the Joystick is initialized"

#define DOC_JOYSTICKGETID "Joystick.get_id(): return int\nget the Joystick ID"

#define DOC_JOYSTICKGETNAME "Joystick.get_name(): return string\nget the Joystick system name"

#define DOC_JOYSTICKGETNUMAXES "Joystick.get_numaxes(): return int\nget the number of axes on a Joystick"

#define DOC_JOYSTICKGETAXIS "Joystick.get_axis(axis_number): return float\nget the current position of an axis"

#define DOC_JOYSTICKGETNUMBALLS "Joystick.get_numballs(): return int\nget the number of trackballs on a Joystick"

#define DOC_JOYSTICKGETBALL "Joystick.get_ball(ball_number): return x, y\nget the relative position of a trackball"

#define DOC_JOYSTICKGETNUMBUTTONS "Joystick.get_numbuttons(): return int\nget the number of buttons on a Joystick"

#define DOC_JOYSTICKGETBUTTON "Joystick.get_button(button): return bool\nget the current button state"

#define DOC_JOYSTICKGETNUMHATS "Joystick.get_numhats(): return int\nget the number of hat controls on a Joystick"

#define DOC_JOYSTICKGETHAT "Joystick.get_hat(hat_number): return x, y\nget the position of a joystick hat"

#define DOC_PYGAMEKEY "pygame module to work with the keyboard"

#define DOC_PYGAMEKEYGETFOCUSED "pygame.key.get_focused(): return bool\ntrue if the display is receiving keyboard input from the system"

#define DOC_PYGAMEKEYGETPRESSED "pygame.key.get_pressed(): return bools\nget the state of all keyboard buttons"

#define DOC_PYGAMEKEYGETMODS "pygame.key.get_mods(): return int\ndetermine which modifier keys are being held"

#define DOC_PYGAMEKEYSETMODS "pygame.key.set_mods(int): return None\ntemporarily set which modifier keys are pressed"

#define DOC_PYGAMEKEYSETREPEAT "pygame.key.set_repeat(): return None\npygame.key.set_repeat(delay, interval): return None\ncontrol how held keys are repeated"

#define DOC_PYGAMEKEYGETREPEAT "pygame.key.get_repeat(): return (delay, interval)\nsee how held keys are repeated"

#define DOC_PYGAMEKEYNAME "pygame.key.name(key): return string\nget the name of a key identifier"

#define DOC_PYGAMELOCALS "pygame constants"

#define DOC_PYGAMEMASK "pygame module for image masks."

#define DOC_PYGAMEMASKFROMSURFACE "pygame.mask.from_surface(Surface, threshold = 127) -> Mask\nReturns a Mask from the given surface."

#define DOC_PYGAMEMASKFROMTHRESHOLD "pygame.mask.from_surface(Surface, color, threshold = (0,0,0,255), othersurface = None, palette_colors = 1) -> Mask\nCreates a mask by thresholding Surfaces"

#define DOC_PYGAMEMASKMASK "pygame.Mask((width, height)): return Mask\npygame object for representing 2d bitmasks"

#define DOC_MASKGETSIZE "Mask.get_size() -> width,height\nReturns the size of the mask."

#define DOC_MASKGETAT "Mask.get_at((x,y)) -> int\nReturns nonzero if the bit at (x,y) is set."

#define DOC_MASKSETAT "Mask.set_at((x,y),value)\nSets the position in the mask given by x and y."

#define DOC_MASKOVERLAP "Mask.overlap(othermask, offset) -> x,y\nReturns the point of intersection if the masks overlap with the given offset - or None if it does not overlap."

#define DOC_MASKOVERLAPAREA "Mask.overlap_area(othermask, offset) -> numpixels\nReturns the number of overlapping 'pixels'."

#define DOC_MASKOVERLAPMASK "Mask.overlap_mask(othermask, offset) -> Mask\nReturns a mask of the overlapping pixels"

#define DOC_MASKFILL "Mask.fill()\nSets all bits to 1"

#define DOC_MASKCLEAR "Mask.clear()\nSets all bits to 0"

#define DOC_MASKINVERT "Mask.invert()\nFlips the bits in a Mask"

#define DOC_MASKSCALE "Mask.scale((x, y)) -> Mask\nResizes a mask"

#define DOC_MASKDRAW "Mask.draw(othermask, offset)\nDraws a mask onto another"

#define DOC_MASKERASE "Mask.erase(othermask, offset)\nErases a mask from another"

#define DOC_MASKCOUNT "Mask.count() -> pixels\nReturns the number of set pixels"

#define DOC_MASKCENTROID "Mask.centroid() -> (x, y)\nReturns the centroid of the pixels in a Mask"

#define DOC_MASKANGLE "Mask.angle() -> theta\nReturns the orientation of the pixels"

#define DOC_MASKOUTLINE "Mask.outline(every = 1) -> [(x,y), (x,y) ...]\nlist of points outlining an object"

#define DOC_MASKCONVOLVE "Mask.convolve(othermask, outputmask = None, offset = (0,0)) -> Mask\nReturn the convolution of self with another mask."

#define DOC_MASKCONNECTEDCOMPONENT "Mask.connected_component((x,y) = None) -> Mask\nReturns a mask of a connected region of pixels."

#define DOC_MASKCONNECTEDCOMPONENTS "Mask.connected_components(min = 0) -> [Masks]\nReturns a list of masks of connected regions of pixels."

#define DOC_MASKGETBOUNDINGRECTS "Mask.get_bounding_rects() -> Rects\nReturns a list of bounding rects of regions of set pixels."

#define DOC_PYGAMEMIDI "pygame module for interacting with midi input and output."

#define DOC_PYGAMEMIDIINPUT "Input(device_id)\nInput(device_id, buffer_size)\nInput is used to get midi input from midi devices."

#define DOC_INPUTCLOSE "Input.close(): return None\n closes a midi stream, flushing any pending buffers."

#define DOC_INPUTPOLL "Input.poll(): return Bool\nreturns true if there's data, or false if not."

#define DOC_INPUTREAD "Input.read(num_events): return midi_event_list\nreads num_events midi events from the buffer."

#define DOC_PYGAMEMIDIMIDIEXCEPTION "MidiException(errno)\nexception that pygame.midi functions and classes can raise"

#define DOC_PYGAMEMIDIOUTPUT "Output(device_id)\nOutput(device_id, latency = 0)\nOutput(device_id, buffer_size = 4096)\nOutput(device_id, latency, buffer_size)\nOutput is used to send midi to an output device"

#define DOC_OUTPUTABORT "Output.abort(): return None\n terminates outgoing messages immediately"

#define DOC_OUTPUTCLOSE "Output.close(): return None\n closes a midi stream, flushing any pending buffers."

#define DOC_OUTPUTNOTEOFF "Output.note_off(note, velocity=None, channel = 0)\nturns a midi note off.  Note must be on."

#define DOC_OUTPUTNOTEON "Output.note_on(note, velocity=None, channel = 0)\nturns a midi note on.  Note must be off."

#define DOC_OUTPUTSETINSTRUMENT "Output.set_instrument(instrument_id, channel = 0)\nselect an instrument, with a value between 0 and 127"

#define DOC_OUTPUTWRITE "Output.write(data)\nwrites a list of midi data to the Output"

#define DOC_OUTPUTWRITESHORT "Output.write_short(status)\nOutput.write_short(status, data1 = 0, data2 = 0)\nwrite_short(status <, data1><, data2>)"

#define DOC_OUTPUTWRITESYSEX "Output.write_sys_ex(when, msg)\nwrites a timestamped system-exclusive midi message."

#define DOC_PYGAMEMIDIGETCOUNT "pygame.midi.get_count(): return num_devices\ngets the number of devices."

#define DOC_PYGAMEMIDIGETDEFAULTINPUTID "pygame.midi.get_default_input_id(): return default_id\ngets default input device number"

#define DOC_PYGAMEMIDIGETDEFAULTOUTPUTID "pygame.midi.get_default_output_id(): return default_id\ngets default output device number"

#define DOC_PYGAMEMIDIGETDEVICEINFO "pygame.midi.get_device_info(an_id): return (interf, name, input, output, opened)\n returns information about a midi device"

#define DOC_PYGAMEMIDIINIT "pygame.midi.init(): return None\ninitialize the midi module"

#define DOC_PYGAMEMIDIMIDIS2EVENTS "pygame.midi.midis2events(midis, device_id): return [Event, ...]\nconverts midi events to pygame events"

#define DOC_PYGAMEMIDIQUIT "pygame.midi.quit(): return None\nuninitialize the midi module"

#define DOC_PYGAMEMIDITIME "pygame.midi.time(): return time\nreturns the current time in ms of the PortMidi timer"

#define DOC_PYGAMEMIXER "pygame module for loading and playing sounds"

#define DOC_PYGAMEMIXERINIT "pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=4096): return None\ninitialize the mixer module"

#define DOC_PYGAMEMIXERPREINIT "pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffersize=4096): return None\npreset the mixer init arguments"

#define DOC_PYGAMEMIXERQUIT "pygame.mixer.quit(): return None\nuninitialize the mixer"

#define DOC_PYGAMEMIXERGETINIT "pygame.mixer.get_init(): return (frequency, format, channels)\ntest if the mixer is initialized"

#define DOC_PYGAMEMIXERSTOP "pygame.mixer.stop(): return None\nstop playback of all sound channels"

#define DOC_PYGAMEMIXERPAUSE "pygame.mixer.pause(): return None\ntemporarily stop playback of all sound channels"

#define DOC_PYGAMEMIXERUNPAUSE "pygame.mixer.unpause(): return None\nresume paused playback of sound channels"

#define DOC_PYGAMEMIXERFADEOUT "pygame.mixer.fadeout(time): return None\nfade out the volume on all sounds before stopping"

#define DOC_PYGAMEMIXERSETNUMCHANNELS "pygame.mixer.set_num_channels(count): return None\nset the total number of playback channels"

#define DOC_PYGAMEMIXERGETNUMCHANNELS "get the total number of playback channels"

#define DOC_PYGAMEMIXERSETRESERVED "pygame.mixer.set_reserved(count): return None\nreserve channels from being automatically used"

#define DOC_PYGAMEMIXERFINDCHANNEL "pygame.mixer.find_channel(force=False): return Channel\nfind an unused channel"

#define DOC_PYGAMEMIXERGETBUSY "pygame.mixer.get_busy(): return bool\ntest if any sound is being mixed"

#define DOC_PYGAMEMIXERSOUND "pygame.mixer.Sound(filename): return Sound\npygame.mixer.Sound(buffer): return Sound\npygame.mixer.Sound(object): return Sound\nCreate a new Sound object from a file"

#define DOC_SOUNDPLAY "Sound.play(loops=0, maxtime=0, fade_ms=0): return Channel\nbegin sound playback"

#define DOC_SOUNDSTOP "Sound.stop(): return None\nstop sound playback"

#define DOC_SOUNDFADEOUT "Sound.fadeout(time): return None\nstop sound playback after fading out"

#define DOC_SOUNDSETVOLUME "Sound.set_volume(value): return None\nset the playback volume for this Sound"

#define DOC_SOUNDGETVOLUME "Sound.get_volume(): return value\nget the playback volume"

#define DOC_SOUNDGETNUMCHANNELS "Sound.get_num_channels(): return count\ncount how many times this Sound is playing"

#define DOC_SOUNDGETLENGTH "Sound.get_length(): return seconds\nget the length of the Sound"

#define DOC_SOUNDGETBUFFER "Sound.get_buffer(): return BufferProxy\nacquires a buffer object for the sameples of the Sound."

#define DOC_PYGAMEMIXERCHANNEL "pygame.mixer.Channel(id): return Channel\nCreate a Channel object for controlling playback"

#define DOC_CHANNELPLAY "Channel.play(Sound, loops=0, maxtime=0, fade_ms=0): return None\nplay a Sound on a specific Channel"

#define DOC_CHANNELSTOP "Channel.stop(): return None\nstop playback on a Channel"

#define DOC_CHANNELPAUSE "Channel.pause(): return None\ntemporarily stop playback of a channel"

#define DOC_CHANNELUNPAUSE "Channel.unpause(): return None\nresume pause playback of a channel"

#define DOC_CHANNELFADEOUT "Channel.fadeout(time): return None\nstop playback after fading channel out"

#define DOC_CHANNELSETVOLUME "Channel.set_volume(value): return None\nChannel.set_volume(left, right): return None\nset the volume of a playing channel"

#define DOC_CHANNELGETVOLUME "Channel.get_volume(): return value\nget the volume of the playing channel"

#define DOC_CHANNELGETBUSY "Channel.get_busy(): return bool\ncheck if the channel is active"

#define DOC_CHANNELGETSOUND "Channel.get_sound(): return Sound\nget the currently playing Sound"

#define DOC_CHANNELQUEUE "Channel.queue(Sound): return None\nqueue a Sound object to follow the current"

#define DOC_CHANNELGETQUEUE "Channel.get_queue(): return Sound\nreturn any Sound that is queued"

#define DOC_CHANNELSETENDEVENT "Channel.set_endevent(): return None\nChannel.set_endevent(type): return None\nhave the channel send an event when playback stops"

#define DOC_CHANNELGETENDEVENT "Channel.get_endevent(): return type\nget the event a channel sends when playback stops"

#define DOC_PYGAMEMOUSE "pygame module to work with the mouse"

#define DOC_PYGAMEMOUSEGETPRESSED "pygame.moouse.get_pressed(): return (button1, button2, button3)\nget the state of the mouse buttons"

#define DOC_PYGAMEMOUSEGETPOS "pygame.mouse.get_pos(): return (x, y)\nget the mouse cursor position"

#define DOC_PYGAMEMOUSEGETREL "pygame.mouse.get_rel(): return (x, y)\nget the amount of mouse movement"

#define DOC_PYGAMEMOUSESETPOS "pygame.mouse.set_pos([x, y]): return None\nset the mouse cursor position"

#define DOC_PYGAMEMOUSESETVISIBLE "pygame.mouse.set_visible(bool): return bool\nhide or show the mouse cursor"

#define DOC_PYGAMEMOUSEGETFOCUSED "pygame.mouse.get_focused(): return bool\ncheck if the display is receiving mouse input"

#define DOC_PYGAMEMOUSESETCURSOR "pygame.mouse.set_cursor(size, hotspot, xormasks, andmasks): return None\nset the image for the system mouse cursor"

#define DOC_PYGAMEMOUSEGETCURSOR "pygame.mouse.get_cursor(): return (size, hotspot, xormasks, andmasks)\nget the image for the system mouse cursor"

#define DOC_PYGAMEMOVIE "pygame module for playback of mpeg video"

#define DOC_PYGAMEMOVIEMOVIE "pygame.movie.Movie(filename): return Movie\npygame.movie.Movie(object): return Movie\nload an mpeg movie file"

#define DOC_MOVIEPLAY "Movie.play(loops=0): return None\nstart playback of a movie"

#define DOC_MOVIESTOP "Movie.stop(): return None\nstop movie playback"

#define DOC_MOVIEPAUSE "Movie.pause(): return None\ntemporarily stop and resume playback"

#define DOC_MOVIESKIP "Movie.skip(seconds): return None\nadvance the movie playback position"

#define DOC_MOVIEREWIND "Movie.rewind(): return None\nrestart the movie playback"

#define DOC_MOVIERENDERFRAME "Movie.render_frame(frame_number): return frame_number\nset the current video frame"

#define DOC_MOVIEGETFRAME "Movie.get_frame(): return frame_number\nget the current video frame"

#define DOC_MOVIEGETTIME "Movie.get_time(): return seconds\nget the current vide playback time"

#define DOC_MOVIEGETBUSY "Movie.get_busy(): return bool\ncheck if the movie is currently playing"

#define DOC_MOVIEGETLENGTH "Movie.get_length(): return seconds\nthe total length of the movie in seconds"

#define DOC_MOVIEGETSIZE "Movie.get_size(): return (width, height)\nget the resolution of the video"

#define DOC_MOVIEHASVIDEO "Movie.get_video(): return bool\ncheck if the movie file contains video"

#define DOC_MOVIEHASAUDIO "Movie.get_audio(): return bool\ncheck if the movie file contains audio"

#define DOC_MOVIESETVOLUME "Movie.set_volume(value): return None\nset the audio playback volume"

#define DOC_MOVIESETDISPLAY "Movie.set_display(Surface, rect=None): return None\nset the video target Surface"

#define DOC_PYGAMEMIXERMUSIC "pygame module for controlling streamed audio"

#define DOC_PYGAMEMIXERMUSICLOAD "pygame.mixer.music.load(filename): return None\npygame.mixer.music.load(object): return None\nLoad a music file for playback"

#define DOC_PYGAMEMIXERMUSICPLAY "pygame.mixer.music.play(loops=0, start=0.0): return None\nStart the playback of the music stream"

#define DOC_PYGAMEMIXERMUSICREWIND "pygame.mixer.music.rewind(): return None\nrestart music"

#define DOC_PYGAMEMIXERMUSICSTOP "pygame.mixer.music.stop(): return None\nstop the music playback"

#define DOC_PYGAMEMIXERMUSICPAUSE "pygame.mixer.music.pause(): return None\ntemporarily stop music playback"

#define DOC_PYGAMEMIXERMUSICUNPAUSE "pygame.mixer.music.unpause(): return None\nresume paused music"

#define DOC_PYGAMEMIXERMUSICFADEOUT "pygame.mixer.music.fadeout(time): return None\nstop music playback after fading out"

#define DOC_PYGAMEMIXERMUSICSETVOLUME "pygame.mixer.music.set_volume(value): return None\nset the music volume"

#define DOC_PYGAMEMIXERMUSICGETVOLUME "pygame.mixer.music.get_volume(): return value\nget the music volume"

#define DOC_PYGAMEMIXERMUSICGETBUSY "pygame.mixer.music.get_busy(): return bool\ncheck if the music stream is playing"

#define DOC_PYGAMEMIXERMUSICGETPOS "pygame.mixer.music.get_pos(): return time\nget the music play time"

#define DOC_PYGAMEMIXERMUSICQUEUE "pygame.mixer.music.queue(filename): return None\nqueue a music file to follow the current"

#define DOC_PYGAMEMIXERMUSICSETENDEVENT "pygame.mixer.music.set_endevent(): return None\npygame.mixer.music.set_endevent(type): return None\nhave the music send an event when playback stops"

#define DOC_PYGAMEMIXERMUSICGETENDEVENT "pygame.mixer.music.get_endevent(): return type\nget the event a channel sends when playback stops"

#define DOC_PYGAMEOVERLAY "pygame.Overlay(format, (width, height)): return Overlay\npygame object for video overlay graphics"

#define DOC_OVERLAYDISPLAY "Overlay.display((y, u, v)): return None\nOverlay.display(): return None\nset the overlay pixel data"

#define DOC_OVERLAYSETLOCATION "Overlay.set_location(rect): return None\ncontrol where the overlay is displayed"

#define DOC_OVERLAYGETHARDWARE "Overlay.get_hardware(rect): return int\ntest if the Overlay is hardware accelerated"

#define DOC_PYGAMEPIXELARRAY "pygame.PixelArray(Surface): return PixelArray\npygame object for direct pixel access of surfaces"

#define DOC_PIXELARRAYSURFACE "PixelArray.surface: Return Surface\nGets the Surface the PixelArray uses."

#define DOC_PIXELARRAYMAKESURFACE "PixelArray.make_surface (): Return Surface\nCreates a new Surface from the current PixelArray."

#define DOC_PIXELARRAYREPLACE "PixelArray.replace (color, repcolor, distance=0, weights=(0.299, 0.587, 0.114)): Return None\nReplaces the passed color in the PixelArray with another one."

#define DOC_PIXELARRAYEXTRACT "PixelArray.extract (color, distance=0, weights=(0.299, 0.587, 0.114)): Return PixelArray\nExtracts the passed color from the PixelArray."

#define DOC_PIXELARRAYCOMPARE "PixelArray.compare (array, distance=0, weights=(0.299, 0.587, 0.114)): Return PixelArray\nCompares the PixelArray with another one."

#define DOC_PYGAMERECT "pygame.Rect(left, top, width, height): return Rect\npygame.Rect((left, top), (width, height)): return Rect\npygame.Rect(object): return Rect\npygame object for storing rectangular coordinates"

#define DOC_RECTCOPY "Rect.copy(): return Rect\ncopy the rectangle"

#define DOC_RECTMOVE "Rect.move(x, y): return Rect\nmoves the rectangle"

#define DOC_RECTMOVEIP "Rect.move_ip(x, y): return None\nmoves the rectangle, in place"

#define DOC_RECTINFLATE "Rect.inflate(x, y): return Rect\ngrow or shrink the rectangle size"

#define DOC_RECTINFLATEIP "Rect.inflate_ip(x, y): return None\ngrow or shrink the rectangle size, in place"

#define DOC_RECTCLAMP "Rect.clamp(Rect): return Rect\nmoves the rectangle inside another"

#define DOC_RECTCLAMPIP "Rect.clamp_ip(Rect): return None\nmoves the rectangle inside another, in place"

#define DOC_RECTCLIP "Rect.clip(Rect): return Rect\ncrops a rectangle inside another"

#define DOC_RECTUNION "Rect.union(Rect): return Rect\njoins two rectangles into one"

#define DOC_RECTUNIONIP "Rect.union_ip(Rect): return None\njoins two rectangles into one, in place"

#define DOC_RECTUNIONALL "Rect.unionall(Rect_sequence): return Rect\nthe union of many rectangles"

#define DOC_RECTUNIONALLIP "Rect.unionall_ip(Rect_sequence): return None\nthe union of many rectangles, in place"

#define DOC_RECTFIT "Rect.fit(Rect): return Rect\nresize and move a rectangle with aspect ratio"

#define DOC_RECTNORMALIZE "Rect.normalize(): return None\ncorrect negative sizes"

#define DOC_RECTCONTAINS "Rect.contains(Rect): return bool\ntest if one rectangle is inside another"

#define DOC_RECTCOLLIDEPOINT "Rect.collidepoint(x, y): return bool\nRect.collidepoint((x,y)): return bool\ntest if a point is inside a rectangle"

#define DOC_RECTCOLLIDERECT "Rect.colliderect(Rect): return bool\ntest if two rectangles overlap"

#define DOC_RECTCOLLIDELIST "Rect.collidelist(list): return index\ntest if one rectangle in a list intersects"

#define DOC_RECTCOLLIDELISTALL "Rect.collidelistall(list): return indices\ntest if all rectangles in a list intersect"

#define DOC_RECTCOLLIDEDICT "Rect.collidedict(dict): return (key, value)\ntest if one rectangle in a dictionary intersects"

#define DOC_RECTCOLLIDEDICTALL "Rect.collidedictall(dict): return [(key, value), ...]\ntest if all rectangles in a dictionary intersect"

#define DOC_PYGAMESCRAP "pygame module for clipboard support."

#define DOC_PYGAMESCRAPINIT "scrap.init () -> None\nInitializes the scrap module."

#define DOC_PYGAMESCRAPGET "scrap.get (type) -> string\nGets the data for the specified type from the clipboard."

#define DOC_PYGAMESCRAPGETTYPES "scrap.get_types () -> list\nGets a list of the available clipboard types."

#define DOC_PYGAMESCRAPPUT "scrap.put(type, data) -> None\nPlaces data into the clipboard."

#define DOC_PYGAMESCRAPCONTAINS "scrap.contains (type) -> bool\nChecks, whether a certain type is available in the clipboard."

#define DOC_PYGAMESCRAPLOST "scrap.lost() -> bool\nChecks whether the clipboard is currently owned by the application."

#define DOC_PYGAMESCRAPSETMODE "scrap.set_mode(mode) -> None\nSets the clipboard access mode."

#define DOC_PYGAMESNDARRAY "pygame module for accessing sound sample data"

#define DOC_PYGAMESNDARRAYARRAY "pygame.sndarray.array(Sound): return array\ncopy Sound samples into an array"

#define DOC_PYGAMESNDARRAYSAMPLES "pygame.sndarray.samples(Sound): return array\nreference Sound samples into an array"

#define DOC_PYGAMESNDARRAYMAKESOUND "pygame.sndarray.make_sound(array): return Sound\nconvert an array into a Sound object"

#define DOC_PYGAMESNDARRAYUSEARRAYTYPE "pygame.sndarray.use_arraytype (arraytype): return None\nSets the array system to be used for sound arrays"

#define DOC_PYGAMESNDARRAYGETARRAYTYPE "pygame.sndarray.get_arraytype (): return str\nGets the currently active array type."

#define DOC_PYGAMESNDARRAYGETARRAYTYPES "pygame.sndarray.get_arraytypes (): return tuple\nGets the array system types currently supported."

#define DOC_PYGAMESPRITE "pygame module with basic game object classes"

#define DOC_PYGAMESPRITESPRITE "pygame.sprite.Sprite(*groups): return Sprite\nsimple base class for visible game objects"

#define DOC_SPRITEUPDATE "Sprite.update(*args):\nmethod to control sprite behavior"

#define DOC_SPRITEADD "Sprite.add(*groups): return None\nadd the sprite to groups"

#define DOC_SPRITEREMOVE "Sprite.remove(*groups): return None\nremove the sprite from groups"

#define DOC_SPRITEKILL "Sprite.kill(): return None\nremove the Sprite from all Groups"

#define DOC_SPRITEALIVE "Sprite.alive(): return bool\ndoes the sprite belong to any groups"

#define DOC_SPRITEGROUPS "Sprite.groups(): return group_list\nlist of Groups that contain this Sprite"

#define DOC_PYGAMESPRITEDIRTYSPRITE "pygame.sprite.DirtySprite(*groups): return DirtySprite\na more featureful subclass of Sprite with more attributes"

#define DOC_ ""

#define DOC_PYGAMESPRITEGROUP "pygame.sprite.Group(*sprites): return Group\ncontainer class for many Sprites"

#define DOC_GROUPSPRITES "Group.sprites(): return sprite_list\nlist of the Sprites this Group contains"

#define DOC_GROUPCOPY "Group.copy(): return Group\nduplicate the Group"

#define DOC_GROUPADD "Group.add(*sprites): return None\nadd Sprites to this Group"

#define DOC_GROUPREMOVE "Group.remove(*sprites): return None\nremove Sprites from the Group"

#define DOC_GROUPHAS "Group.has(*sprites): return None\ntest if a Group contains Sprites"

#define DOC_GROUPUPDATE "Group.update(*args): return None\ncall the update method on contained Sprites"

#define DOC_GROUPDRAW "Group.draw(Surface): return None\nblit the Sprite images"

#define DOC_GROUPCLEAR "Group.clear(Surface_dest, background): return None\ndraw a background over the Sprites"

#define DOC_GROUPEMPTY "Group.empty(): return None\nremove all Sprites"

#define DOC_PYGAMESPRITERENDERUPDATES "pygame.sprite.RenderUpdates(*sprites): return RenderUpdates\nGroup class that tracks dirty updates"

#define DOC_RENDERUPDATESDRAW "RenderUpdates.draw(surface): return Rect_list\nblit the Sprite images and track changed areas"

#define DOC_PYGAMESPRITEORDEREDUPDATES "pygame.sprite.OrderedUpdates(*spites): return OrderedUpdates\nRenderUpdates class that draws Sprites in order of addition"

#define DOC_PYGAMESPRITELAYEREDUPDATES "pygame.sprite.LayeredUpdates(*spites, **kwargs): return LayeredUpdates\nLayeredUpdates Group handles layers, that draws like OrderedUpdates."

#define DOC_LAYEREDUPDATESADD "LayeredUpdates.add(*sprites, **kwargs): return None\nadd a sprite or sequence of sprites to a group"

#define DOC_LAYEREDUPDATESSPRITES "LayeredUpdates.sprites(): return sprites\nreturns a ordered list of sprites (first back, last top)."

#define DOC_LAYEREDUPDATESDRAW "LayeredUpdates.draw(surface): return Rect_list\ndraw all sprites in the right order onto the passed surface."

#define DOC_LAYEREDUPDATESGETSPRITESAT "LayeredUpdates.get_sprites_at(pos): return colliding_sprites\nreturns a list with all sprites at that position."

#define DOC_LAYEREDUPDATESGETSPRITE "LayeredUpdates.get_sprite(idx): return sprite\nreturns the sprite at the index idx from the groups sprites"

#define DOC_LAYEREDUPDATESREMOVESPRITESOFLAYER "LayeredUpdates.remove_sprites_of_layer(layer_nr): return sprites\nremoves all sprites from a layer and returns them as a list."

#define DOC_LAYEREDUPDATESLAYERS "LayeredUpdates.layers(): return layers\nreturns a list of layers defined (unique), sorted from botton up."

#define DOC_LAYEREDUPDATESCHANGELAYER "LayeredUpdates.change_layer(sprite, new_layer): return None\nchanges the layer of the sprite"

#define DOC_LAYEREDUPDATESGETLAYEROFSPRITE "LayeredUpdates.get_layer_of_sprite(sprite): return layer\nreturns the layer that sprite is currently in."

#define DOC_LAYEREDUPDATESGETTOPLAYER "LayeredUpdates.get_top_layer(): return layer\nreturns the top layer"

#define DOC_LAYEREDUPDATESGETBOTTOMLAYER "LayeredUpdates.get_bottom_layer(): return layer\nreturns the bottom layer"

#define DOC_LAYEREDUPDATESMOVETOFRONT "LayeredUpdates.move_to_front(sprite): return None\nbrings the sprite to front layer"

#define DOC_LAYEREDUPDATESMOVETOBACK "LayeredUpdates.move_to_back(sprite): return None\nmoves the sprite to the bottom layer"

#define DOC_LAYEREDUPDATESGETTOPSPRITE "LayeredUpdates.get_top_sprite(): return Sprite\nreturns the topmost sprite"

#define DOC_LAYEREDUPDATESGETSPRITESFROMLAYER "LayeredUpdates.get_sprites_from_layer(layer): return sprites\nreturns all sprites from a layer, ordered by how they where added"

#define DOC_LAYEREDUPDATESSWITCHLAYER "LayeredUpdates.switch_layer(layer1_nr, layer2_nr): return None\nswitches the sprites from layer1 to layer2"

#define DOC_PYGAMESPRITELAYEREDDIRTY "pygame.sprite.LayeredDirty(*spites, **kwargs): return LayeredDirty\nLayeredDirty Group is for DirtySprites.  Subclasses LayeredUpdates."

#define DOC_LAYEREDDIRTYDRAW "LayeredDirty.draw(surface, bgd=None): return Rect_list\ndraw all sprites in the right order onto the passed surface."

#define DOC_LAYEREDDIRTYCLEAR "LayeredDirty.clear(surface, bgd): return None\nused to set background"

#define DOC_LAYEREDDIRTYREPAINTRECT "LayeredDirty.repaint_rect(screen_rect): return None\nrepaints the given area"

#define DOC_LAYEREDDIRTYSETCLIP "LayeredDirty.set_clip(screen_rect=None): return None\nclip the area where to draw. Just pass None (default) to reset the clip"

#define DOC_LAYEREDDIRTYGETCLIP "LayeredDirty.get_clip(): return Rect\nclip the area where to draw. Just pass None (default) to reset the clip"

#define DOC_LAYEREDDIRTYCHANGELAYER "change_layer(sprite, new_layer): return None\nchanges the layer of the sprite"

#define DOC_LAYEREDDIRTYSETTIMINGTRESHOLD "set_timing_treshold(time_ms): return None\nsets the treshold in milliseconds"

#define DOC_PYGAMESPRITEGROUPSINGLE "pygame.sprite.GroupSingle(sprite=None): return GroupSingle\nGroup container that holds a single Sprite"

#define DOC_PYGAMESPRITESPRITECOLLIDE "pygame.sprite.spritecollide(sprite, group, dokill, collided = None): return Sprite_list\nfind Sprites in a Group that intersect another Sprite"

#define DOC_PYGAMESPRITECOLLIDERECT "pygame.sprite.collide_rect(left, right): return bool\ncollision detection between two sprites, using rects."

#define DOC_PYGAMESPRITECOLLIDERECTRATIO "pygame.sprite.collide_rect_ratio(ratio): return collided_callable\ncollision detection between two sprites, using rects scaled to a ratio."

#define DOC_PYGAMESPRITECOLLIDECIRCLE "pygame.sprite.collide_circle(left, right): return bool\ncollision detection between two sprites, using circles."

#define DOC_PYGAMESPRITECOLLIDECIRCLERATIO "pygame.sprite.collide_circle_ratio(ratio): return collided_callable\ncollision detection between two sprites, using circles scaled to a ratio."

#define DOC_PYGAMESPRITECOLLIDEMASK "pygame.sprite.collide_mask(SpriteLeft, SpriteRight): return bool\ncollision detection between two sprites, using masks."

#define DOC_PYGAMESPRITEGROUPCOLLIDE "pygame.sprite.groupcollide(group1, group2, dokill1, dokill2): return Sprite_dict\nfind all Sprites that collide between two Groups"

#define DOC_PYGAMESPRITESPRITECOLLIDEANY "pygame.sprite.spritecollideany(sprite, group): return bool\nsimple test if a Sprite intersects anything in a Group"

#define DOC_ ""

#define DOC_PYGAMESURFACE "pygame.Surface((width, height), flags=0, depth=0, masks=None): return Surface\npygame.Surface((width, height), flags=0, Surface): return Surface\npygame object for representing images"

#define DOC_SURFACEBLIT "Surface.blit(source, dest, area=None, special_flags = 0): return Rect\ndraw one image onto another"

#define DOC_SURFACECONVERT "Surface.convert(Surface): return Surface\nSurface.convert(depth, flags=0): return Surface\nSurface.convert(masks, flags=0): return Surface\nSurface.convert(): return Surface\nchange the pixel format of an image"

#define DOC_SURFACECONVERTALPHA "Surface.convert_alpha(Surface): return Surface\nSurface.convert_alpha(): return Surface\nchange the pixel format of an image including per pixel alphas"

#define DOC_SURFACECOPY "Surface.copy(): return Surface\ncreate a new copy of a Surface"

#define DOC_SURFACEFILL "Surface.fill(color, rect=None, special_flags=0): return Rect\nfill Surface with a solid color"

#define DOC_SURFACESCROLL "Surface.scroll(dx=0, dy=0): return None\nShift the surface image in place"

#define DOC_SURFACESETCOLORKEY "Surface.set_colorkey(Color, flags=0): return None\nSurface.set_colorkey(None): return None\nSet the transparent colorkey"

#define DOC_SURFACEGETCOLORKEY "Surface.get_colorkey(): return RGB or None\nGet the current transparent colorkey"

#define DOC_SURFACESETALPHA "Surface.set_alpha(value, flags=0): return None\nSurface.set_alpha(None): return None\nset the alpha value for the full Surface image"

#define DOC_SURFACEGETALPHA "Surface.get_alpha(): return int_value or None\nget the current Surface transparency value"

#define DOC_SURFACELOCK "Surface.lock(): return None\nlock the Surface memory for pixel access"

#define DOC_SURFACEUNLOCK "Surface.unlock(): return None\nunlock the Surface memory from pixel access"

#define DOC_SURFACEMUSTLOCK "Surface.mustlock(): return bool\ntest if the Surface requires locking"

#define DOC_SURFACEGETLOCKED "Surface.get_locked(): return bool\ntest if the Surface is current locked"

#define DOC_SURFACEGETLOCKS "Surface.get_locks(): return tuple\nGets the locks for the Surface"

#define DOC_SURFACEGETAT "Surface.get_at((x, y)): return Color\nget the color value at a single pixel"

#define DOC_SURFACESETAT "Surface.set_at((x, y), Color): return None\nset the color value for a single pixel"

#define DOC_SURFACEGETPALETTE "Surface.get_palette(): return [RGB, RGB, RGB, ...]\nget the color index palette for an 8bit Surface"

#define DOC_SURFACEGETPALETTEAT "Surface.get_palette_at(index): return RGB\nget the color for a single entry in a palette"

#define DOC_SURFACESETPALETTE "Surface.set_palette([RGB, RGB, RGB, ...]): return None\nset the color palette for an 8bit Surface"

#define DOC_SURFACESETPALETTEAT "Surface.set_at(index, RGB): return None\nset the color for a single index in an 8bit Surface palette"

#define DOC_SURFACEMAPRGB "Surface.map_rgb(Color): return mapped_int\nconvert a color into a mapped color value"

#define DOC_SURFACEUNMAPRGB "Surface.map_rgb(mapped_int): return Color\nconvert a mapped integer color value into a Color"

#define DOC_SURFACESETCLIP "Surface.set_clip(rect): return None\nSurface.set_clip(None): return None\nset the current clipping area of the Surface"

#define DOC_SURFACEGETCLIP "Surface.get_clip(): return Rect\nget the current clipping area of the Surface"

#define DOC_SURFACESUBSURFACE "Surface.subsurface(Rect): return Surface\ncreate a new surface that references its parent"

#define DOC_SURFACEGETPARENT "Surface.get_parent(): return Surface\nfind the parent of a subsurface"

#define DOC_SURFACEGETABSPARENT "Surface.get_abs_parent(): return Surface\nfind the top level parent of a subsurface"

#define DOC_SURFACEGETOFFSET "Surface.get_offset(): return (x, y)\nfind the position of a child subsurface inside a parent"

#define DOC_SURFACEGETABSOFFSET "Surface.get_abs_offset(): return (x, y)\nfind the absolute position of a child subsurface inside its top level parent"

#define DOC_SURFACEGETSIZE "Surface.get_size(): return (width, height)\nget the dimensions of the Surface"

#define DOC_SURFACEGETWIDTH "Surface.get_width(): return width\nget the width of the Surface"

#define DOC_SURFACEGETHEIGHT "Surface.get_height(): return height\nget the height of the Surface"

#define DOC_SURFACEGETRECT "Surface.get_rect(**kwargs): return Rect\nget the rectangular area of the Surface"

#define DOC_SURFACEGETBITSIZE "Surface.get_bitsize(): return int\nget the bit depth of the Surface pixel format"

#define DOC_SURFACEGETBYTESIZE "Surface.get_bytesize(): return int\nget the bytes used per Surface pixel"

#define DOC_SURFACEGETFLAGS "Surface.get_flags(): return int\nget the additional flags used for the Surface"

#define DOC_SURFACEGETPITCH "Surface.get_pitch(): return int\nget the number of bytes used per Surface row"

#define DOC_SURFACEGETMASKS "Surface.get_masks(): return (R, G, B, A)\nthe bitmasks needed to convert between a color and a mapped integer"

#define DOC_SURFACESETMASKS "Surface.set_masks((r,g,b,a)): return None\nset the bitmasks needed to convert between a color and a mapped integer"

#define DOC_SURFACEGETSHIFTS "Surface.get_shifts(): return (R, G, B, A)\nthe bit shifts needed to convert between a color and a mapped integer"

#define DOC_SURFACESETSHIFTS "Surface.get_shifts((r,g,b,a)): return None\nsets the bit shifts needed to convert between a color and a mapped integer"

#define DOC_SURFACEGETLOSSES "Surface.get_losses(): return (R, G, B, A)\nthe significant bits used to convert between a color and a mapped integer"

#define DOC_SURFACEGETBOUNDINGRECT "Surface.get_bounding_rect(min_alpha = 1): return Rect\nfind the smallest rect containing data"

#define DOC_SURFACEGETBUFFER "Surface.get_buffer(): return BufferProxy\nacquires a buffer object for the pixels of the Surface."

#define DOC_PYGAMESURFARRAY "pygame module for accessing surface pixel data using array interfaces"

#define DOC_PYGAMESURFARRAYARRAY2D "pygame.surfarray.array2d(Surface): return array\nCopy pixels into a 2d array"

#define DOC_PYGAMESURFARRAYPIXELS2D "pygame.surfarray.pixels2d(Surface): return array\nReference pixels into a 2d array"

#define DOC_PYGAMESURFARRAYARRAY3D "pygame.surfarray.array3d(Surface): return array\nCopy pixels into a 3d array"

#define DOC_PYGAMESURFARRAYPIXELS3D "pygame.surfarray.pixels3d(Surface): return array\nReference pixels into a 3d array"

#define DOC_PYGAMESURFARRAYARRAYALPHA "pygame.surfarray.array_alpha(Surface): return array\nCopy pixel alphas into a 2d array"

#define DOC_PYGAMESURFARRAYPIXELSALPHA "pygame.surfarray.pixels_alpha(Surface): return array\nReference pixel alphas into a 2d array"

#define DOC_PYGAMESURFARRAYARRAYCOLORKEY "pygame.surfarray.array_colorkey(Surface): return array\nCopy the colorkey values into a 2d array"

#define DOC_PYGAMESURFARRAYMAKESURFACE "pygame.surfarray.make_surface(array): return Surface\nCopy an array to a new surface"

#define DOC_PYGAMESURFARRAYBLITARRAY "pygame.surfarray.blit_array(Surface, array): return None\nBlit directly from a array values"

#define DOC_PYGAMESURFARRAYMAPARRAY "pygame.surfarray.map_array(Surface, array3d): return array2d\nMap a 3d array into a 2d array"

#define DOC_PYGAMESURFARRAYUSEARRAYTYPE "pygame.surfarray.use_arraytype (arraytype): return None\nSets the array system to be used for surface arrays"

#define DOC_PYGAMESURFARRAYGETARRAYTYPE "pygame.surfarray.get_arraytype (): return str\nGets the currently active array type."

#define DOC_PYGAMESURFARRAYGETARRAYTYPES "pygame.surfarray.get_arraytypes (): return tuple\nGets the array system types currently supported."

#define DOC_PYGAMETESTS "Pygame unit test suite package"

#define DOC_PYGAMETESTSRUN "pygame.tests.run(*args, **kwds): return tuple\nRun the Pygame unit test suite"

#define DOC_PYGAMETIME "pygame module for monitoring time"

#define DOC_PYGAMETIMEGETTICKS "pygame.time.get_ticks(): return milliseconds\nget the time in milliseconds"

#define DOC_PYGAMETIMEWAIT "pygame.time.wait(milliseconds): return time\npause the program for an amount of time"

#define DOC_PYGAMETIMEDELAY "pygame.time.delay(milliseconds): return time\npause the program for an amount of time"

#define DOC_PYGAMETIMESETTIMER "pygame.time.set_timer(eventid, milliseconds): return None\nrepeatedly create an event on the event queue"

#define DOC_PYGAMETIMECLOCK "pygame.time.Clock(): return Clock\ncreate an object to help track time"

#define DOC_CLOCKTICK "Clock.tick(framerate=0): return milliseconds\ncontrol timer events\nupdate the clock"

#define DOC_CLOCKTICKBUSYLOOP "Clock.tick_busy_loop(framerate=0): return milliseconds\ncontrol timer events\nupdate the clock"

#define DOC_CLOCKGETTIME "Clock.get_time(): return milliseconds\ntime used in the previous tick"

#define DOC_CLOCKGETRAWTIME "Clock.get_rawtime(): return milliseconds\nactual time used in the previous tick"

#define DOC_CLOCKGETFPS "Clock.get_fps(): return float\ncompute the clock framerate"

#define DOC_PYGAMETRANSFORM "pygame module to transform surfaces"

#define DOC_PYGAMETRANSFORMFLIP "pygame.transform.flip(Surface, xbool, ybool): return Surface\nflip vertically and horizontally"

#define DOC_PYGAMETRANSFORMSCALE "pygame.transform.scale(Surface, (width, height), DestSurface = None): return Surface\nresize to new resolution"

#define DOC_PYGAMETRANSFORMROTATE "pygame.transform.rotate(Surface, angle): return Surface\nrotate an image"

#define DOC_PYGAMETRANSFORMROTOZOOM "pygame.transform.rotozoom(Surface, angle, scale): return Surface\nfiltered scale and rotation"

#define DOC_PYGAMETRANSFORMSCALE2X "pygame.transform.scale2x(Surface, DestSurface = None): Surface\nspecialized image doubler"

#define DOC_PYGAMETRANSFORMSMOOTHSCALE "pygame.transform.smoothscale(Surface, (width, height), DestSurface = None): return Surface\nscale a surface to an arbitrary size smoothly"

#define DOC_PYGAMETRANSFORMGETSMOOTHSCALEBACKEND "pygame.transform.get_smoothscale_backend(): return String\nreturn smoothscale filter version in use: 'GENERIC', 'MMX', or 'SSE'"

#define DOC_PYGAMETRANSFORMSETSMOOTHSCALEBACKEND "pygame.transform.get_smoothscale_backend(type): return None\nset smoothscale filter version to one of: 'GENERIC', 'MMX', or 'SSE'"

#define DOC_PYGAMETRANSFORMCHOP "pygame.transform.chop(Surface, rect): return Surface\ngets a copy of an image with an interior area removed"

#define DOC_PYGAMETRANSFORMLAPLACIAN "pygame.transform.laplacian(Surface, DestSurface = None): return Surface\nfind edges in a surface"

#define DOC_PYGAMETRANSFORMAVERAGESURFACES "pygame.transform.average_surfaces(Surfaces, DestSurface = None, palette_colors = 1): return Surface\nfind the average surface from many surfaces."

#define DOC_PYGAMETRANSFORMAVERAGECOLOR "pygame.transform.average_color(Surface, Rect = None): return Color\nfinds the average color of a surface"

#define DOC_PYGAMETRANSFORMTHRESHOLD "pygame.transform.threshold(DestSurface, Surface, color, threshold = (0,0,0,0), diff_color = (0,0,0,0), change_return = 1, Surface = None, inverse = False): return num_threshold_pixels\nfinds which, and how many pixels in a surface are within a threshold of a color."

#define DOC_PYGAMEVECTOR2 "pygame.math.Vector2(x=0, y=0): return Vector2\nA two-dimensional vector"
#define DOC_PYGAMEVECTOR3 "pygame.math.Vector3(x=0, y=0, z=0): return Vector3\nA three-dimensional vector"

/* Docs in a comments... slightly easier to read. */


/*

pygame
 the top level pygame package



pygame.init
 pygame.init(): return (numpass, numfail)
initialize all imported pygame modules



pygame.quit
 pygame.quit(): return None
uninitialize all pygame modules



pygame.error
 raise pygame.error, message
standard pygame exception



pygame.get_error
 pygame.get_error(): return errorstr
get the current error message



pygame.set_error
 pygame.set_error(error_msg): return None
set the current error message



pygame.get_sdl_version
 pygame.get_sdl_version(): return major, minor, patch
get the version number of SDL



pygame.get_sdl_byteorder
 pygame.get_sdl_byteorder(): return int
get the byte order of SDL



pygame.register_quit
 register_quit(callable): return None
register a function to be called when pygame quits



pygame.version
 module pygame.version
small module containing version information



pygame.version.ver
 pygame.version.ver = '1.2'
version number as a string



pygame.version.vernum
 pygame.version.vernum = (1, 5, 3)
tupled integers of the version



pygame.camera
 pygame module for camera use



pygame.camera.colorspace
 pygame.camera.colorspace(Surface, format, DestSurface = None): return Surface
Surface colorspace conversion



pygame.camera.list_cameras
 pygame.camera.list_cameras(): return [cameras]
returns a list of available cameras



pygame.camera.Camera
 pygame.camera.Camera(device, (width, height), format): return Camera
load a camera



Camera.start
 Camera.start(): return None
opens, initializes, and starts capturing



Camera.stop
 Camera.stop(): return None
stops, uninitializes, and closes the camera



Camera.get_controls
 Camera.get_controls(): return (hflip = bool, vflip = bool, brightness)
gets current values of user controls



Camera.set_controls
 Camera.set_controls(hflip = bool, vflip = bool, brightness): return (hflip = bool, vflip = bool, brightness)
changes camera settings if supported by the camera



Camera.get_size
 Camera.get_size(): return (width, height)
returns the dimensions of the images being recorded



Camera.query_image
 Camera.query_image(): return bool
checks if a frame is ready



Camera.get_image
 Camera.get_image(Surface = None): return Surface
captures an image as a Surface



Camera.get_raw
 Camera.get_raw(): return string
returns an unmodified image as a string



pygame.cdrom
 pygame module for audio cdrom control



pygame.cdrom.init
 pygame.cdrom.init(): return None
initialize the cdrom module



pygame.cdrom.quit
 pygame.cdrom.quit(): return None
uninitialize the cdrom module



pygame.cdrom.get_init
 pygame.cdrom.get_init(): return bool
true if the cdrom module is initialized



pygame.cdrom.get_count
 pygame.cdrom.get_count(): return count
number of cd drives on the system



pygame.cdrom.CD
 pygame.cdrom.CD(id): return CD
class to manage a cdrom drive



CD.init
 CD.init(): return None
initialize a cdrom drive for use



CD.quit
 CD.quit(): return None
uninitialize a cdrom drive for use



CD.get_init
 CD.get_init(): return bool
true if this cd device initialized



CD.play
 CD.play(track, start=None, end=None): return None
start playing audio



CD.stop
 CD.stop(): return None
stop audio playback



CD.pause
 CD.pause(): return None
temporarily stop audio playback



CD.resume
 CD.resume(): return None
unpause audio playback



CD.eject
 CD.eject(): return None
eject or open the cdrom drive



CD.get_id
 CD.get_id(): return id
the index of the cdrom drive



CD.get_name
 CD.get_name(): return name
the system name of the cdrom drive



CD.get_busy
 CD.get_busy(): return bool
true if the drive is playing audio



CD.get_paused
 CD.get_paused(): return bool
true if the drive is paused



CD.get_current
 CD.get_current(): return track, seconds
the current audio playback position



CD.get_empty
 CD.get_empty(): return bool
False if a cdrom is in the drive



CD.get_numtracks
 CD.get_numtracks(): return count
the number of tracks on the cdrom



CD.get_track_audio
 CD.get_track_audio(track): return bool
true if the cdrom track has audio data



CD.get_all
 CD.get_all(): return [(audio, start, end, lenth), ...]
get all track information



CD.get_track_start
 CD.get_track_start(track): return seconds
start time of a cdrom track



CD.get_track_length
 CD.get_track_length(track): return seconds
length of a cdrom track



pygame.Color
 pygame.Color(name): Return Color
pygame.Color(r, g, b, a): Return Color
pygame.Color(rgbvalue): Return Color
pygame object for color representations



Color.r
 Color.r: Return int
Gets or sets the red value of the Color.



Color.g
 Color.g: Return int
Gets or sets the green value of the Color.



Color.b
 Color.b: Return int
Gets or sets the blue value of the Color.



Color.a
 Color.a: Return int
Gets or sets the alpha value of the Color.



Color.cmy
 Color.cmy: Return tuple
Gets or sets the CMY representation of the Color.



Color.hsva
 Color.hsva: Return tuple
Gets or sets the HSVA representation of the Color.



Color.hsla
 Color.hsla: Return tuple
Gets or sets the HSLA representation of the Color.



Color.i1i2i3
 Color.i1i2i3: Return tuple
Gets or sets the I1I2I3 representation of the Color.



Color.normalize
 Color.normalize(): Return tuple
Returns the normalized RGBA values of the Color.



Color.correct_gamma
 Color.correct_gamma (gamma): Return Color
Applies a certain gamma value to the Color.



Color.set_length
 Color.set_length(len)
Set the number of elements in the Color to 1,2,3, or 4.



pygame.cursors
 pygame module for cursor resources



pygame.cursors.compile
 pygame.cursor.compile(strings, black='X', white='.', xor='o'): return data, mask
create binary cursor data from simple strings



pygame.cursors.load_xbm
 pygame.cursors.load_xbm(cursorfile): return cursor_args
pygame.cursors.load_xbm(cursorfile, maskfile): return cursor_args
load cursor data from an xbm file



pygame.display
 pygame module to control the display window and screen



pygame.display.init
 pygame.display.init(): return None
initialize the display module



pygame.display.quit
 pygame.display.quit(): return None
uninitialize the display module



pygame.display.get_init
 pygame.display.get_init(): return bool
true if the display module is initialized



pygame.display.set_mode
 pygame.display.set_mode(resolution=(0,0), flags=0, depth=0): return Surface
initialize a window or screen for display



pygame.display.get_surface
 pygame.display.get_surface(): return Surface
get a reference to the currently set display surface



pygame.display.flip
 pygame.display.flip(): return None
update the full display Surface to the screen



pygame.display.update
 pygame.display.update(rectangle=None): return None
pygame.display.update(rectangle_list): return None
update portions of the screen for software displays



pygame.display.get_driver
 pygame.display.get_driver(): return name
get the name of the pygame display backend



pygame.display.Info
 pygame.display.Info(): return VideoInfo
Create a video display information object



pygame.display.get_wm_info
 pygame.display.get_wm_info(): return dict
Get information about the current windowing system



pygame.display.list_modes
 pygame.display.list_modes(depth=0, flags=pygame.FULLSCREEN): return list
get list of available fullscreen modes



pygame.display.mode_ok
 pygame.display.mode_ok(size, flags=0, depth=0): return depth
pick the best color depth for a display mode



pygame.display.gl_get_attribute
 pygame.display.gl_get_attribute(flag): return value
get the value for an opengl flag for the current display



pygame.display.gl_set_attribute
 pygame.display.gl_set_attribute(flag, value): return None
request an opengl display attribute for the display mode



pygame.display.get_active
 pygame.display.get_active(): return bool
true when the display is active on the display



pygame.display.iconify
 pygame.display.iconify(): return bool
iconify the display surface



pygame.display.toggle_fullscreen
 pygame.display.toggle_fullscreen(): return bool
switch between fullscreen and windowed displays



pygame.display.set_gamma
 pygame.display.set_gamma(red, green=None, blue=None): return bool
change the hardware gamma ramps



pygame.display.set_gamma_ramp
 change the hardware gamma ramps with a custom lookup
pygame.display.set_gamma_ramp(red, green, blue): return bool
set_gamma_ramp(red, green, blue): return bool



pygame.display.set_icon
 pygame.display.set_icon(Surface): return None
change the system image for the display window



pygame.display.set_caption
 pygame.display.set_caption(title, icontitle=None): return None
set the current window caption



pygame.display.get_caption
 pygame.display.get_caption(): return (title, icontitle)
get the current window caption



pygame.display.set_palette
 pygame.display.set_palette(palette=None): return None
set the display color palette for indexed displays



pygame.draw
 pygame module for drawing shapes



pygame.draw.rect
 pygame.draw.rect(Surface, color, Rect, width=0): return Rect
draw a rectangle shape



pygame.draw.polygon
 pygame.draw.polygon(Surface, color, pointlist, width=0): return Rect
draw a shape with any number of sides



pygame.draw.circle
 pygame.draw.circle(Surface, color, pos, radius, width=0): return Rect
draw a circle around a point



pygame.draw.ellipse
 pygame.draw.ellipse(Surface, color, Rect, width=0): return Rect
draw a round shape inside a rectangle



pygame.draw.arc
 pygame.draw.arc(Surface, color, Rect, start_angle, stop_angle, width=1): return Rect
draw a partial section of an ellipse



pygame.draw.line
 pygame.draw.line(Surface, color, start_pos, end_pos, width=1): return Rect
draw a straight line segment



pygame.draw.lines
 pygame.draw.lines(Surface, color, closed, pointlist, width=1): return Rect
draw multiple contiguous line segments



pygame.draw.aaline
 pygame.draw.aaline(Surface, color, startpos, endpos, blend=1): return Rect
draw fine antialiased lines



pygame.draw.aalines
 pygame.draw.aalines(Surface, color, closed, pointlist, blend=1): return Rect



pygame.event
 pygame module for interacting with events and queues



pygame.event.pump
 pygame.event.pump(): return None
internally process pygame event handlers



pygame.event.get
 pygame.event.get(): return Eventlist
pygame.event.get(type): return Eventlist
pygame.event.get(typelist): return Eventlist
get events from the queue



pygame.event.poll
 pygame.event.poll(): return Event
get a single event from the queue



pygame.event.wait
 pygame.event.wait(): return Event
wait for a single event from the queue



pygame.event.peek
 pygame.event.peek(type): return bool
pygame.event.peek(typelist): return bool
test if event types are waiting on the queue



pygame.event.clear
 pygame.event.clear(): return None
pygame.event.clear(type): return None
pygame.event.clear(typelist): return None
remove all events from the queue



pygame.event.event_name
 pygame.event.event_name(type): return string
get the string name from and event id



pygame.event.set_blocked
 pygame.event.set_blocked(type): return None
pygame.event.set_blocked(typelist): return None
pygame.event.set_blocked(None): return None
control which events are allowed on the queue



pygame.event.set_allowed
 pygame.event.set_allowed(type): return None
pygame.event.set_allowed(typelist): return None
pygame.event.set_allowed(None): return None
control which events are allowed on the queue



pygame.event.get_blocked
 pygame.event.get_blocked(type): return bool
test if a type of event is blocked from the queue



pygame.event.set_grab
 pygame.event.set_grab(bool): return None
control the sharing of input devices with other applications



pygame.event.get_grab
 pygame.event.get_grab(): return bool
test if the program is sharing input devices



pygame.event.post
 pygame.event.post(Event): return None
place a new event on the queue



pygame.event.Event
 pygame.event.Event(type, dict): return Event
pygame.event.Event(type, **attributes): return Event
create a new event object



pygame.examples
 module of example programs



pygame.examples.aliens.main
 pygame.aliens.main(): return None
play the full aliens example



pygame.examples.oldalien.main
 pygame.examples.oldalien.main(): return None
play the original aliens example



pygame.examples.stars.main
 pygame.examples.stars.main(): return None
run a simple starfield example



pygame.examples.chimp.main
 pygame.examples.chimp.main(): return None
hit the moving chimp



pygame.examples.moveit.main
 pygame.examples.moveit.main(): return None
display animated objects on the screen



pygame.examples.fonty.main
 pygame.examples.fonty.main(): return None
run a font rendering example



pygame.examples.vgrade.main
 pygame.examples.vgrade.main(): return None
display a vertical gradient



pygame.examples.eventlist.main
 pygame.examples.eventlist.main(): return None
display pygame events



pygame.examples.arraydemo.main
 pygame.examples.arraydemo.main(arraytype=None): return None
show various surfarray effects



pygame.examples.sound.main
 pygame.examples.sound.main(file_path=None): return None
load and play a sound



pygame.examples.sound_array_demos.main
 pygame.examples.sound_array_demos.main(arraytype=None): return None
play various sndarray effects



pygame.examples.liquid.main
 pygame.examples.liquid.main(): return None
display an animated liquid effect



pygame.examples.glcube.main
 pygame.examples.glcube.main(): return None
display an animated 3D cube using OpenGL



pygame.examples.scrap_clipboard.main
 pygame.examples.scrap_clipboard.main(): return None
access the clipboard



pygame.examples.mask.main
 pygame.examples.mask.main(*args): return None
display multiple images bounce off each other using collision detection



pygame.examples.testsprite.main
 pygame.examples.testsprite.main(update_rects = True, use_static = False, use_FastRenderGroup = False, screen_dims = [640, 480], use_alpha = False, flags = 0): return None
show lots of sprites moving around



pygame.examples.headless_no_windows_needed.main
 pygame.examples.headless_no_windows_needed.main(fin, fout, w, h): return None
write an image file that is smoothscaled copy of an input file



pygame.examples.fastevents.main
 pygame.examples.fastevents.main(): return None
stress test the fastevents module



pygame.examples.overlay.main
 pygame.examples.overlay.main(fname): return None
play a .pgm video using overlays



pygame.examples.blend_fill.main
 pygame.examples.blend_fill.main(): return None
demonstrate the various surface.fill method blend options



pygame.examples.cursors.main
 pygame.examples.cursors.main(): return None
display two different custom cursors



pygame.examples.pixelarray.main
 pygame.examples.pixelarray.main(): return None
display various pixelarray generated effects



pygame.examples.scaletest.main
 pygame.examples.scaletest.main(imagefile, convert_alpha=False, run_speed_test=True): return None
interactively scale an image using smoothscale



pygame.examples.midi.main
 pygame.examples.midi.main(mode='output', device_id=None): return None
run a midi example



pygame.examples.scroll.main
 pygame.examples.scroll.main(image_file=None): return None
run a Surface.scroll example that shows a magnified image



pygame.examples.movieplayer.main
 pygame.examples.moveplayer.main(filepath): return None
play an MPEG movie



pygame.font
 pygame module for loading and rendering fonts



pygame.font.init
 pygame.font.init(): return None
initialize the font module



pygame.font.quit
 pygame.font.quit(): return None
uninitialize the font module



pygame.font.get_init
 pygame.font.get_init(): return bool
true if the font module is initialized



pygame.font.get_default_font
 pygame.font.get_default_font(): return string
get the filename of the default font



pygame.font.get_fonts
 pygame.font.get_fonts(): return list of strings
get all available fonts



pygame.font.match_font
 pygame.font.match_font(name, bold=False, italic=False): return path
find a specific font on the system



pygame.font.SysFont
 pygame.font.SysFont(name, size, bold=False, italic=False): return Font
create a Font object from the system fonts



pygame.font.Font
 pygame.font.Font(filename, size): return Font
pygame.font.Font(object, size): return Font
create a new Font object from a file



Font.render
 Font.render(text, antialias, color, background=None): return Surface
draw text on a new Surface



Font.size
 Font.size(text): return (width, height)
determine the amount of space needed to render text



Font.set_underline
 Font.set_underline(bool): return None
control if text is rendered with an underline



Font.get_underline
 Font.get_underline(): return bool
check if text will be rendered with an underline



Font.set_bold
 Font.set_bold(bool): return None
enable fake rendering of bold text



Font.get_bold
 Font.get_bold(): return bool
check if text will be rendered bold



Font.set_italic
 Font.set_bold(bool): return None
enable fake rendering of italic text



Font.metrics
 Font.metrics(text): return list
Gets the metrics for each character in the pased string.



Font.get_italic
 Font.get_italic(): return bool
check if the text will be rendered italic



Font.get_linesize
 Font.get_linesize(): return int
get the line space of the font text



Font.get_height
 Font.get_height(): return int
get the height of the font



Font.get_ascent
 Font.get_ascent(): return int
get the ascent of the font



Font.get_descent
 Font.get_descent(): return int
get the descent of the font



pygame.gfxdraw
 pygame module for drawing shapes



pygame.gfxdraw.pixel
 pygame.gfxdraw.pixel(surface, x, y, color): return None
place a pixel



pygame.gfxdraw.hline
 pygame.gfxdraw.hline(surface, x1, x2, y, color): return None
draw a horizontal line



pygame.gfxdraw.vline
 pgyame.gfxdraw.vline(surface, x, y1, y2, color): return None
draw a vertical line



pygame.gfxdraw.rectangle
 pgyame.gfxdraw.rectangle(surface, rect, color): return None
draw a rectangle



pygame.gfxdraw.box
 pgyame.gfxdraw.box(surface, rect, color): return None
draw a box



pygame.gfxdraw.line
 pgyame.gfxdraw.line(surface, x1, y1, x2, y2, color): return None
draw a line



pygame.gfxdraw.circle
 pgyame.gfxdraw.circle(surface, x, y, r, color): return None
draw a circle



pygame.gfxdraw.arc
 pgyame.gfxdraw.arc(surface, x, y, r, start, end, color): return None
draw an arc



pygame.gfxdraw.aacircle
 pgyame.gfxdraw.aacircle(surface, x, y, r, color): return None
draw an anti-aliased circle



pygame.gfxdraw.filled_circle
 pgyame.gfxdraw.filled_circle(surface, x, y, r, color): return None
draw a filled circle



pygame.gfxdraw.ellipse
 pgyame.gfxdraw.ellipse(surface, x, y, rx, ry, color): return None
draw an ellipse



pygame.gfxdraw.aaellipse
 pgyame.gfxdraw.aaellipse(surface, x, y, rx, ry, color): return None
draw an anti-aliased ellipse



pygame.gfxdraw.filled_ellipse
 pgyame.gfxdraw.filled_ellipse(surface, x, y, rx, ry, color): return None
draw a filled ellipse



pygame.gfxdraw.pie
 pgyame.gfxdraw.pie(surface, x, y, r, start, end, color): return None
draw a pie



pygame.gfxdraw.trigon
 pgyame.gfxdraw.trigon(surface, x1, y1, x2, y2, x3, y3, color): return None
draw a triangle



pygame.gfxdraw.aatrigon
 pgyame.gfxdraw.aatrigon(surface, x1, y1, x2, y2, x3, y3, color): return None
draw an anti-aliased triangle



pygame.gfxdraw.filled_trigon
 pgyame.gfxdraw.filled_trigon(surface, x1, y1, x3, y2, x3, y3, color): return None
draw a filled trigon



pygame.gfxdraw.polygon
 pgyame.gfxdraw.polygon(surface, points, color): return None
draw a polygon



pygame.gfxdraw.aapolygon
 pgyame.gfxdraw.aapolygon(surface, points, color): return None
draw an anti-aliased polygon



pygame.gfxdraw.filled_polygon
 pgyame.gfxdraw.filled_polygon(surface, points, color): return None
draw a filled polygon



pygame.gfxdraw.textured_polygon
 pgyame.gfxdraw.textured_polygon(surface, points, texture, tx, ty): return None
draw a textured polygon



pygame.gfxdraw.bezier
 pgyame.gfxdraw.bezier(surface, points, steps, color): return None
draw a bezier curve



pygame.image
 pygame module for image transfer



pygame.image.load
 pygame.image.load(filename): return Surface
pygame.image.load(fileobj, namehint=""): return Surface
load new image from a file



pygame.image.save
 pygame.image.save(Surface, filename): return None
save an image to disk



pygame.image.get_extended
 pygame.image.get_extended(): return bool
test if extended image formats can be loaded



pygame.image.tostring
 pygame.image.tostring(Surface, format, flipped=False): return string
transfer image to string buffer



pygame.image.fromstring
 pygame.image.fromstring(string, size, format, flipped=False): return Surface
create new Surface from a string buffer



pygame.image.frombuffer
 pygame.image.frombuffer(string, size, format): return Surface
create a new Surface that shares data inside a string buffer



pygame.joystick
 pygame module for interacting with joystick devices



pygame.joystick.init
 pygame.joystick.init(): return None
initialize the joystick module



pygame.joystick.quit
 pygame.joystick.quit(): return None
uninitialize the joystick module



pygame.joystick.get_init
 pygame.joystick.get_init(): return bool
true if the joystick module is initialized



pygame.joystick.get_count
 pygame.joystick.get_count(): return count
number of joysticks on the system



pygame.joystick.Joystick
 pygame.joystick.Joystick(id): return Joystick
create a new Joystick object



Joystick.init
 Joystick.init(): return None
initialize the Joystick



Joystick.quit
 Joystick.quit(): return None
uninitialize the Joystick



Joystick.get_init
 Joystick.get_init(): return bool
check if the Joystick is initialized



Joystick.get_id
 Joystick.get_id(): return int
get the Joystick ID



Joystick.get_name
 Joystick.get_name(): return string
get the Joystick system name



Joystick.get_numaxes
 Joystick.get_numaxes(): return int
get the number of axes on a Joystick



Joystick.get_axis
 Joystick.get_axis(axis_number): return float
get the current position of an axis



Joystick.get_numballs
 Joystick.get_numballs(): return int
get the number of trackballs on a Joystick



Joystick.get_ball
 Joystick.get_ball(ball_number): return x, y
get the relative position of a trackball



Joystick.get_numbuttons
 Joystick.get_numbuttons(): return int
get the number of buttons on a Joystick



Joystick.get_button
 Joystick.get_button(button): return bool
get the current button state



Joystick.get_numhats
 Joystick.get_numhats(): return int
get the number of hat controls on a Joystick



Joystick.get_hat
 Joystick.get_hat(hat_number): return x, y
get the position of a joystick hat



pygame.key
 pygame module to work with the keyboard



pygame.key.get_focused
 pygame.key.get_focused(): return bool
true if the display is receiving keyboard input from the system



pygame.key.get_pressed
 pygame.key.get_pressed(): return bools
get the state of all keyboard buttons



pygame.key.get_mods
 pygame.key.get_mods(): return int
determine which modifier keys are being held



pygame.key.set_mods
 pygame.key.set_mods(int): return None
temporarily set which modifier keys are pressed



pygame.key.set_repeat
 pygame.key.set_repeat(): return None
pygame.key.set_repeat(delay, interval): return None
control how held keys are repeated



pygame.key.get_repeat
 pygame.key.get_repeat(): return (delay, interval)
see how held keys are repeated



pygame.key.name
 pygame.key.name(key): return string
get the name of a key identifier



pygame.locals
 pygame constants



pygame.mask
 pygame module for image masks.



pygame.mask.from_surface
 pygame.mask.from_surface(Surface, threshold = 127) -> Mask
Returns a Mask from the given surface.



pygame.mask.from_threshold
 pygame.mask.from_surface(Surface, color, threshold = (0,0,0,255), othersurface = None, palette_colors = 1) -> Mask
Creates a mask by thresholding Surfaces



pygame.mask.Mask
 pygame.Mask((width, height)): return Mask
pygame object for representing 2d bitmasks



Mask.get_size
 Mask.get_size() -> width,height
Returns the size of the mask.



Mask.get_at
 Mask.get_at((x,y)) -> int
Returns nonzero if the bit at (x,y) is set.



Mask.set_at
 Mask.set_at((x,y),value)
Sets the position in the mask given by x and y.



Mask.overlap
 Mask.overlap(othermask, offset) -> x,y
Returns the point of intersection if the masks overlap with the given offset - or None if it does not overlap.



Mask.overlap_area
 Mask.overlap_area(othermask, offset) -> numpixels
Returns the number of overlapping 'pixels'.



Mask.overlap_mask
 Mask.overlap_mask(othermask, offset) -> Mask
Returns a mask of the overlapping pixels



Mask.fill
 Mask.fill()
Sets all bits to 1



Mask.clear
 Mask.clear()
Sets all bits to 0



Mask.invert
 Mask.invert()
Flips the bits in a Mask



Mask.scale
 Mask.scale((x, y)) -> Mask
Resizes a mask



Mask.draw
 Mask.draw(othermask, offset)
Draws a mask onto another



Mask.erase
 Mask.erase(othermask, offset)
Erases a mask from another



Mask.count
 Mask.count() -> pixels
Returns the number of set pixels



Mask.centroid
 Mask.centroid() -> (x, y)
Returns the centroid of the pixels in a Mask



Mask.angle
 Mask.angle() -> theta
Returns the orientation of the pixels



Mask.outline
 Mask.outline(every = 1) -> [(x,y), (x,y) ...]
list of points outlining an object



Mask.convolve
 Mask.convolve(othermask, outputmask = None, offset = (0,0)) -> Mask
Return the convolution of self with another mask.



Mask.connected_component
 Mask.connected_component((x,y) = None) -> Mask
Returns a mask of a connected region of pixels.



Mask.connected_components
 Mask.connected_components(min = 0) -> [Masks]
Returns a list of masks of connected regions of pixels.



Mask.get_bounding_rects
 Mask.get_bounding_rects() -> Rects
Returns a list of bounding rects of regions of set pixels.



pygame.midi
 pygame module for interacting with midi input and output.



pygame.midi.Input
 Input(device_id)
Input(device_id, buffer_size)
Input is used to get midi input from midi devices.



Input.close
 Input.close(): return None
 closes a midi stream, flushing any pending buffers.



Input.poll
 Input.poll(): return Bool
returns true if there's data, or false if not.



Input.read
 Input.read(num_events): return midi_event_list
reads num_events midi events from the buffer.



pygame.midi.MidiException
 MidiException(errno)
exception that pygame.midi functions and classes can raise



pygame.midi.Output
 Output(device_id)
Output(device_id, latency = 0)
Output(device_id, buffer_size = 4096)
Output(device_id, latency, buffer_size)
Output is used to send midi to an output device



Output.abort
 Output.abort(): return None
 terminates outgoing messages immediately



Output.close
 Output.close(): return None
 closes a midi stream, flushing any pending buffers.



Output.note_off
 Output.note_off(note, velocity=None, channel = 0)
turns a midi note off.  Note must be on.



Output.note_on
 Output.note_on(note, velocity=None, channel = 0)
turns a midi note on.  Note must be off.



Output.set_instrument
 Output.set_instrument(instrument_id, channel = 0)
select an instrument, with a value between 0 and 127



Output.write
 Output.write(data)
writes a list of midi data to the Output



Output.write_short
 Output.write_short(status)
Output.write_short(status, data1 = 0, data2 = 0)
write_short(status <, data1><, data2>)



Output.write_sys_ex
 Output.write_sys_ex(when, msg)
writes a timestamped system-exclusive midi message.



pygame.midi.get_count
 pygame.midi.get_count(): return num_devices
gets the number of devices.



pygame.midi.get_default_input_id
 pygame.midi.get_default_input_id(): return default_id
gets default input device number



pygame.midi.get_default_output_id
 pygame.midi.get_default_output_id(): return default_id
gets default output device number



pygame.midi.get_device_info
 pygame.midi.get_device_info(an_id): return (interf, name, input, output, opened)
 returns information about a midi device



pygame.midi.init
 pygame.midi.init(): return None
initialize the midi module



pygame.midi.midis2events
 pygame.midi.midis2events(midis, device_id): return [Event, ...]
converts midi events to pygame events



pygame.midi.quit
 pygame.midi.quit(): return None
uninitialize the midi module



pygame.midi.time
 pygame.midi.time(): return time
returns the current time in ms of the PortMidi timer



pygame.mixer
 pygame module for loading and playing sounds



pygame.mixer.init
 pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=4096): return None
initialize the mixer module



pygame.mixer.pre_init
 pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffersize=4096): return None
preset the mixer init arguments



pygame.mixer.quit
 pygame.mixer.quit(): return None
uninitialize the mixer



pygame.mixer.get_init
 pygame.mixer.get_init(): return (frequency, format, channels)
test if the mixer is initialized



pygame.mixer.stop
 pygame.mixer.stop(): return None
stop playback of all sound channels



pygame.mixer.pause
 pygame.mixer.pause(): return None
temporarily stop playback of all sound channels



pygame.mixer.unpause
 pygame.mixer.unpause(): return None
resume paused playback of sound channels



pygame.mixer.fadeout
 pygame.mixer.fadeout(time): return None
fade out the volume on all sounds before stopping



pygame.mixer.set_num_channels
 pygame.mixer.set_num_channels(count): return None
set the total number of playback channels



pygame.mixer.get_num_channels
 get the total number of playback channels



pygame.mixer.set_reserved
 pygame.mixer.set_reserved(count): return None
reserve channels from being automatically used



pygame.mixer.find_channel
 pygame.mixer.find_channel(force=False): return Channel
find an unused channel



pygame.mixer.get_busy
 pygame.mixer.get_busy(): return bool
test if any sound is being mixed



pygame.mixer.Sound
 pygame.mixer.Sound(filename): return Sound
pygame.mixer.Sound(buffer): return Sound
pygame.mixer.Sound(object): return Sound
Create a new Sound object from a file



Sound.play
 Sound.play(loops=0, maxtime=0, fade_ms=0): return Channel
begin sound playback



Sound.stop
 Sound.stop(): return None
stop sound playback



Sound.fadeout
 Sound.fadeout(time): return None
stop sound playback after fading out



Sound.set_volume
 Sound.set_volume(value): return None
set the playback volume for this Sound



Sound.get_volume
 Sound.get_volume(): return value
get the playback volume



Sound.get_num_channels
 Sound.get_num_channels(): return count
count how many times this Sound is playing



Sound.get_length
 Sound.get_length(): return seconds
get the length of the Sound



Sound.get_buffer
 Sound.get_buffer(): return BufferProxy
acquires a buffer object for the sameples of the Sound.



pygame.mixer.Channel
 pygame.mixer.Channel(id): return Channel
Create a Channel object for controlling playback



Channel.play
 Channel.play(Sound, loops=0, maxtime=0, fade_ms=0): return None
play a Sound on a specific Channel



Channel.stop
 Channel.stop(): return None
stop playback on a Channel



Channel.pause
 Channel.pause(): return None
temporarily stop playback of a channel



Channel.unpause
 Channel.unpause(): return None
resume pause playback of a channel



Channel.fadeout
 Channel.fadeout(time): return None
stop playback after fading channel out



Channel.set_volume
 Channel.set_volume(value): return None
Channel.set_volume(left, right): return None
set the volume of a playing channel



Channel.get_volume
 Channel.get_volume(): return value
get the volume of the playing channel



Channel.get_busy
 Channel.get_busy(): return bool
check if the channel is active



Channel.get_sound
 Channel.get_sound(): return Sound
get the currently playing Sound



Channel.queue
 Channel.queue(Sound): return None
queue a Sound object to follow the current



Channel.get_queue
 Channel.get_queue(): return Sound
return any Sound that is queued



Channel.set_endevent
 Channel.set_endevent(): return None
Channel.set_endevent(type): return None
have the channel send an event when playback stops



Channel.get_endevent
 Channel.get_endevent(): return type
get the event a channel sends when playback stops



pygame.mouse
 pygame module to work with the mouse



pygame.mouse.get_pressed
 pygame.moouse.get_pressed(): return (button1, button2, button3)
get the state of the mouse buttons



pygame.mouse.get_pos
 pygame.mouse.get_pos(): return (x, y)
get the mouse cursor position



pygame.mouse.get_rel
 pygame.mouse.get_rel(): return (x, y)
get the amount of mouse movement



pygame.mouse.set_pos
 pygame.mouse.set_pos([x, y]): return None
set the mouse cursor position



pygame.mouse.set_visible
 pygame.mouse.set_visible(bool): return bool
hide or show the mouse cursor



pygame.mouse.get_focused
 pygame.mouse.get_focused(): return bool
check if the display is receiving mouse input



pygame.mouse.set_cursor
 pygame.mouse.set_cursor(size, hotspot, xormasks, andmasks): return None
set the image for the system mouse cursor



pygame.mouse.get_cursor
 pygame.mouse.get_cursor(): return (size, hotspot, xormasks, andmasks)
get the image for the system mouse cursor



pygame.movie
 pygame module for playback of mpeg video



pygame.movie.Movie
 pygame.movie.Movie(filename): return Movie
pygame.movie.Movie(object): return Movie
load an mpeg movie file



Movie.play
 Movie.play(loops=0): return None
start playback of a movie



Movie.stop
 Movie.stop(): return None
stop movie playback



Movie.pause
 Movie.pause(): return None
temporarily stop and resume playback



Movie.skip
 Movie.skip(seconds): return None
advance the movie playback position



Movie.rewind
 Movie.rewind(): return None
restart the movie playback



Movie.render_frame
 Movie.render_frame(frame_number): return frame_number
set the current video frame



Movie.get_frame
 Movie.get_frame(): return frame_number
get the current video frame



Movie.get_time
 Movie.get_time(): return seconds
get the current vide playback time



Movie.get_busy
 Movie.get_busy(): return bool
check if the movie is currently playing



Movie.get_length
 Movie.get_length(): return seconds
the total length of the movie in seconds



Movie.get_size
 Movie.get_size(): return (width, height)
get the resolution of the video



Movie.has_video
 Movie.get_video(): return bool
check if the movie file contains video



Movie.has_audio
 Movie.get_audio(): return bool
check if the movie file contains audio



Movie.set_volume
 Movie.set_volume(value): return None
set the audio playback volume



Movie.set_display
 Movie.set_display(Surface, rect=None): return None
set the video target Surface



pygame.mixer.music
 pygame module for controlling streamed audio



pygame.mixer.music.load
 pygame.mixer.music.load(filename): return None
pygame.mixer.music.load(object): return None
Load a music file for playback



pygame.mixer.music.play
 pygame.mixer.music.play(loops=0, start=0.0): return None
Start the playback of the music stream



pygame.mixer.music.rewind
 pygame.mixer.music.rewind(): return None
restart music



pygame.mixer.music.stop
 pygame.mixer.music.stop(): return None
stop the music playback



pygame.mixer.music.pause
 pygame.mixer.music.pause(): return None
temporarily stop music playback



pygame.mixer.music.unpause
 pygame.mixer.music.unpause(): return None
resume paused music



pygame.mixer.music.fadeout
 pygame.mixer.music.fadeout(time): return None
stop music playback after fading out



pygame.mixer.music.set_volume
 pygame.mixer.music.set_volume(value): return None
set the music volume



pygame.mixer.music.get_volume
 pygame.mixer.music.get_volume(): return value
get the music volume



pygame.mixer.music.get_busy
 pygame.mixer.music.get_busy(): return bool
check if the music stream is playing



pygame.mixer.music.get_pos
 pygame.mixer.music.get_pos(): return time
get the music play time



pygame.mixer.music.queue
 pygame.mixer.music.queue(filename): return None
queue a music file to follow the current



pygame.mixer.music.set_endevent
 pygame.mixer.music.set_endevent(): return None
pygame.mixer.music.set_endevent(type): return None
have the music send an event when playback stops



pygame.mixer.music.get_endevent
 pygame.mixer.music.get_endevent(): return type
get the event a channel sends when playback stops



pygame.Overlay
 pygame.Overlay(format, (width, height)): return Overlay
pygame object for video overlay graphics



Overlay.display
 Overlay.display((y, u, v)): return None
Overlay.display(): return None
set the overlay pixel data



Overlay.set_location
 Overlay.set_location(rect): return None
control where the overlay is displayed



Overlay.get_hardware
 Overlay.get_hardware(rect): return int
test if the Overlay is hardware accelerated



pygame.PixelArray
 pygame.PixelArray(Surface): return PixelArray
pygame object for direct pixel access of surfaces



PixelArray.surface
 PixelArray.surface: Return Surface
Gets the Surface the PixelArray uses.



PixelArray.make_surface
 PixelArray.make_surface (): Return Surface
Creates a new Surface from the current PixelArray.



PixelArray.replace
 PixelArray.replace (color, repcolor, distance=0, weights=(0.299, 0.587, 0.114)): Return None
Replaces the passed color in the PixelArray with another one.



PixelArray.extract
 PixelArray.extract (color, distance=0, weights=(0.299, 0.587, 0.114)): Return PixelArray
Extracts the passed color from the PixelArray.



PixelArray.compare
 PixelArray.compare (array, distance=0, weights=(0.299, 0.587, 0.114)): Return PixelArray
Compares the PixelArray with another one.



pygame.Rect
 pygame.Rect(left, top, width, height): return Rect
pygame.Rect((left, top), (width, height)): return Rect
pygame.Rect(object): return Rect
pygame object for storing rectangular coordinates



Rect.copy
 Rect.copy(): return Rect
copy the rectangle



Rect.move
 Rect.move(x, y): return Rect
moves the rectangle



Rect.move_ip
 Rect.move_ip(x, y): return None
moves the rectangle, in place



Rect.inflate
 Rect.inflate(x, y): return Rect
grow or shrink the rectangle size



Rect.inflate_ip
 Rect.inflate_ip(x, y): return None
grow or shrink the rectangle size, in place



Rect.clamp
 Rect.clamp(Rect): return Rect
moves the rectangle inside another



Rect.clamp_ip
 Rect.clamp_ip(Rect): return None
moves the rectangle inside another, in place



Rect.clip
 Rect.clip(Rect): return Rect
crops a rectangle inside another



Rect.union
 Rect.union(Rect): return Rect
joins two rectangles into one



Rect.union_ip
 Rect.union_ip(Rect): return None
joins two rectangles into one, in place



Rect.unionall
 Rect.unionall(Rect_sequence): return Rect
the union of many rectangles



Rect.unionall_ip
 Rect.unionall_ip(Rect_sequence): return None
the union of many rectangles, in place



Rect.fit
 Rect.fit(Rect): return Rect
resize and move a rectangle with aspect ratio



Rect.normalize
 Rect.normalize(): return None
correct negative sizes



Rect.contains
 Rect.contains(Rect): return bool
test if one rectangle is inside another



Rect.collidepoint
 Rect.collidepoint(x, y): return bool
Rect.collidepoint((x,y)): return bool
test if a point is inside a rectangle



Rect.colliderect
 Rect.colliderect(Rect): return bool
test if two rectangles overlap



Rect.collidelist
 Rect.collidelist(list): return index
test if one rectangle in a list intersects



Rect.collidelistall
 Rect.collidelistall(list): return indices
test if all rectangles in a list intersect



Rect.collidedict
 Rect.collidedict(dict): return (key, value)
test if one rectangle in a dictionary intersects



Rect.collidedictall
 Rect.collidedictall(dict): return [(key, value), ...]
test if all rectangles in a dictionary intersect



pygame.scrap
 pygame module for clipboard support.



pygame.scrap.init
 scrap.init () -> None
Initializes the scrap module.



pygame.scrap.get
 scrap.get (type) -> string
Gets the data for the specified type from the clipboard.



pygame.scrap.get_types
 scrap.get_types () -> list
Gets a list of the available clipboard types.



pygame.scrap.put
 scrap.put(type, data) -> None
Places data into the clipboard.



pygame.scrap.contains
 scrap.contains (type) -> bool
Checks, whether a certain type is available in the clipboard.



pygame.scrap.lost
 scrap.lost() -> bool
Checks whether the clipboard is currently owned by the application.



pygame.scrap.set_mode
 scrap.set_mode(mode) -> None
Sets the clipboard access mode.



pygame.sndarray
 pygame module for accessing sound sample data



pygame.sndarray.array
 pygame.sndarray.array(Sound): return array
copy Sound samples into an array



pygame.sndarray.samples
 pygame.sndarray.samples(Sound): return array
reference Sound samples into an array



pygame.sndarray.make_sound
 pygame.sndarray.make_sound(array): return Sound
convert an array into a Sound object



pygame.sndarray.use_arraytype
 pygame.sndarray.use_arraytype (arraytype): return None
Sets the array system to be used for sound arrays



pygame.sndarray.get_arraytype
 pygame.sndarray.get_arraytype (): return str
Gets the currently active array type.



pygame.sndarray.get_arraytypes
 pygame.sndarray.get_arraytypes (): return tuple
Gets the array system types currently supported.



pygame.sprite
 pygame module with basic game object classes



pygame.sprite.Sprite
 pygame.sprite.Sprite(*groups): return Sprite
simple base class for visible game objects



Sprite.update
 Sprite.update(*args):
method to control sprite behavior



Sprite.add
 Sprite.add(*groups): return None
add the sprite to groups



Sprite.remove
 Sprite.remove(*groups): return None
remove the sprite from groups



Sprite.kill
 Sprite.kill(): return None
remove the Sprite from all Groups



Sprite.alive
 Sprite.alive(): return bool
does the sprite belong to any groups



Sprite.groups
 Sprite.groups(): return group_list
list of Groups that contain this Sprite



pygame.sprite.DirtySprite
 pygame.sprite.DirtySprite(*groups): return DirtySprite
a more featureful subclass of Sprite with more attributes




 



pygame.sprite.Group
 pygame.sprite.Group(*sprites): return Group
container class for many Sprites



Group.sprites
 Group.sprites(): return sprite_list
list of the Sprites this Group contains



Group.copy
 Group.copy(): return Group
duplicate the Group



Group.add
 Group.add(*sprites): return None
add Sprites to this Group



Group.remove
 Group.remove(*sprites): return None
remove Sprites from the Group



Group.has
 Group.has(*sprites): return None
test if a Group contains Sprites



Group.update
 Group.update(*args): return None
call the update method on contained Sprites



Group.draw
 Group.draw(Surface): return None
blit the Sprite images



Group.clear
 Group.clear(Surface_dest, background): return None
draw a background over the Sprites



Group.empty
 Group.empty(): return None
remove all Sprites



pygame.sprite.RenderUpdates
 pygame.sprite.RenderUpdates(*sprites): return RenderUpdates
Group class that tracks dirty updates



RenderUpdates.draw
 RenderUpdates.draw(surface): return Rect_list
blit the Sprite images and track changed areas



pygame.sprite.OrderedUpdates
 pygame.sprite.OrderedUpdates(*spites): return OrderedUpdates
RenderUpdates class that draws Sprites in order of addition



pygame.sprite.LayeredUpdates
 pygame.sprite.LayeredUpdates(*spites, **kwargs): return LayeredUpdates
LayeredUpdates Group handles layers, that draws like OrderedUpdates.



LayeredUpdates.add
 LayeredUpdates.add(*sprites, **kwargs): return None
add a sprite or sequence of sprites to a group



LayeredUpdates.sprites
 LayeredUpdates.sprites(): return sprites
returns a ordered list of sprites (first back, last top).



LayeredUpdates.draw
 LayeredUpdates.draw(surface): return Rect_list
draw all sprites in the right order onto the passed surface.



LayeredUpdates.get_sprites_at
 LayeredUpdates.get_sprites_at(pos): return colliding_sprites
returns a list with all sprites at that position.



LayeredUpdates.get_sprite
 LayeredUpdates.get_sprite(idx): return sprite
returns the sprite at the index idx from the groups sprites



LayeredUpdates.remove_sprites_of_layer
 LayeredUpdates.remove_sprites_of_layer(layer_nr): return sprites
removes all sprites from a layer and returns them as a list.



LayeredUpdates.layers
 LayeredUpdates.layers(): return layers
returns a list of layers defined (unique), sorted from botton up.



LayeredUpdates.change_layer
 LayeredUpdates.change_layer(sprite, new_layer): return None
changes the layer of the sprite



LayeredUpdates.get_layer_of_sprite
 LayeredUpdates.get_layer_of_sprite(sprite): return layer
returns the layer that sprite is currently in.



LayeredUpdates.get_top_layer
 LayeredUpdates.get_top_layer(): return layer
returns the top layer



LayeredUpdates.get_bottom_layer
 LayeredUpdates.get_bottom_layer(): return layer
returns the bottom layer



LayeredUpdates.move_to_front
 LayeredUpdates.move_to_front(sprite): return None
brings the sprite to front layer



LayeredUpdates.move_to_back
 LayeredUpdates.move_to_back(sprite): return None
moves the sprite to the bottom layer



LayeredUpdates.get_top_sprite
 LayeredUpdates.get_top_sprite(): return Sprite
returns the topmost sprite



LayeredUpdates.get_sprites_from_layer
 LayeredUpdates.get_sprites_from_layer(layer): return sprites
returns all sprites from a layer, ordered by how they where added



LayeredUpdates.switch_layer
 LayeredUpdates.switch_layer(layer1_nr, layer2_nr): return None
switches the sprites from layer1 to layer2



pygame.sprite.LayeredDirty
 pygame.sprite.LayeredDirty(*spites, **kwargs): return LayeredDirty
LayeredDirty Group is for DirtySprites.  Subclasses LayeredUpdates.



LayeredDirty.draw
 LayeredDirty.draw(surface, bgd=None): return Rect_list
draw all sprites in the right order onto the passed surface.



LayeredDirty.clear
 LayeredDirty.clear(surface, bgd): return None
used to set background



LayeredDirty.repaint_rect
 LayeredDirty.repaint_rect(screen_rect): return None
repaints the given area



LayeredDirty.set_clip
 LayeredDirty.set_clip(screen_rect=None): return None
clip the area where to draw. Just pass None (default) to reset the clip



LayeredDirty.get_clip
 LayeredDirty.get_clip(): return Rect
clip the area where to draw. Just pass None (default) to reset the clip



LayeredDirty.change_layer
 change_layer(sprite, new_layer): return None
changes the layer of the sprite



LayeredDirty.set_timing_treshold
 set_timing_treshold(time_ms): return None
sets the treshold in milliseconds



pygame.sprite.GroupSingle
 pygame.sprite.GroupSingle(sprite=None): return GroupSingle
Group container that holds a single Sprite



pygame.sprite.spritecollide
 pygame.sprite.spritecollide(sprite, group, dokill, collided = None): return Sprite_list
find Sprites in a Group that intersect another Sprite



pygame.sprite.collide_rect
 pygame.sprite.collide_rect(left, right): return bool
collision detection between two sprites, using rects.



pygame.sprite.collide_rect_ratio
 pygame.sprite.collide_rect_ratio(ratio): return collided_callable
collision detection between two sprites, using rects scaled to a ratio.



pygame.sprite.collide_circle
 pygame.sprite.collide_circle(left, right): return bool
collision detection between two sprites, using circles.



pygame.sprite.collide_circle_ratio
 pygame.sprite.collide_circle_ratio(ratio): return collided_callable
collision detection between two sprites, using circles scaled to a ratio.



pygame.sprite.collide_mask
 pygame.sprite.collide_mask(SpriteLeft, SpriteRight): return bool
collision detection between two sprites, using masks.



pygame.sprite.groupcollide
 pygame.sprite.groupcollide(group1, group2, dokill1, dokill2): return Sprite_dict
find all Sprites that collide between two Groups



pygame.sprite.spritecollideany
 pygame.sprite.spritecollideany(sprite, group): return bool
simple test if a Sprite intersects anything in a Group




 



pygame.Surface
 pygame.Surface((width, height), flags=0, depth=0, masks=None): return Surface
pygame.Surface((width, height), flags=0, Surface): return Surface
pygame object for representing images



Surface.blit
 Surface.blit(source, dest, area=None, special_flags = 0): return Rect
draw one image onto another



Surface.convert
 Surface.convert(Surface): return Surface
Surface.convert(depth, flags=0): return Surface
Surface.convert(masks, flags=0): return Surface
Surface.convert(): return Surface
change the pixel format of an image



Surface.convert_alpha
 Surface.convert_alpha(Surface): return Surface
Surface.convert_alpha(): return Surface
change the pixel format of an image including per pixel alphas



Surface.copy
 Surface.copy(): return Surface
create a new copy of a Surface



Surface.fill
 Surface.fill(color, rect=None, special_flags=0): return Rect
fill Surface with a solid color



Surface.scroll
 Surface.scroll(dx=0, dy=0): return None
Shift the surface image in place



Surface.set_colorkey
 Surface.set_colorkey(Color, flags=0): return None
Surface.set_colorkey(None): return None
Set the transparent colorkey



Surface.get_colorkey
 Surface.get_colorkey(): return RGB or None
Get the current transparent colorkey



Surface.set_alpha
 Surface.set_alpha(value, flags=0): return None
Surface.set_alpha(None): return None
set the alpha value for the full Surface image



Surface.get_alpha
 Surface.get_alpha(): return int_value or None
get the current Surface transparency value



Surface.lock
 Surface.lock(): return None
lock the Surface memory for pixel access



Surface.unlock
 Surface.unlock(): return None
unlock the Surface memory from pixel access



Surface.mustlock
 Surface.mustlock(): return bool
test if the Surface requires locking



Surface.get_locked
 Surface.get_locked(): return bool
test if the Surface is current locked



Surface.get_locks
 Surface.get_locks(): return tuple
Gets the locks for the Surface



Surface.get_at
 Surface.get_at((x, y)): return Color
get the color value at a single pixel



Surface.set_at
 Surface.set_at((x, y), Color): return None
set the color value for a single pixel



Surface.get_palette
 Surface.get_palette(): return [RGB, RGB, RGB, ...]
get the color index palette for an 8bit Surface



Surface.get_palette_at
 Surface.get_palette_at(index): return RGB
get the color for a single entry in a palette



Surface.set_palette
 Surface.set_palette([RGB, RGB, RGB, ...]): return None
set the color palette for an 8bit Surface



Surface.set_palette_at
 Surface.set_at(index, RGB): return None
set the color for a single index in an 8bit Surface palette



Surface.map_rgb
 Surface.map_rgb(Color): return mapped_int
convert a color into a mapped color value



Surface.unmap_rgb
 Surface.map_rgb(mapped_int): return Color
convert a mapped integer color value into a Color



Surface.set_clip
 Surface.set_clip(rect): return None
Surface.set_clip(None): return None
set the current clipping area of the Surface



Surface.get_clip
 Surface.get_clip(): return Rect
get the current clipping area of the Surface



Surface.subsurface
 Surface.subsurface(Rect): return Surface
create a new surface that references its parent



Surface.get_parent
 Surface.get_parent(): return Surface
find the parent of a subsurface



Surface.get_abs_parent
 Surface.get_abs_parent(): return Surface
find the top level parent of a subsurface



Surface.get_offset
 Surface.get_offset(): return (x, y)
find the position of a child subsurface inside a parent



Surface.get_abs_offset
 Surface.get_abs_offset(): return (x, y)
find the absolute position of a child subsurface inside its top level parent



Surface.get_size
 Surface.get_size(): return (width, height)
get the dimensions of the Surface



Surface.get_width
 Surface.get_width(): return width
get the width of the Surface



Surface.get_height
 Surface.get_height(): return height
get the height of the Surface



Surface.get_rect
 Surface.get_rect(**kwargs): return Rect
get the rectangular area of the Surface



Surface.get_bitsize
 Surface.get_bitsize(): return int
get the bit depth of the Surface pixel format



Surface.get_bytesize
 Surface.get_bytesize(): return int
get the bytes used per Surface pixel



Surface.get_flags
 Surface.get_flags(): return int
get the additional flags used for the Surface



Surface.get_pitch
 Surface.get_pitch(): return int
get the number of bytes used per Surface row



Surface.get_masks
 Surface.get_masks(): return (R, G, B, A)
the bitmasks needed to convert between a color and a mapped integer



Surface.set_masks
 Surface.set_masks((r,g,b,a)): return None
set the bitmasks needed to convert between a color and a mapped integer



Surface.get_shifts
 Surface.get_shifts(): return (R, G, B, A)
the bit shifts needed to convert between a color and a mapped integer



Surface.set_shifts
 Surface.get_shifts((r,g,b,a)): return None
sets the bit shifts needed to convert between a color and a mapped integer



Surface.get_losses
 Surface.get_losses(): return (R, G, B, A)
the significant bits used to convert between a color and a mapped integer



Surface.get_bounding_rect
 Surface.get_bounding_rect(min_alpha = 1): return Rect
find the smallest rect containing data



Surface.get_buffer
 Surface.get_buffer(): return BufferProxy
acquires a buffer object for the pixels of the Surface.



pygame.surfarray
 pygame module for accessing surface pixel data using array interfaces



pygame.surfarray.array2d
 pygame.surfarray.array2d(Surface): return array
Copy pixels into a 2d array



pygame.surfarray.pixels2d
 pygame.surfarray.pixels2d(Surface): return array
Reference pixels into a 2d array



pygame.surfarray.array3d
 pygame.surfarray.array3d(Surface): return array
Copy pixels into a 3d array



pygame.surfarray.pixels3d
 pygame.surfarray.pixels3d(Surface): return array
Reference pixels into a 3d array



pygame.surfarray.array_alpha
 pygame.surfarray.array_alpha(Surface): return array
Copy pixel alphas into a 2d array



pygame.surfarray.pixels_alpha
 pygame.surfarray.pixels_alpha(Surface): return array
Reference pixel alphas into a 2d array



pygame.surfarray.array_colorkey
 pygame.surfarray.array_colorkey(Surface): return array
Copy the colorkey values into a 2d array



pygame.surfarray.make_surface
 pygame.surfarray.make_surface(array): return Surface
Copy an array to a new surface



pygame.surfarray.blit_array
 pygame.surfarray.blit_array(Surface, array): return None
Blit directly from a array values



pygame.surfarray.map_array
 pygame.surfarray.map_array(Surface, array3d): return array2d
Map a 3d array into a 2d array



pygame.surfarray.use_arraytype
 pygame.surfarray.use_arraytype (arraytype): return None
Sets the array system to be used for surface arrays



pygame.surfarray.get_arraytype
 pygame.surfarray.get_arraytype (): return str
Gets the currently active array type.



pygame.surfarray.get_arraytypes
 pygame.surfarray.get_arraytypes (): return tuple
Gets the array system types currently supported.



pygame.tests
 Pygame unit test suite package



pygame.tests.run
 pygame.tests.run(*args, **kwds): return tuple
Run the Pygame unit test suite



pygame.time
 pygame module for monitoring time



pygame.time.get_ticks
 pygame.time.get_ticks(): return milliseconds
get the time in milliseconds



pygame.time.wait
 pygame.time.wait(milliseconds): return time
pause the program for an amount of time



pygame.time.delay
 pygame.time.delay(milliseconds): return time
pause the program for an amount of time



pygame.time.set_timer
 pygame.time.set_timer(eventid, milliseconds): return None
repeatedly create an event on the event queue



pygame.time.Clock
 pygame.time.Clock(): return Clock
create an object to help track time



Clock.tick
 Clock.tick(framerate=0): return milliseconds
control timer events
update the clock



Clock.tick_busy_loop
 Clock.tick_busy_loop(framerate=0): return milliseconds
control timer events
update the clock



Clock.get_time
 Clock.get_time(): return milliseconds
time used in the previous tick



Clock.get_rawtime
 Clock.get_rawtime(): return milliseconds
actual time used in the previous tick



Clock.get_fps
 Clock.get_fps(): return float
compute the clock framerate



pygame.transform
 pygame module to transform surfaces



pygame.transform.flip
 pygame.transform.flip(Surface, xbool, ybool): return Surface
flip vertically and horizontally



pygame.transform.scale
 pygame.transform.scale(Surface, (width, height), DestSurface = None): return Surface
resize to new resolution



pygame.transform.rotate
 pygame.transform.rotate(Surface, angle): return Surface
rotate an image



pygame.transform.rotozoom
 pygame.transform.rotozoom(Surface, angle, scale): return Surface
filtered scale and rotation



pygame.transform.scale2x
 pygame.transform.scale2x(Surface, DestSurface = None): Surface
specialized image doubler



pygame.transform.smoothscale
 pygame.transform.smoothscale(Surface, (width, height), DestSurface = None): return Surface
scale a surface to an arbitrary size smoothly



pygame.transform.get_smoothscale_backend
 pygame.transform.get_smoothscale_backend(): return String
return smoothscale filter version in use: 'GENERIC', 'MMX', or 'SSE'



pygame.transform.set_smoothscale_backend
 pygame.transform.get_smoothscale_backend(type): return None
set smoothscale filter version to one of: 'GENERIC', 'MMX', or 'SSE'



pygame.transform.chop
 pygame.transform.chop(Surface, rect): return Surface
gets a copy of an image with an interior area removed



pygame.transform.laplacian
 pygame.transform.laplacian(Surface, DestSurface = None): return Surface
find edges in a surface



pygame.transform.average_surfaces
 pygame.transform.average_surfaces(Surfaces, DestSurface = None, palette_colors = 1): return Surface
find the average surface from many surfaces.



pygame.transform.average_color
 pygame.transform.average_color(Surface, Rect = None): return Color
finds the average color of a surface



pygame.transform.threshold
 pygame.transform.threshold(DestSurface, Surface, color, threshold = (0,0,0,0), diff_color = (0,0,0,0), change_return = 1, Surface = None, inverse = False): return num_threshold_pixels
finds which, and how many pixels in a surface are within a threshold of a color.



*/

