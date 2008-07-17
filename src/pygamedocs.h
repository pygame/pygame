/* Auto generated file: with makeref.py .  Docs go in src/ *.doc . */
#define DOC_PYGAME "the top level pygame package"

#define DOC_PYGAMEINIT "pygame.init(): return (numpass, numfail)\ninitialize all imported pygame modules"

#define DOC_PYGAMEQUIT "pygame.quit(): return None\nuninitialize all pygame modules"

#define DOC_PYGAMEERROR "raise pygame.error, message\nstandard pygame exception"

#define DOC_PYGAMEGETERROR "pygame.get_error(): return errorstr\nget the current error message"

#define DOC_PYGAMEGETSDLVERSION "pygame.get_sdl_version(): return major, minor, patch\nget the version number of SDL"

#define DOC_PYGAMEGETSDLBYTEORDER "pygame.get_sdl_byteorder(): return int\nget the byte order of SDL"

#define DOC_PYGAMEREGISTERQUIT "register_quit(callable): return None\nregister a function to be called when pygame quits"

#define DOC_PYGAMEVERSION "module pygame.version\nsmall module containing version information"

#define DOC_PYGAMEVERSIONVER "pygame.version.ver = '1.2'\nversion number as a string"

#define DOC_PYGAMEVERSIONVERNUM "pygame.version.vernum = (1, 5, 3)\ntupled integers of the version"

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

#define DOC_PYGAMECURSORS "pygame module for cursor resources"

#define DOC_PYGAMECURSORSCOMPILE "pygame.cursor.compile(strings, black='X', white='.', xor='o'): return data, mask\ncreate binary cursor data from simple strings"

#define DOC_PYGAMECURSORSLOADXBM "pygame.cursors.load_xbm(cursorfile, maskfile=None): return cursor_args\nload cursor data from an xbm file"

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

#define DOC_PYGAMEMASK "pygame module for image masks."

#define DOC_PYGAMEMASKFROMSURFACE "pygame.mask.from_surface(Surface, threshold = 127) -> Mask\nReturns a Mask from the given surface."

#define DOC_PYGAMEMASKMASK "pygame.Mask((width, height): return Mask\npygame object for representing 2d bitmasks"

#define DOC_MASKGETSIZE "Mask.get_size() -> width,height\nReturns the size of the mask."

#define DOC_MASKGETAT "Mask.get_at((x,y)) -> int\nReturns nonzero if the bit at (x,y) is set."

#define DOC_MASKSETAT "Mask.set_at((x,y),value)\nSets the position in the mask given by x and y."

#define DOC_MASKOVERLAP "Mask.overlap(othermask, offset) -> x,y\nReturns the point of intersection if the masks overlap with the given offset - or None if it does not overlap."

#define DOC_MASKOVERLAPAREA "Mask.overlap_area(othermask, offset) -> numpixels\nReturns the number of overlapping 'pixels'."

#define DOC_MASKGETBOUNDINGRECTS "Mask.get_bounding_rects() -> Rects\nReturns a list of bounding rects of regions of set pixels."

#define DOC_PYGAMEMIXER "pygame module for loading and playing sounds"

#define DOC_PYGAMEMIXERINIT "pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=3072): return None\ninitialize the mixer module"

#define DOC_PYGAMEMIXERPREINIT "pygame.mixer.pre_init(frequency=0, size=0, channels=0, buffersize=0): return None\npreset the mixer init arguments"

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

#define DOC_PYGAMEMIXERMUSICLOAD "pygame.mixer.music.load(filename): return None\nLoad a music file for playback"

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

#define DOC_PYGAMETRANSFORMCHOP "pygame.transform.chop(Surface, rect): return Surface\ngets a copy of an image with an interior area removed"

#define DOC_PYGAMETRANSFORMLAPLACIAN "pygame.transform.laplacian(Surface, DestSurface = None): return Surface\nfind edges in a surface"

#define DOC_PYGAMETRANSFORMAVERAGESURFACES "pygame.transform.average_surfaces(Surfaces, DestSurface = None): return Surface\nfind the average surface from many surfaces."

#define DOC_PYGAMETRANSFORMTHRESHOLD "finds which, and how many pixels in a surface are within a threshold of a color."

