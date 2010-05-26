####################
Pygame2 for Pygamers
####################

This section is about migrating from Pygame to Pygame2 and explains the
smaller and bigger changes, which you will encounter, if you are already
used to Pygame.

Module Names and Packages
=========================

In Pygame2, the whole module and package layout changed. While Pygame
used a 'get all from module `pygame`' approach, Pygame2 forces you to
explicitly import the required bits and pieces.

While in Pygame, you simply do a ::

  import pygame # Import anything

in Pygame2 you have to ::

  import pygame2           # Import the Pygame2 core
  import pygame2.sdl       # Import the SDL core wrapper
  import pygame2.sdl.video # Import the SDL video system
  import pygame2.sdlmixer  # Import the SDL_mixer wrapper
  ...

Each Pygame2 package and module forces you to explicitly import
it. While this sounds like a lot of typing work all over the place, it
also lets you easily choose and deploy only the minimum of the required
dependencies for your application and keeps your imports free of
superfluous stuff.

Initialization and Shutdown
===========================

Similar to the explicit imports, you also have to explicitly initialize
the modules, where necessary. In Pygame, a ::

  pygame.init ()
  ...
  pygame.quit ()

is enough to initialize nearly all parts and shut them down at the end
of the program. In Pygame2, you have to explicitly initialize the
modules (not all of them, of course. Only the majority of the
SDL-related ones and :mod:`pygame2.freetype`). ::

  pygame2.sdl.video.init ()
  pygame2.sdl.audio.init ()
  pygame2.sdlmixer.init ()
  ...
  pygame2.sdl.video.quit ()
  pygame2.sdl.audio.quit ()
  pygame2.sdlmixer.quit ()

.. note::
 
  The :mod:`pygame2.sdl` modules can also be initialized using
  :func:`pygame2.sdl.init()` and the appropriate flags.

Some modules might contain an :func:`init` and :func:`quit` method,
which do not use anything (such as e.g. :mod:`pygame2.openal`). Those
are mostly for future usage and might be required to be invoked in later
version of Pygame2.

Detail Changes (or: where to find what)
=======================================

The following subsections will provide you a comparision of which Pygame
module, class, method or function can be found where in Pygame2 now.

pygame
------

pygame.init
  There is no similar function or attribute in Pygame2. Use the
  corresponding :func:`init` functions of the modules and packages.

pygame.quit
  There is no similar function or attribute in Pygame2. Use the
  corresponding :func:`quit` functions of the modules and packages.

pygame.error
  :class:`pygame2.Error`

pygame.get_error
  :func:`pygame2.sdl.get_error`, :func:`pygame2.sdlmixer.get_error`, ...

pygame.set_error
  There is no similar function or attribute in Pygame2. 

pygame.get_sdl_version
  :func:`pygame2.sdl.get_version` and :func:`pygame2.sdl.get_compiled_version` 

pygame.get_sdl_byteorder
  :const:`pygame2.sdl.constants.BYTEORDER`

pygame.register_quit
  There is no similar function or attribute in Pygame2. 

pygame.version
  :attr:`pygame2.version_info` and :attr:`pygame2.__version__`

pygame.camera
-------------

The module is not (yet) ported to Pygame2.

pygame.cdrom
------------

pygame.cdrom
  :mod:`pygame2.sdl.cdrom`

pygame.cdrom.init
  :func:`pygame2.sdl.cdrom.init`

pygame.cdrom.quit
  :func:`pygame2.sdl.cdrom.quit`

pygame.cdrom.get_init
  :func:`pygame2.sdl.cdrom.was_init`

pygame.cdrom.get_count
  :func:`pygame2.sdl.cdrom.num_drives`

pygame.cdrom
  :mod:`pygame2.sdl.cdrom`

pygame.cdrom.CD
  :class:`pygame2.sdl.cdrom.CD`

pygame.cdrom.CD.init
  :meth:`pygame2.sdl.cdrom.CD.open`

pygame.cdrom.CD.quit
  :meth:`pygame2.sdl.cdrom.CD.close`

pygame.cdrom.CD.get_init
   There is no similar function or attribute in Pygame2.

pygame.cdrom.CD.play
  :meth:`pygame2.sdl.cdrom.CD.play` and
  :meth:`pygame2.sdl.cdrom.CD.play_tracks`

pygame.cdrom.CD.stop
  :meth:`pygame2.sdl.cdrom.CD.stop`

pygame.cdrom.CD.pause
  :meth:`pygame2.sdl.cdrom.CD.pause`

pygame.cdrom.CD.resume
  :meth:`pygame2.sdl.cdrom.CD.resume`

pygame.cdrom.CD.eject
  :meth:`pygame2.sdl.cdrom.CD.eject`

pygame.cdrom.CD.get_id
  :attr:`pygame2.sdl.cdrom.CD.index`

pygame.cdrom.CD.get_name
  :attr:`pygame2.sdl.cdrom.CD.name`

pygame.cdrom.CD.get_busy
  :attr:`pygame2.sdl.cdrom.CD.status` == :const:`pygame2.sdl.constants.CD_PLAYING`

pygame.cdrom.CD.get_paused
  :attr:`pygame2.sdl.cdrom.CD.status` == :const:`pygame2.sdl.constants.CD_PAUSED`

pygame.cdrom.CD.get_current
  :attr:`pygame2.sdl.cdrom.CD.cur_track` and
  :attr:`pygame2.sdl.cdrom.CD.cur_frame`

  :attr:`pygame2.sdl.cdrom.CD.cur_track` will return a
  :class:`pygame2.sdl.cdrom.CDTrack` object with additional information
  about the track length, type, etc.

pygame.cdrom.CD.get_empty
  :attr:`pygame2.sdl.cdrom.CD.status` == :const:`pygame2.sdl.constants.CD_TRAYEMPTY`

pygame.cdrom.CD.get_numtracks
  :attr:`pygame2.sdl.cdrom.CD.num_tracks`

pygame.cdrom.CD.get_track_audio
  :attr:`pygame2.sdl.cdrom.CDTrack.type` == :const:`pygame2.sdl.constants.AUDIO_TRACK`

pygame.cdrom.CD.get_all
  :attr:`pygame2.sdl.cdrom.CD.tracks`

  :attr:`pygame2.sdl.cdrom.CD.tracks` returns a list of
  :attr:`pygame2.sdl.cdrom.CDTrack` object, which provide additional
  information about the tracks.

pygame.cdrom.CD.get_track_start
  :attr:`pygame2.sdl.cdrom.CDTrack.offset`

  :attr:`pygame2.sdl.cdrom.CDTrack.offset` will return the track offset
  in frames.

pygame.cdrom.CD.get_track_length
  :attr:`pygame2.sdl.cdrom.CDTrack.length`

pygame.Color
------------

No notable changes apply here, except that there is no replacement for
:meth:`pygame.Color.set_length`.

pygame.cursors
--------------

pygame.cursors can be found under :mod:`pygame2.sdl.cursors`.

pygame.display
--------------

pygame.display
  There is no 1:1 replacement for :mod:`pygame.display`.

  Instead you have to use a mixture of :mod:`pygame2.sdl.video`,
  :mod:`pygame2.sdl.wm` and :mod:`pygame2.sdl.gl`

pygame.display.init
  :func:`pygame2.sdl.video.init`

pygame.display.quit
  :func:`pygame2.sdl.video.quit`

pygame.display.get_init
  :func:`pygame2.sdl.video.was_init`

pygame.display.set_mode
  :func:`pygame2.sdl.video.set_mode`

pygame.display.get_surface
  :func:`pygame2.sdl.video.get_videosurface`

pygame.display.flip
  Use :meth:`pygame2.sdl.video.Surface.flip` of the video surface.

pygame.display.update
  Use :meth:`pygame2.sdl.video.Surface.update` of the video surface.

pygame.display.get_driver
  :func:`pygame2.sdl.video.get_drivername`

pygame.display.Info
  :func:`pygame2.sdl.video.get_info`

pygame.display.get_wm_info
  :func:`pygame2.sdl.wm.get_info`

pygame.display.list_modes
  :func:`pygame2.sdl.video.list_modes`

pygame.display.mode_ok
  :func:`pygame2.sdl.video.is_mode_ok`

pygame.display.gl_get_attribute
  :func:`pygame2.sdl.gl.get_attribute`

pygame.display.gl_set_attribute
  :func:`pygame2.sdl.gl.set_attribute`

pygame.display.get_active
  :func:`pygame2.sdl.event.get_app_state` ::

    (pygame2.sdl.event.get_app_state () & pygame2.sdl.constants.APPACTIVE) == pygame2.sdl.constants.APPACTIVE

pygame.display.iconify
  :func:`pygame2.sdl.wm.iconify_window`

pygame.display.toggle_fullscreen
  :func:`pygame2.sdl.wm.toggle_fullscreen`

pygame.display.set_gamma
  :func:`pygame2.sdl.video.set_gamma`

pygame.display.set_icon
  :func:`pygame2.sdl.wm.set_icon`

pygame.display.set_caption
  :func:`pygame2.sdl.wm.set_caption`

pygame.display.get_caption
  :func:`pygame2.sdl.wm.get_caption`

pygame.display.set_palette
  Use :meth:`pygame2.sdl.video.Surface.set_palette` of the video surface.

pygame.draw
-----------

pygame.draw can be found under :mod:`pygame2.sdlext.draw`.

pygame.event
------------

pygame.event
  :mod:`pygame2.sdl.event`

pygame.event.pump
  :func:`pygame2.sdl.event.pump`

pygame.event.get
  :func:`pygame2.sdl.event.get`

pygame.event.poll
  :func:`pygame2.sdl.event.poll`

pygame.event.wait
  :func:`pygame2.sdl.event.wait`

pygame.event.peek
  :func:`pygame2.sdl.event.peek`

pygame.event.clear
  :func:`pygame2.sdl.event.clear`

pygame.event.event_name
  :attr:`pygame2.sdl.event.Event.name`

pygame.event.set_blocked
  :func:`pygame2.sdl.event.set_blocked`

pygame.event.set_allowed
  Use :func:`pygame2.sdl.event.state` with :attr:`pygame2.sdl.constants.ENABLE`

pygame.event.get_blocked
  :func:`pygame2.sdl.event.get_blocked`

pygame.event.set_grab
  :func:`pygame2.sdl.wm.grab_input`

pygame.event.get_grab
  :func:`pygame2.sdl.event.get_app_state` ::

    (pygame2.sdl.event.get_app_state () & (pygame2.sdl.constants.APPMOUSEFOCUS | pygame2.sdl.constants.APPINPUTFOCUS) == 
      (pygame2.sdl.constants.APPMOUSEFOCUS | pygame2.sdl.constants.APPINPUTFOCUS) 

pygame.event.post
  :func:`pygame2.sdl.event.push`

pygame.event.Event
  :class:`pygame2.sdl.event.Event`

pygame.font
-----------

:mod:`pygame.font` separates into two different modules in Pygame2,
:mod:`pygame2.font` for font-file related tasks and
:mod:`pygame2.sdlttf` for the SDL_ttf wrapper.

pygame.font.init
  :func:`pygame2.sdlttf.init`

pygame.font.quit
  :func:`pygame2.sdlttf.quit`

pygame.font.get_init
  :func:`pygame2.sdlttf.was_init`

pygame.font.init
  :func:`pygame2.sdlttf.init`

pygame.font.get_default_font
  There is no similar function or attribute in Pygame2.

pygame.font.get_fonts
  There is no similar function or attribute in Pygame2.

pygame.font.match_font
  :func:`pygame2.font.find_font`

pygame.font.SysFont
  :func:`pygame2.sdlttf.sysfont.get_sys_font`

pygame.font.Font
  :class:`pygame2.sdlttf.Font`

pygame.font.Font.render
  :meth:`pygame2.sdlttf.Font.render`

pygame.font.Font.size
  :meth:`pygame2.sdlttf.Font.get_size`

pygame.font.Font.set_underline
  :attr:`pygame2.sdlttf.Font.style` \|= :attr:`pygame2.sdlttf.constants.STYLE_UNDERLINE` 

pygame.font.Font.get_underline
  :attr:`pygame2.sdlttf.Font.style` ::

    (pygame2.sdlttf.Font.style & pygame2.sdlttf.constants.STYLE_UNDERLINE) ==
       pygame2.sdlttf.constants.STYLE_UNDERLINE

pygame.font.Font.set_bold
  :attr:`pygame2.sdlttf.Font.style` \|= :attr:`pygame2.sdlttf.constants.STYLE_BOLD` 
pygame.font.Font.get_bold
  :attr:`pygame2.sdlttf.Font.style` ::

    (pygame2.sdlttf.Font.style & pygame2.sdlttf.constants.STYLE_BOLD) ==
       pygame2.sdlttf.constants.STYLE_BOLD`

pygame.font.Font.set_italic
  :attr:`pygame2.sdlttf.Font.style` \|= :attr:`pygame2.sdlttf.constants.STYLE_ITALIC` 

pygame.font.Font.get_italic
  :attr:`pygame2.sdlttf.Font.style` ::
  
    (pygame2.sdlttf.Font.style & pygame2.sdlttf.constants.STYLE_ITALIC) ==
       pygame2.sdlttf.constants.STYLE_ITALIC

pygame.font.Font.metrics
  :meth:`pygame2.sdlttf.Font.get_glyph_metrics`

pygame.font.Font.get_linesize
  :attr:`pygame2.sdlttf.Font.line_skip`

pygame.font.Font.get_height
  :attr:`pygame2.sdlttf.Font.height`

pygame.font.Font.get_ascent
  :attr:`pygame2.sdlttf.Font.ascent`

pygame.font.Font.get_descent
  :attr:`pygame2.sdlttf.Font.descent`

pygame.gfxdraw
--------------

pygame.gfxdraw can be found under :mod:`pygame2.sdlgfx`.

pygame.image
------------

:mod:`pygame.image` separates into two different modules in Pygame2,
:mod:`pygame2.sdl.image` for BMP loading and :mod:`pygame2.sdlimage` for
the SDL_image wrapper.

pygame.image.load
  :func:`pygame2.sdl.image.load_bmp` and :func:`pygame2.sdlimage.load`

pygame.image.save
  :func:`pygame2.sdl.image.save_bmp` and :meth:`pygame2.sdl.video.Surface.save`

pygame.image.get_extended
  There is no similar function or attribute in Pygame2. If the
  :mod:`pygame2.sdlimage` module is available and
  :mod:`pygame2.sdl.video` is built with PNG and JPEG support, you can
  assume to have extended format support for loading and saving images.
  
pygame.image.tostring
  Use the :attr:`pygame2.sdl.video.Surface.pixels` attribute.

pygame.image.fromstring
  There is no similar function or attribute in Pygame2. Use the
  :attr:`pygame2.sdl.video.Surface.pixels` attribute to access the raw
  buffer.

pygame.image.frombuffer
  There is no similar function or attribute in Pygame2. Use the
  :attr:`pygame2.sdl.video.Surface.pixels` attribute to access the raw
  buffer.

pygame.joystick
---------------

pygame.joystick
  :mod:`pygame2.sdl.joystick`

pygame.joystick.init
  :func:`pygame2.sdl.joystick.init`

pygame.joystick.quit
  :func:`pygame2.sdl.joystick.quit`

pygame.joystick.get_init
  :func:`pygame2.sdl.joystick.was_init`

pygame.joystick.get_count
  :func:`pygame2.sdl.joystick.num_joysticks`

pygame.joystick.Joystick
  :class:`pygame2.sdl.joystick.Joystick`

pygame.joystick.Joystick.init
  :meth:`pygame2.sdl.joystick.Joystick.open`

pygame.joystick.Joystick.quit
  :meth:`pygame2.sdl.joystick.Joystick.close`

pygame.joystick.Joystick.get_init
  :attr:`pygame2.sdl.joystick.Joystick.opened`

pygame.joystick.Joystick.get_id
  :attr:`pygame2.sdl.joystick.Joystick.index`

pygame.joystick.Joystick.get_name
  :attr:`pygame2.sdl.joystick.Joystick.name`

pygame.joystick.Joystick.get_numaxes
  :attr:`pygame2.sdl.joystick.Joystick.num_axes`

pygame.joystick.Joystick.get_axis
  :attr:`pygame2.sdl.joystick.Joystick.get_axis`

pygame.joystick.Joystick.get_numballs
  :attr:`pygame2.sdl.joystick.Joystick.num_balls`

pygame.joystick.Joystick.get_ball
  :meth:`pygame2.sdl.joystick.Joystick.get_ball`

pygame.joystick.Joystick.get_numbuttons
  :attr:`pygame2.sdl.joystick.Joystick.num_buttons`

pygame.joystick.Joystick.get_button
  :meth:`pygame2.sdl.joystick.Joystick.get_button`

pygame.joystick.Joystick.get_numhats
  :attr:`pygame2.sdl.joystick.Joystick.num_hats`

pygame.joystick.Joystick.get_hat
  :meth:`pygame2.sdl.joystick.Joystick.get_hat`

pygame.key
----------

pygame.key
  :mod:`pygame2.sdl.keyboard`

pygame.key.get_focused
  :func:`pygame2.sdl.event.get_app_state` ::

    (pygame2.sdl.event.get_app_state () & (pygame2.sdl.constants.APPINPUTFOCUS) == pygame2.sdl.constants.APPINPUTFOCUS

pygame.key.get_pressed
  :meth:`pygame2.sdl.keyboard.get_state`

pygame.key.get_mods
  :meth:`pygame2.sdl.keyboard.get_mod_state`

pygame.key.set_mods
  :meth:`pygame2.sdl.keyboard.set_mod_state`

pygame.key.set_repeat
  :meth:`pygame2.sdl.keyboard.enable_repeat`

pygame.key.get_repeat
  :meth:`pygame2.sdl.keyboard.get_repeat`

pygame.key.name
  :meth:`pygame2.sdl.keyboard.get_key_name`

pygame.locals
-------------

Constants are put into the corresponding :mod:`constants` module of the
package, they belong to, e.g. :mod:`pygame2.sdl.constants`,
:mod:`pygame2.sdlttf.constants`, etc.

pygame.mask
-----------

pygame.mask can be found under :mod:`pygame2.mask`.


pygame.midi
-----------

pygame.midi can be found under :mod:`pygame2.midi`.

pygame.mixer
------------

:mod:`pygame.mixer.music` is handled by the
:class:`pygame2.sdlmixer.Music` class within the
:mod:`pygame2.sdlmixer` module.

pygame.mixer
  :mod:`pygame2.sdlmixer`

pygame.mixer.init
  :func:`pygame2.sdlmixer.init` and :func:`pygame2.sdlmixer.open_audio`

pygame.mixer.pre_init
  :func:`pygame2.sdlmixer.open_audio`

pygame.mixer.quit
  :func:`pygame2.sdlmixer.quit` and :func:`pygame2.sdlmixer.close_audio`

pygame.mixer.get_init
  :func:`pygame2.sdlmixer.was_init`

pygame.mixer.stop
  Use :meth:`pygame2.sdlmixer.Channel.halt` or
  :func:`pygame2.sdlmixer.channel.halt`.

pygame.mixer.pause
  Use :meth:`pygame2.sdlmixer.Channel.pause` or
  :func:`pygame2.sdlmixer.channel.pause`.

pygame.mixer.unpause
  Use :meth:`pygame2.sdlmixer.Channel.resume` or
  :func:`pygame2.sdlmixer.channel.resume`.

pygame.mixer.fadeout
  Use :meth:`pygame2.sdlmixer.Channel.fade_out` or
  :func:`pygame2.sdlmixer.channel.fade_out`.

pygame.mixer.set_num_channels
   There is no similar function or attribute in Pygame2. Use
   :meth:`pygame2.sdlmixer.open_audio`.

pygame.mixer.get_num_channels
   There is no similar function or attribute in Pygame2.

pygame.mixer.set_reserved
   There is no similar function or attribute in Pygame2.

pygame.mixer.find_channel
   There is no similar function or attribute in Pygame2.

pygame.mixer.get_busy
   There is no similar function or attribute in Pygame2.

pygame.mixer.Sound
  :class:`pygame2.sdlmixer.Sound` and :class:`pygame2.sdlmixer.Chunk`

pygame.mixer.Sound.play
  There is no similar function or attribute in Pygame2. Use
  :meth:`pygame2.sdlmixer.Channel.play`.

pygame.mixer.Sound.stop
  There is no similar function or attribute in Pygame2. Use
  :meth:`pygame2.sdlmixer.Channel.halt`.

pygame.mixer.Sound.fadeout
   There is no similar function or attribute in Pygame2. Use
   :meth:`pygame2.sdlmixer.Channel.fade_out`.

pygame.mixer.Sound.set_volume
  :attr:`pygame2.sdlmixer.Chunk.volume`

pygame.mixer.Sound.get_volume
  :attr:`pygame2.sdlmixer.Chunk.volume`

pygame.mixer.Sound.get_num_channels
   There is no similar function or attribute in Pygame2.

pygame.mixer.Sound.get_length
  :attr:`pygame2.sdlmixer.Chunk.len`

pygame.mixer.Sound.get_buffer
  :attr:`pygame2.sdlmixer.Chunk.buf`

pygame.mixer.Channel
  :class:`pygame2.sdlmixer.Channel`

pygame.mixer.Channel.play
  :meth:`pygame2.sdlmixer.Channel.play`

pygame.mixer.Channel.stop
  :meth:`pygame2.sdlmixer.Channel.halt`

pygame.mixer.Channel.pause
  :meth:`pygame2.sdlmixer.Channel.pause`

pygame.mixer.Channel.unpause
  :meth:`pygame2.sdlmixer.Channel.resume`

pygame.mixer.Channel.fadeout
  :meth:`pygame2.sdlmixer.Channel.fade_out`

pygame.mixer.Channel.set_volume
  :attr:`pygame2.sdlmixer.Channel.volume`

pygame.mixer.Channel.get_volume
  :attr:`pygame2.sdlmixer.Channel.volume`

pygame.mixer.Channel.get_busy
  :attr:`pygame2.sdlmixer.Channel.playing` and
  :attr:`pygame2.sdlmixer.Channel.paused` and
  :attr:`pygame2.sdlmixer.Channel.fading`

pygame.mixer.Channel.get_sound
  :attr:`pygame2.sdlmixer.Channel.chunk`

pygame.mixer.Channel.queue
   There is no similar function or attribute in Pygame2.
  
pygame.mixer.Channel.get_queue
   There is no similar function or attribute in Pygame2.

pygame.mixer.Channel.set_endevent
   There is no similar function or attribute in Pygame2.

pygame.mixer.Channel.get_endevent
   There is no similar function or attribute in Pygame2.

pygame.mixer.music
------------------

pygame.mixer.music
  :mod:`pygame2.sdlmixer.music`

pygame.mixer.music.load
  Instantiate a :class:`pygame2.sdlmixer.Music` object.

pygame.mixer.music.play
  :meth:`pygame2.sdlmixer.Music.play`

pygame.mixer.music.rewind
  :func:`pygame2.sdlmixer.music.rewind`

pygame.mixer.music.stop
  :func:`pygame2.sdlmixer.music.halt`

pygame.mixer.music.pause
  :func:`pygame2.sdlmixer.music.pause`

pygame.mixer.music.unpause
  :func:`pygame2.sdlmixer.music.resume`

pygame.mixer.music.fadeout
  :func:`pygame2.sdlmixer.music.fade_out`

pygame.mixer.music.set_volume
  :func:`pygame2.sdlmixer.music.set_volume`

pygame.mixer.music.get_volume
  :func:`pygame2.sdlmixer.music.get_volume`

pygame.mixer.music.get_busy
  :func:`pygame2.sdlmixer.music.playing` and
  :func:`pygame2.sdlmixer.music.fading`
  :func:`pygame2.sdlmixer.music.paused`

pygame.mixer.music.get_pos
   There is no similar function or attribute in Pygame2.

pygame.mixer.music.queue
   There is no similar function or attribute in Pygame2.

pygame.mixer.music.set_endevent
   There is no similar function or attribute in Pygame2.

pygame.mixer.music.get_endevent
   There is no similar function or attribute in Pygame2.

pygame.mouse
------------

pygame.mouse
  :mod:`pygame2.sdl.mouse`

pygame.mouse.get_pressed
  :func:`pygame2.sdl.mouse.get_state`

pygame.mouse.get_pos
  :func:`pygame2.sdl.mouse.get_position`

pygame.mouse.get_rel
  :func:`pygame2.sdl.mouse.get_rel_position`

pygame.mouse.set_pos
  :func:`pygame2.sdl.mouse.set_position` and :func:`pygame2.sdl.mouse.warp`

pygame.mouse.set_visible
 :func:`pygame2.sdl.mouse.set_visible` and
 :func:`pygame2.sdl.mouse.show_cursor`

pygame.mouse.get_focused
  :func:`pygame2.sdl.event.get_app_state` ::

    (pygame2.sdl.event.get_app_state () & (pygame2.sdl.constants.APPMOUSEFOCUS) == pygame2.sdl.constants.APPMOUSEFOCUS

pygame.mouse.set_cursor
  :func:`pygame2.sdl.mouse.set_cursor`

pygame.mouse.get_cursor
  There is no similar function or attribute in Pygame2.

pygame.movie
------------

The module is not (yet) ported to Pygame2.

pygame.Overlay
--------------

pygame.Overlay
  :class:`pygame2.sdl.video.Overlay`

pygame.Overlay.display
  :meth:`pygame2.sdl.video.Overlay.display`

pygame.Overlay.set_location
  There is no similar function or attribute in Pygame2.
 
pygame.Overlay.get_hardware
  :attr:`pygame2.sdl.video.Overlay.hw_overlay`

pygame.PixelArray
-----------------

pygame.PixelArray can be found under :class:`pygame2.sdlext.PixelArray`.

pygame.Rect
-----------

No notable changes apply here.

pygame.scrap
------------

pygame.scrap can be found under :mod:`pygame2.sdlext.scrap`.

pygame.sndarray
---------------

pygame.sndarray can be found under :mod:`pygame2.sdlmixer.sndarray`.

pygame.sprite
-------------

No notable changes apply here.

pygame.surfarray
----------------

pygame.surfarray can be found under :mod:`pygame2.sdlext.surfarray`.

pygame.Surface
--------------

pygame.Surface
  :class:`pygame2.sdl.video.Surface`

pygame.Surface.
  :meth:`pygame2.sdl.video.Surface.blit`

pygame.Surface.convert
  :meth:`pygame2.sdl.video.Surface.convert`

pygame.Surface.convert_alpha
  :meth:`pygame2.sdl.video.Surface.convert`

pygame.Surface.copy
  :meth:`pygame2.sdl.video.Surface.copy`

pygame.Surface.fill
  :meth:`pygame2.sdl.video.Surface.fill`

pygame.Surface.scroll
  :meth:`pygame2.sdl.video.Surface.scroll`

pygame.Surface.set_colorkey
  :meth:`pygame2.sdl.video.Surface.set_colorkey`

pygame.Surface.get_colorkey
  :meth:`pygame2.sdl.video.Surface.get_colorkey`

pygame.Surface.set_alpha
  :meth:`pygame2.sdl.video.Surface.set_alpha`

pygame.Surface.get_alpha
  :meth:`pygame2.sdl.video.Surface.get_alpha`

pygame.Surface.lock
  :meth:`pygame2.sdl.video.Surface.lock`

pygame.Surface.unlock
  :meth:`pygame2.sdl.video.Surface.unlock`

pygame.Surface.mustlock
  There is no similar function or attribute in Pygame2.
  
pygame.Surface.get_locked
  :attr:`pygame2.sdl.video.Surface.locked`

pygame.Surface.get_locks
  There is no similar function or attribute in Pygame2.

pygame.Surface.get_at
  :meth:`pygame2.sdl.video.Surface.get_at`

pygame.Surface.set_at
  :meth:`pygame2.sdl.video.Surface.set_at`

pygame.Surface.get_palette
  :meth:`pygame2.sdl.video.Surface.get_palette`

pygame.Surface.get_palette_at
  There is no similar function or attribute in Pygame2.

pygame.Surface.set_palette
  :meth:`pygame2.sdl.video.Surface.set_palette`

pygame.Surface.set_palette_at
  There is no similar function or attribute in Pygame2.

pygame.Surface.map_rgb
  :meth:`pygame2.sdl.video.PixelFormat.map_rgba` of
  :attr:`pygame2.sdl.video.Surface.format`

pygame.Surface.unmap_rgb
  :meth:`pygame2.sdl.video.PixelFormat.get_rgba` of
  :attr:`pygame2.sdl.video.Surface.format`

pygame.Surface.set_clip
  :attr:`pygame2.sdl.video.Surface.clip_rect`

pygame.Surface.get_clip
  :attr:`pygame2.sdl.video.Surface.clip_rect`

pygame.Surface.subsurface
  There is no similar function or attribute in Pygame2.

pygame.Surface.get_parent
  There is no similar function or attribute in Pygame2.

pygame.Surface.get_abs_parent
  There is no similar function or attribute in Pygame2.

pygame.Surface.get_size
  :attr:`pygame2.sdl.video.Surface.size`

pygame.Surface.get_width
  :attr:`pygame2.sdl.video.Surface.width` or
  :attr:`pygame2.sdl.video.Surface.w`

pygame.Surface.get_height
  :attr:`pygame2.sdl.video.Surface.height` or
  :attr:`pygame2.sdl.video.Surface.h`

pygame.Surface.get_rect
  There is no similar function or attribute in Pygame2.

pygame.Surface.get_bitsize
  :attr:`pygame2.sdl.video.PixelFormat.bits_per_pixel` of
  :attr:`pygame2.sdl.video.Surface.format`

pygame.Surface.get_bytesize
  :attr:`pygame2.sdl.video.PixelFormat.bytes_per_pixel` of
  :attr:`pygame2.sdl.video.Surface.format`

pygame.Surface.get_flags
  :attr:`pygame2.sdl.video.Surface.flags`

pygame.Surface.get_pitch
  :attr:`pygame2.sdl.video.Surface.pitch`

pygame.Surface.get_masks
  :attr:`pygame2.sdl.video.PixelFormat.masks` of
  :attr:`pygame2.sdl.video.Surface.format`

pygame.Surface.set_masks
  There is no similar function or attribute in Pygame2.

pygame.Surface.get_shifts
  :attr:`pygame2.sdl.video.PixelFormat.shifts` of
  :attr:`pygame2.sdl.video.Surface.format`

pygame.Surface.set_shifts
  There is no similar function or attribute in Pygame2.

pygame.Surface.get_losses
  :attr:`pygame2.sdl.video.PixelFormat.losses` of
  :attr:`pygame2.sdl.video.Surface.format`

pygame.Surface.get_bounding_rect
  There is no similar function or attribute in Pygame2.

pygame.Surface.get_buffer
  :attr:`pygame2.sdl.video.Surface.pixels`

pygame.time
-----------

pygame.time
  :mod:`pygame2.sdl.time`

pygame.time.get_ticks
  :func:`pygame2.sdl.time.get_ticks`

pygame.time.wait
  :func:`pygame2.sdl.time.delay`

pygame.time.delay
  :func:`pygame2.sdl.time.delay`

pygame.time.set_timer
  Use :func:`pygame2.sdl.time.add_timer`

pygame.time.Clock
  There is no similar class in Pygame2.
