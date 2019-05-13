.. include:: common.txt

:mod:`pygame.mixer.music`
=========================

.. module:: pygame.mixer.music
   :synopsis: pygame module for controlling streamed audio

| :sl:`pygame module for controlling streamed audio`

The music module is closely tied to :mod:`pygame.mixer`. Use the music module
to control the playback of music in the sound mixer.

The difference between the music playback and regular Sound playback is that
the music is streamed, and never actually loaded all at once. The mixer system
only supports a single music stream at once.

Be aware that ``MP3`` support is limited. On some systems an unsupported format
can crash the program, ``e.g``. Debian Linux. Consider using ``OGG`` instead.

.. function:: load

   | :sl:`Load a music file for playback`
   | :sg:`load(filename) -> None`
   | :sg:`load(object) -> None`

   This will load a music filename/file object and prepare it for playback. If
   a music stream is already playing it will be stopped. This does not start
   the music playing.

   .. ## pygame.mixer.music.load ##

.. function:: play

   | :sl:`Start the playback of the music stream`
   | :sg:`play(loops=0, start=0.0) -> None`

   This will play the loaded music stream. If the music is already playing it
   will be restarted.

   The loops argument controls the number of repeats a music will play.
   ``play(5)`` will cause the music to played once, then repeated five times,
   for a total of six. If the loops is -1 then the music will repeat
   indefinitely.

   The starting position argument controls where in the music the song starts
   playing. The starting position is dependent on the format of music playing.
   ``MP3`` and ``OGG`` use the position as time (in seconds). ``MOD`` music it
   is the pattern order number. Passing a startpos will raise a
   NotImplementedError if it cannot set the start position

   .. ## pygame.mixer.music.play ##

.. function:: rewind

   | :sl:`restart music`
   | :sg:`rewind() -> None`

   Resets playback of the current music to the beginning.

   .. ## pygame.mixer.music.rewind ##

.. function:: stop

   | :sl:`stop the music playback`
   | :sg:`stop() -> None`

   Stops the music playback if it is currently playing.

   .. ## pygame.mixer.music.stop ##

.. function:: pause

   | :sl:`temporarily stop music playback`
   | :sg:`pause() -> None`

   Temporarily stop playback of the music stream. It can be resumed with the
   ``pygame.mixer.music.unpause()`` function.

   .. ## pygame.mixer.music.pause ##

.. function:: unpause

   | :sl:`resume paused music`
   | :sg:`unpause() -> None`

   This will resume the playback of a music stream after it has been paused.

   .. ## pygame.mixer.music.unpause ##

.. function:: fadeout

   | :sl:`stop music playback after fading out`
   | :sg:`fadeout(time) -> None`

   This will stop the music playback after it has been faded out over the
   specified time (measured in milliseconds).

   Note, that this function blocks until the music has faded out.

   .. ## pygame.mixer.music.fadeout ##

.. function:: set_volume

   | :sl:`set the music volume`
   | :sg:`set_volume(value) -> None`

   Set the volume of the music playback. The value argument is between 0.0 and
   1.0. When new music is loaded the volume is reset.

   .. ## pygame.mixer.music.set_volume ##

.. function:: get_volume

   | :sl:`get the music volume`
   | :sg:`get_volume() -> value`

   Returns the current volume for the mixer. The value will be between 0.0 and
   1.0.

   .. ## pygame.mixer.music.get_volume ##

.. function:: get_busy

   | :sl:`check if the music stream is playing`
   | :sg:`get_busy() -> bool`

   Returns True when the music stream is actively playing. When the music is
   idle this returns False.

   .. ## pygame.mixer.music.get_busy ##

.. function:: set_pos

   | :sl:`set position to play from`
   | :sg:`set_pos(pos) -> None`

   This sets the position in the music file where playback will start.
   The meaning of "pos", a float (or a number that can be converted to a float),
   depends on the music format. For ``MOD`` files, it is the integer pattern
   number in the module. For ``OGG`` it the absolute position, in seconds, from
   the beginning of the sound. For ``MP3`` files, it is the relative position,
   in seconds, from the current position. For absolute positioning in an ``MP3``
   file, first call :func:`rewind`.
   Other file formats are unsupported. Newer versions of SDL_mixer have
   better positioning support than earlier. An SDLError is raised if a
   particular format does not support positioning.

   Function :func:`set_pos` calls underlining SDL_mixer function
   ``Mix_SetMusicPosition``.

   .. versionadded:: 1.9.2

   .. ## pygame.mixer.music.set_pos ##

.. function:: get_pos

   | :sl:`get the music play time`
   | :sg:`get_pos() -> time`

   This gets the number of milliseconds that the music has been playing for.
   The returned time only represents how long the music has been playing; it
   does not take into account any starting position offsets.

   .. ## pygame.mixer.music.get_pos ##

.. function:: queue

   | :sl:`queue a music file to follow the current`
   | :sg:`queue(filename) -> None`

   This will load a music file and queue it. A queued music file will begin as
   soon as the current music naturally ends. If the current music is ever
   stopped or changed, the queued song will be lost.

   The following example will play music by Bach six times, then play music by
   Mozart once:

   ::

       pygame.mixer.music.load('bach.ogg')
       pygame.mixer.music.play(5)        # Plays six times, not five!
       pygame.mixer.music.queue('mozart.ogg')

   .. ## pygame.mixer.music.queue ##

.. function:: set_endevent

   | :sl:`have the music send an event when playback stops`
   | :sg:`set_endevent() -> None`
   | :sg:`set_endevent(type) -> None`

   This causes pygame to signal (by means of the event queue) when the music is
   done playing. The argument determines the type of event that will be queued.

   The event will be queued every time the music finishes, not just the first
   time. To stop the event from being queued, call this method with no
   argument.

   .. ## pygame.mixer.music.set_endevent ##

.. function:: get_endevent

   | :sl:`get the event a channel sends when playback stops`
   | :sg:`get_endevent() -> type`

   Returns the event type to be sent every time the music finishes playback. If
   there is no endevent the function returns ``pygame.NOEVENT``.

   .. ## pygame.mixer.music.get_endevent ##

.. ## pygame.mixer.music ##
