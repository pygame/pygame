.. include:: common.txt

:mod:`pygame.movie`
===================

.. module:: pygame.movie
   :synopsis: pygame module for playback of mpeg video

| :sl:`pygame module for playback of mpeg video`

``NOTE``: On ``NT`` derived Windows versions (e.g. ``XT``) the default ``SDL``
directx video driver is problematic. For :mod:`pygame.movie`, use the windib
driver instead. To enable windib set the ``SDL_VIDEODRIVER`` environment
variable to 'windib' before importing pygame (see the
:func:`pygame.examples.movieplayer.main` example).

Pygame can playback video and audio from basic encoded MPEG-1 video files.
Movie playback happens in background threads, which makes playback easy to
manage.

The audio for Movies must have full control over the sound system. This means
the :mod:`pygame.mixer` module must be uninitialized if the movie's sound is to
be played. The common solution is to call ``pygame.mixer.quit()`` before the
movie begins. The mixer can be reinitialized after the movie is finished.

The video overlay planes are drawn on top of everything in the display window.
To draw the movie as normal graphics into the display window, create an
offscreen Surface and set that as the movie target. Then once per frame blit
that surface to the screen.

Videos can be converted to the .mpg file format (MPEG-1 video, MPEG-1 Audio
Layer ``III`` (MP3) sound) using ffmpeg video conversion program
(http://ffmpeg.org/):

::

  ffmpeg -i <infile> -vcodec mpeg1video -acodec libmp3lame -intra <outfile.mpg>

.. class:: Movie

   | :sl:`load an mpeg movie file`
   | :sg:`Movie(filename) -> Movie`
   | :sg:`Movie(object) -> Movie`

   Load a new ``MPEG`` movie stream from a file or a python file object. The
   Movie object operates similar to the Sound objects from :mod:`pygame.mixer`.

   Movie objects have a target display Surface. The movie is rendered into this
   Surface in a background thread. If the target Surface is the display
   Surface, the movie will try to use the hardware accelerated video overlay.
   The default target is the display Surface.

   .. method:: play

      | :sl:`start playback of a movie`
      | :sg:`play(loops=0) -> None`

      Starts playback of the movie. Sound and video will begin playing if they
      are not disabled. The optional loops argument controls how many times the
      movie will be repeated. A loop value of -1 means the movie will repeat
      forever.

      .. ## Movie.play ##

   .. method:: stop

      | :sl:`stop movie playback`
      | :sg:`stop() -> None`

      Stops the playback of a movie. The video and audio playback will be
      stopped at their current position.

      .. ## Movie.stop ##

   .. method:: pause

      | :sl:`temporarily stop and resume playback`
      | :sg:`pause() -> None`

      This will temporarily stop or restart movie playback.

      .. ## Movie.pause ##

   .. method:: skip

      | :sl:`advance the movie playback position`
      | :sg:`skip(seconds) -> None`

      Advance the movie playback time in seconds. This can be called before the
      movie is played to set the starting playback time. This can only skip the
      movie forward, not backwards. The argument is a floating point number.

      .. ## Movie.skip ##

   .. method:: rewind

      | :sl:`restart the movie playback`
      | :sg:`rewind() -> None`

      Sets the movie playback position to the start of the movie. The movie
      will automatically begin playing even if it stopped.

      The can raise a ValueError if the movie cannot be rewound. If the rewind
      fails the movie object is considered invalid.

      .. ## Movie.rewind ##

   .. method:: render_frame

      | :sl:`set the current video frame`
      | :sg:`render_frame(frame_number) -> frame_number`

      This takes an integer frame number to render. It attempts to render the
      given frame from the movie to the target Surface. It returns the real
      frame number that got rendered.

      .. ## Movie.render_frame ##

   .. method:: get_frame

      | :sl:`get the current video frame`
      | :sg:`get_frame() -> frame_number`

      Returns the integer frame number of the current video frame.

      .. ## Movie.get_frame ##

   .. method:: get_time

      | :sl:`get the current vide playback time`
      | :sg:`get_time() -> seconds`

      Return the current playback time as a floating point value in seconds.
      This method currently seems broken and always returns 0.0.

      .. ## Movie.get_time ##

   .. method:: get_busy

      | :sl:`check if the movie is currently playing`
      | :sg:`get_busy() -> bool`

      Returns true if the movie is currently being played.

      .. ## Movie.get_busy ##

   .. method:: get_length

      | :sl:`the total length of the movie in seconds`
      | :sg:`get_length() -> seconds`

      Returns the length of the movie in seconds as a floating point value.

      .. ## Movie.get_length ##

   .. method:: get_size

      | :sl:`get the resolution of the video`
      | :sg:`get_size() -> (width, height)`

      Gets the resolution of the movie video. The movie will be stretched to
      the size of any Surface, but this will report the natural video size.

      .. ## Movie.get_size ##

   .. method:: has_video

      | :sl:`check if the movie file contains video`
      | :sg:`has_video() -> bool`

      True when the opened movie file contains a video stream.

      .. ## Movie.has_video ##

   .. method:: has_audio

      | :sl:`check if the movie file contains audio`
      | :sg:`has_audio() -> bool`

      True when the opened movie file contains an audio stream.

      .. ## Movie.has_audio ##

   .. method:: set_volume

      | :sl:`set the audio playback volume`
      | :sg:`set_volume(value) -> None`

      Set the playback volume for this movie. The argument is a value between
      0.0 and 1.0. If the volume is set to 0 the movie audio will not be
      decoded.

      .. ## Movie.set_volume ##

   .. method:: set_display

      | :sl:`set the video target Surface`
      | :sg:`set_display(Surface, rect=None) -> None`

      Set the output target Surface for the movie video. You may also pass a
      rectangle argument for the position, which will move and stretch the
      video into the given area.

      If None is passed as the target Surface, the video decoding will be
      disabled.

      .. ## Movie.set_display ##

   .. ## pygame.movie.Movie ##

.. ## pygame.movie ##
