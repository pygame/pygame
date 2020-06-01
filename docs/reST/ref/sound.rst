.. include:: common.txt

:mod:`pygame.sound`
===================

.. module:: pygame.sound
   :synopsis: pygame module for loading sounds

| :sl:`pygame module for sound loading`

The sound module is a thin wrapper around :mod:`pygame.mixer` and it's Sound
class. For more extensive documentation you should check there.

.. versionadded:: 2.0

.. function:: load

   | :sl:`load new sound from a file`
   | :sg:`load(file) -> mixer.Sound`

   Load a sound from a file source. You can pass either a file path or a Python
   file-like object.

   Pygame will automatically determine the sound type (your choices are
   uncompressed ``wav`` or ``ogg``) and create a new Sound object from the
   data.

   The returned Sound can be played by calling it's ``.play()`` method.

   This function is a thin wrapper around the constructor of the
   pygame.mixer.Sound class. See :mod:`pygame.mixer` which has more detailed
   documentation on the capabilities of the Sound class.

   You should use ``os.path.join()`` for multi-platform compatibility when
   loading from paths.

   Example usage:
   ::

     my_sound = pygame.sound.load(os.path.join('data', 'metronome.wav'))
     my_sound.play()

   .. ## pygame.sound.load ##

.. ## pygame.sound ##