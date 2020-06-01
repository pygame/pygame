"""pygame.sound

pygame wrapper module for loading a sound. The real work is done by
the mixer module.
"""

import pygame.mixer


__all__ = ['load']


def load(file):
    """
    Load a sound file and return a Sound object. This is a loose wrapper
    around the mixer.Sound class initializer to give a similar interface to
    image.load().

    :param file: The file to load. This can be a buffer or a file name.

    :return: A mixer.Sound object.
    """

    # check we have initialised the mixer and initialise it if we haven't
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    return pygame.mixer.Sound(file)
