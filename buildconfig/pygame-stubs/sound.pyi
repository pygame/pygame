from typing import Union, Any, IO

import numpy

from pygame.mixer import Sound


def load(file: Union[IO, numpy.ndarray, Any]) -> Sound: ...
