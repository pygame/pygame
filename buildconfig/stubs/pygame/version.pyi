"""
Simply the current installed pygame version. The version information is
stored in the regular pygame module as 'pygame.ver'. Keeping the version
information also available in a separate module allows you to test the
pygame version without importing the main pygame module.

The python version information should always compare greater than any previous
releases. (hmm, until we get to versions > 10)
"""
from typing import Tuple

from ._common import Literal

class SoftwareVersion(Tuple[int, int, int]):
    def __new__(cls, major: int, minor: int, patch: int) -> SoftwareVersion: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @property
    def major(self) -> int: ...
    @property
    def minor(self) -> int: ...
    @property
    def patch(self) -> int: ...
    fields: Tuple[Literal["major"], Literal["minor"], Literal["patch"]]

class PygameVersion(SoftwareVersion): ...
class SDLVersion(SoftwareVersion): ...

SDL: SDLVersion
ver: str
vernum: PygameVersion
rev: str
