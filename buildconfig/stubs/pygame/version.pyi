from typing import Tuple, Literal

class SoftwareVersion(Tuple[int, int, int]):
    def __new__(cls, major: int, minor: int, patch: int) -> PygameVersion: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    major: int
    minor: int
    patch: int
    fields: Tuple[Literal["major"], Literal["minor"], Literal["patch"]]

class PygameVersion(SoftwareVersion): ...
class SDLVersion(SoftwareVersion): ...

SDL: SDLVersion
ver: str
vernum: PygameVersion
rev: str
