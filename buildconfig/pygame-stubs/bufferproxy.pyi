import sys
from typing import Any, overload, Optional

class BufferProxy(object):
    parent: Any
    length: int

    if sys.version_info > (3,):
        raw: bytes
    else:
        raw: str
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, parent: Any) -> None: ...
    def write(self, buffer: bytes, offset: int = 0) -> None: ...
