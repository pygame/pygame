from typing import Optional, Type, overload, Any

from ._common import _AnyPath

def encode_string(
    obj: Optional[_AnyPath],
    encoding: Optional[str] = "unicode_escape",
    errors: Optional[str] = "backslashreplace",
    etype: Optional[Type[Exception]] = UnicodeEncodeError,
) -> bytes: ...
@overload
def encode_file_path(
    obj: Optional[_AnyPath], etype: Optional[Type[Exception]] = UnicodeEncodeError
) -> bytes: ...
@overload
def encode_file_path(
    obj: Any, etype: Optional[Type[Exception]] = UnicodeEncodeError
) -> bytes: ...
