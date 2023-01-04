from typing import List, Optional

from typing_extensions import TypedDict

# dict at runtime, TypedDict exists solely for the typechecking benefits
class _Locale(TypedDict):
    language: str
    country: Optional[str]

def get_pref_path(org: str, app: str) -> str: ...
def get_pref_locales() -> List[_Locale]: ...
