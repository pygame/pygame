from typing import Callable, Optional


def set_post_mix(mix_func: Callable[[bytes, bytes, Optional[int]], None]) -> None: ...
