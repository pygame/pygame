"""Utility functions for working with SDL2 video functionality."""

from typing import List, Tuple
from pygame._sdl2 import video


def get_display_modes(display_index: int = 0) -> List[Tuple[int, int]]:
    """
    Get all available fullscreen display resolutions for a specific display.

    This function retrieves all display modes supported by the graphics driver
    for a given display device. Each mode represents a resolution that can be
    used for fullscreen applications.

    Args:
        display_index: Index of the display device (default: 0 for primary display).

    Returns:
        A list of tuples containing (width, height) for each available display mode.
        The list is sorted by resolution (width first, then height) in descending order.

    Raises:
        pygame.error: If the display index is invalid or SDL2 cannot retrieve modes.

    Example:
        >>> import pygame
        >>> pygame.init()
        >>> from pygame._sdl2 import get_display_modes
        >>> modes = get_display_modes()
        >>> print(f"Available resolutions: {len(modes)}")
        >>> print(f"Highest resolution: {modes[0]}")
        >>> print(f"Common 1080p available: {(1920, 1080) in modes}")
        >>> pygame.quit()

    Note:
        - pygame.init() or pygame.display.init() must be called before using this function.
        - The returned modes depend on your hardware and driver capabilities.
        - Duplicate resolutions with different refresh rates are included as separate entries.
    """
    desktop_modes = video.get_desktop_display_modes(display_index)
    
    # Extract (width, height) tuples from the display mode objects
    modes = [(mode.w, mode.h) for mode in desktop_modes]
    
    # Sort by width (descending), then height (descending)
    modes.sort(reverse=True)
    
    return modes


def get_unique_display_modes(display_index: int = 0) -> List[Tuple[int, int]]:
    """
    Get unique fullscreen display resolutions, removing duplicate refresh rates.

    This function is similar to get_display_modes() but removes duplicate
    resolutions that differ only in refresh rate, returning only unique
    (width, height) combinations.

    Args:
        display_index: Index of the display device (default: 0 for primary display).

    Returns:
        A list of unique tuples containing (width, height) for each resolution.
        The list is sorted by resolution (width first, then height) in descending order.

    Raises:
        pygame.error: If the display index is invalid or SDL2 cannot retrieve modes.

    Example:
        >>> import pygame
        >>> pygame.init()
        >>> from pygame._sdl2 import get_unique_display_modes
        >>> unique_modes = get_unique_display_modes()
        >>> print(f"Unique resolutions: {len(unique_modes)}")
        >>> for width, height in unique_modes[:5]:
        ...     print(f"{width}x{height}")
        >>> pygame.quit()
    """
    modes = get_display_modes(display_index)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_modes = []
    for mode in modes:
        if mode not in seen:
            seen.add(mode)
            unique_modes.append(mode)
    
    return unique_modes
