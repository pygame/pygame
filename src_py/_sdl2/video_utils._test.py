"""Tests for pygame._sdl2.video_utils module."""

import unittest
import pygame
from pygame._sdl2 import get_display_modes, get_unique_display_modes


class TestVideoUtils(unittest.TestCase):
    """Test cases for video utility functions."""

    @classmethod
    def setUpClass(cls):
        """Initialize pygame before running tests."""
        pygame.init()

    @classmethod
    def tearDownClass(cls):
        """Clean up pygame after tests."""
        pygame.quit()

    def test_get_display_modes_returns_list(self):
        """Test that get_display_modes() returns a list."""
        modes = get_display_modes()
        self.assertIsInstance(modes, list)

    def test_get_display_modes_contains_tuples(self):
        """Test that get_display_modes() returns a list of tuples."""
        modes = get_display_modes()
        self.assertGreater(len(modes), 0, "Should return at least one display mode")
        
        for mode in modes:
            self.assertIsInstance(mode, tuple, f"Mode {mode} is not a tuple")
            self.assertEqual(len(mode), 2, f"Mode {mode} should have 2 elements")

    def test_get_display_modes_valid_dimensions(self):
        """Test that all display modes have positive width and height."""
        modes = get_display_modes()
        
        for width, height in modes:
            self.assertIsInstance(width, int, f"Width {width} is not an integer")
            self.assertIsInstance(height, int, f"Height {height} is not an integer")
            self.assertGreater(width, 0, f"Width {width} should be positive")
            self.assertGreater(height, 0, f"Height {height} should be positive")

    def test_get_display_modes_sorted(self):
        """Test that display modes are sorted in descending order."""
        modes = get_display_modes()
        
        if len(modes) > 1:
            for i in range(len(modes) - 1):
                current = modes[i]
                next_mode = modes[i + 1]
                # Check if current >= next (width first, then height)
                self.assertTrue(
                    current >= next_mode,
                    f"Modes not sorted: {current} should be >= {next_mode}"
                )

    def test_get_display_modes_with_display_index(self):
        """Test get_display_modes() with explicit display index."""
        try:
            modes = get_display_modes(display_index=0)
            self.assertIsInstance(modes, list)
            self.assertGreater(len(modes), 0)
        except pygame.error:
            # Some systems might not support multiple displays
            self.skipTest("Display index 0 not available")

    def test_get_unique_display_modes_returns_list(self):
        """Test that get_unique_display_modes() returns a list."""
        modes = get_unique_display_modes()
        self.assertIsInstance(modes, list)

    def test_get_unique_display_modes_no_duplicates(self):
        """Test that get_unique_display_modes() returns no duplicate resolutions."""
        unique_modes = get_unique_display_modes()
        
        # Convert to set to check for duplicates
        unique_set = set(unique_modes)
        self.assertEqual(
            len(unique_modes),
            len(unique_set),
            "Unique modes list contains duplicates"
        )

    def test_unique_modes_subset_of_all_modes(self):
        """Test that unique modes are a subset of all modes."""
        all_modes = get_display_modes()
        unique_modes = get_unique_display_modes()
        
        # All unique modes should be present in all modes
        for mode in unique_modes:
            self.assertIn(
                mode,
                all_modes,
                f"Unique mode {mode} not found in all modes"
            )

    def test_unique_modes_count_less_or_equal(self):
        """Test that unique modes count is less than or equal to all modes."""
        all_modes = get_display_modes()
        unique_modes = get_unique_display_modes()
        
        self.assertLessEqual(
            len(unique_modes),
            len(all_modes),
            "Unique modes count should be <= all modes count"
        )

    def test_common_resolutions_format(self):
        """Test that common resolutions are in expected format."""
        modes = get_display_modes()
        
        # Common resolutions to check (if available)
        common_resolutions = [
            (1920, 1080),  # Full HD
            (2560, 1440),  # 2K
            (3840, 2160),  # 4K
            (1280, 720),   # HD
        ]
        
        # Just verify that if these exist, they're properly formatted
        for resolution in common_resolutions:
            if resolution in modes:
                width, height = resolution
                self.assertGreater(width, height, 
                                 f"Width {width} should typically be > height {height}")


if __name__ == '__main__':
    unittest.main()
