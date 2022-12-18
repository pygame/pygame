import unittest, unittest.mock
import io

import pygame


class DebugTest(unittest.TestCase):
    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def assert_stdout(self, expected_output, mock_stdout):
        pygame.print_debug_info()
        self.assertEqual(mock_stdout.getvalue(), expected_output)

    def test_print_debug(self):
        import os

        pygame.print_debug_info("temp_file.txt")
        with open("temp_file.txt", "r") as temp_file:
            text = temp_file.read()

        self.assertNotEqual(text, "")
        self.assert_stdout(text + "\n")

        os.remove("temp_file.txt")
