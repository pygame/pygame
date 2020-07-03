import unittest
import pygame
from pygame._sdl2 import touch


has_touchdevice = touch.get_num_devices() > 0


class TouchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.display.init()

    @classmethod
    def tearDownClass(cls):
        pygame.display.quit()

    def test_num_devices(self):
        touch.get_num_devices()

    @unittest.skipIf(not has_touchdevice, "no touch devices found")
    def test_get_device(self):
        touch.get_device(0)

    def test_num_fingers__invalid(self):
        self.assertRaises(pygame.error, touch.get_device, -1234)
        self.assertRaises(TypeError, touch.get_device, "test")

    @unittest.skipIf(not has_touchdevice, "no touch devices found")
    def test_num_fingers(self):
        touch.get_num_fingers(touch.get_device(0))

    def test_num_fingers__invalid(self):
        self.assertRaises(TypeError, touch.get_num_fingers, "test")
        self.assertRaises(pygame.error, touch.get_num_fingers, -1234)

    @unittest.skipIf(not has_touchdevice, "no touch devices found")
    def test_get_finger(self):
        """ask for touch input and check the dict"""

        # parameters: touch device id and index of finger to get info about
        touchid = touch.get_device(0)
        index = 0

        dict = touch.get_finger(touchid, index)

        # check resulting dict's keys
        self.assertIs(type(dict['id']),int)

        self.assertIs(type(dict['x']), float)
        self.assertTrue(dict['x'] >= 0.0 and dict['x'] <= 1.0)

        self.assertIs(type(dict['y']), float)
        self.assertTrue(dict['y'] >= 0.0 and dict['y'] <= 1.0)

        self.assertIs(type(dict['pressure']), float)
        self.assertIs(dict['pressure'] >= 0.0 and dict['pressure'] <= 1.0)


if __name__ == "__main__":
    unittest.main()
