import unittest
import pygame
from pygame import touch


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

    @unittest.skipIf(not has_touchdevice, 'no touch devices found')
    def test_get_device(self):
        touch.get_device(0)

    def test_num_fingers__invalid(self):
        self.assertRaises(pygame.error, touch.get_device, -1234)
        self.assertRaises(TypeError, touch.get_device, 'test')

    @unittest.skipIf(not has_touchdevice, 'no touch devices found')
    def test_num_fingers(self):
        touch.get_num_fingers(touch.get_device(0))

    def test_num_fingers__invalid(self):
        self.assertRaises(TypeError, touch.get_num_fingers, 'test')
        self.assertRaises(pygame.error, touch.get_num_fingers, -1234)

    @unittest.skipIf(not has_touchdevice, 'no touch devices found')
    def todo_test_get_finger(self):
        """ask for touch input and check the dict"""


if __name__ == '__main__':
    unittest.main()
