import unittest
import pygame
import pygame.camera


class CameraModuleTest(unittest.TestCase):
    pass


class TestPygameCamera(unittest.TestCase):
    def setUp(self):
        pygame.init()
        pygame.camera.init()

    def test_camera(self):
        cameras = pygame.camera.list_cameras()
        self.assertTrue(len(cameras) > 0, "No cameras found")

        cam = pygame.camera.Camera(cameras[0], (640, 480))
        cam.start()
        image = cam.get_image()
        self.assertIsNotNone(image, "Could not capture image")
        cam.stop()

    def tearDown(self):
        pygame.camera.quit()
        pygame.quit()

if __name__ == '__main__':
    unittest.main()

