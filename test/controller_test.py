import unittest
from unittest import runner
import pygame
import pygame._sdl2.controller as controller
from pygame.tests.test_utils import prompt, question

class ControllerModuleTest(unittest.TestCase):
    def setUp(self):
        controller.init()

    def tearDown(self):
        controller.quit()

    def test_init(self):
        controller.quit()
        controller.init()
        self.assertTrue(controller.get_init())

    def test_init__multiple(self):
        controller.init()
        controller.init()
        self.assertTrue(controller.get_init())

    def test_quit(self):
        controller.quit()
        self.assertFalse(controller.get_init())

    def test_quit__multiple(self):
        controller.quit()
        controller.quit()
        self.assertFalse(controller.get_init())

    def test_get_init(self):
        self.assertTrue(controller.get_init())

    def test_set_eventstate(self):
        pygame.display.init()
        pygame.display.set_mode((5, 5))
        pygame.event.clear()
        for i in range(controller.get_count()):
            if controller.is_controller(i):
                controller.Controller(i)
                break
        else:
            self.skipTest("Controller is not connected therefore controller related events wont be generated")
            pygame.display.quit()

        controller.set_eventstate(True)
        for _ in range(50):
            pygame.event.pump()
            pygame.time.wait(20)
            try:
                self.assertTrue(pygame.event.peek(pygame.CONTROLLERAXISMOTION))
                break
            except AssertionError:
                pass
        else:
            pygame.display.quit()
            self.fail()

        pygame.event.clear()

        controller.set_eventstate(False)
        for _ in range(50):
            pygame.event.pump()
            pygame.time.wait(10)
        self.assertFalse(pygame.event.peek(pygame.CONTROLLERAXISMOTION))

        pygame.display.quit()

    def test_get_eventstate(self):
        controller.set_eventstate(True)
        self.assertTrue(controller.get_eventstate())

        controller.set_eventstate(False)
        self.assertFalse(controller.get_eventstate())

        controller.set_eventstate(True)

    def test_get_count(self):
        self.assertGreaterEqual(controller.get_count(), 0)

    def test_is_controller(self):
        for i in range(controller.get_count()):
            if controller.is_controller(i):
                c = controller.Controller(i)
                self.assertIsInstance(c, controller.Controller)
                c.quit()
            else:
                with self.assertRaises(pygame._sdl2.sdl2.error):
                    c = controller.Controller(i)

        with self.assertRaises(TypeError):
            controller.is_controller('Test')

    def test_name_forindex(self):
        self.assertIsNone(controller.name_forindex(-1))
        with self.assertRaises(TypeError):
            controller.name_forindex('Test')


class ControllerTypeTest(unittest.TestCase):
    def setUp(self):
        controller.init()

    def tearDown(self):
        controller.quit()

    def _get_first_controller(self):
        for i in range(controller.get_count()):
            if controller.is_controller(i):
                return controller.Controller(i)

    def test_construction(self):
        c = self._get_first_controller()
        if c:
            self.assertIsInstance(c, controller.Controller)
        else:
            self.skipTest("No controller connected")

    def test__auto_init(self):
        c = self._get_first_controller()
        if c:
            self.assertTrue(c.get_init())
        else:
            self.skipTest("No controller connected")

    def test_get_init(self):
        c = self._get_first_controller()
        if c:
            self.assertTrue(c.get_init())
            c.quit()
            self.assertFalse(c.get_init())
        else:
            self.skipTest("No controller connected")

    def test_from_joystick(self):
        for i in range(controller.get_count()):
            if controller.is_controller(i):
                joy = pygame.joystick.Joystick(i)
                break
        else:
            self.skipTest("No controller connected")

        c = controller.Controller.from_joystick(joy)
        self.assertIsInstance(c, controller.Controller)

    def test_as_joystick(self):
        c = self._get_first_controller()
        if c:
            joy = c.as_joystick()
            self.assertIsInstance(joy, type(pygame.joystick.Joystick(0)))
        else:
            self.skipTest("No controller connected")

    def test_get_mapping(self):
        c = self._get_first_controller()
        if c:
            mapping = c.get_mapping()
            self.assertIsInstance(mapping, dict)
            self.assertIsNotNone(mapping["a"])
        else:
            self.skipTest("No controller connected")

    def test_set_mapping(self):
        c = self._get_first_controller()
        if c:
            mapping = c.get_mapping()
            mapping["a"] = "b3"
            mapping["y"] = "b0"
            c.set_mapping(mapping)
            new_mapping = c.get_mapping()

            self.assertEqual(len(mapping), len(new_mapping))
            for i in mapping:
                if mapping[i] not in ("a", "y"):
                    self.assertEqual(mapping[i], new_mapping[i])
                else:
                    if i == "a":
                        self.assertEqual(new_mapping[i], mapping["y"])
                    else:
                        self.assertEqual(new_mapping[i], mapping["a"])
        else:
            self.skipTest("No controller connected")


class ControllerInteractiveTest(unittest.TestCase):
    __tags__ = ["interactive"]

    def setUp(self):
        controller.init()

    def tearDown(self):
        controller.quit()

    def test__get_count_interactive(self):
        prompt("Please connect at least one controller "
               "before the test for controller.get_count() starts")

        joystick_num = controller.get_count()
        ans = question("get_count() thinks there are {} joysticks "
                       "connected. Is that correct?".format(joystick_num))

        self.assertTrue(ans)

    def todo_test_get_button_interactive(self):
        self.fail()

    def todo_test_get_axis_interactive(self):
        self.fail()


if __name__ == "__main__":
    unittest.main()
