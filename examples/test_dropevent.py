import unittest
import pygame as pg
import dropevent

class DropEventTestCase(unittest.TestCase):
    def setUp(self):
        pg.init()

    def test_dropfile(self):
        # Simulate a DROPFILE event
        drop_event = pg.event.Event(pg.DROPFILE, file="test_image.png")
        pg.event.post(drop_event)

        # Run the main function to process the event
        dropevent.main()

        # Verify that the file name is rendered as text
        expected_text = "test_image.png"
        self.assertTrue(dropevent)

        # Verify that the image is loaded and displayed
        self.assertIsNotNone(dropevent)

    def test_droptext(self):
        # Simulate a DROPTEXT event
        drop_event = pg.event.Event(pg.DROPTEXT, text="Hello, world!")
        pg.event.post(drop_event)

        # Run the main function to process the event
        dropevent.main()

        # Verify that the text is rendered
        expected_text = "Hello, world!"
        self.assertTrue(dropevent, expected_text)

        # Verify that no image is loaded
        self.assertIsNotNone(dropevent)

if __name__ == "__main__":
    unittest.main()
