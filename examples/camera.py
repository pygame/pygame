#!/usr/bin/env python
""" pygame.examples.camera

Basic image capturing and display using pygame.camera

Keyboard controls
-----------------

- 0, start camera 0.
- 1, start camera 1.
- 9, start camera 9.
- 10, start camera... wait a minute! There's not 10 key!
"""
import pygame as pg
import pygame.camera


class VideoCapturePlayer(object):

    size = (640, 480)

    def __init__(self, **argd):
        self.__dict__.update(**argd)
        super(VideoCapturePlayer, self).__init__(**argd)

        # create a display surface. standard pygame stuff
        self.display = pg.display.set_mode(self.size)
        self.init_cams(0)

    def init_cams(self, which_cam_idx):

        # gets a list of available cameras.
        self.clist = pygame.camera.list_cameras()
        print(self.clist)

        if not self.clist:
            raise ValueError("Sorry, no cameras detected.")

        try:
            cam_id = self.clist[which_cam_idx]
        except IndexError:
            cam_id = self.clist[0]

        # creates the camera of the specified size and in RGB colorspace
        self.camera = pygame.camera.Camera(cam_id, self.size, "RGB")

        # starts the camera
        self.camera.start()

        self.clock = pg.time.Clock()

        # create a surface to capture to.  for performance purposes, you want the
        # bit depth to be the same as that of the display surface.
        self.snapshot = pg.surface.Surface(self.size, 0, self.display)

    def get_and_flip(self):
        # if you don't want to tie the framerate to the camera, you can check and
        # see if the camera has an image ready.  note that while this works
        # on most cameras, some will never return true.

        self.snapshot = self.camera.get_image(self.display)

        # if 0 and self.camera.query_image():
        #     # capture an image

        #     self.snapshot = self.camera.get_image(self.snapshot)

        # if 0:
        #     self.snapshot = self.camera.get_image(self.snapshot)
        #     # self.snapshot = self.camera.get_image()

        #     # blit it to the display surface.  simple!
        #     self.display.blit(self.snapshot, (0, 0))
        # else:

        #     self.snapshot = self.camera.get_image(self.display)
        #     # self.display.blit(self.snapshot, (0,0))

        pg.display.flip()

    def main(self):
        going = True
        while going:
            events = pg.event.get()
            for e in events:
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    going = False
                if e.type == pg.KEYDOWN:
                    if e.key in range(pg.K_0, pg.K_0 + 10):
                        self.init_cams(e.key - pg.K_0)

            self.get_and_flip()
            self.clock.tick()
            pygame.display.set_caption("CAMERA! (" + str(round(self.clock.get_fps())) + " FPS)")


def main():
    pg.init()
    pygame.camera.init()
    VideoCapturePlayer().main()
    pg.quit()


if __name__ == "__main__":
    main()
