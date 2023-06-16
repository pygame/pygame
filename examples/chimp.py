#!/usr/bin/env python
""" pygame.examples.chimp
This simple example is used for the line-by-line tutorial
that comes with pygame. It is based on a 'popular' web banner.
Note there are comments here, but for the full explanation,
follow along in the tutorial.
"""


from pathlib import Path  # Platform independent file paths.
import sys  # sys.exit()
import pygame as pg


class Media():

    def load_image(filename, colorkey=None, scale=1):
        filename = Path('data').joinpath(filename)
        image = pg.image.load(filename).convert()

        x, y = image.get_size()
        x = x * scale
        y = y * scale
        image = pg.transform.scale(image, (x,y))

        if colorkey == -1:
            colorkey = image.get_at((0, 0))

        image.set_colorkey(colorkey, pg.RLEACCEL)

        return image, image.get_rect()

    def load_sound(filename):
        if not pg.mixer or not pg.mixer.get_init():
            print("Warning, sound disabled")

            class NoneSound:
                def play(self):
                    pass

            return NoneSound()
        else:
            filename = Path('data').joinpath(filename)
            return pg.mixer.Sound(filename)


class Fist(pg.sprite.Sprite):
    """Clenched fist (follows mouse)"""

    def __init__(self):
        pg.sprite.Sprite.__init__(self)  # Call before adding sprite to groups.
        self.image, self.rect = Media.load_image("fist.png", colorkey=-1)
        self.whiff_sound = Media.load_sound("whiff.wav")
        self.punch_sound = Media.load_sound("punch.wav")
        self.state = self.FIST_RETRACTED = (-235, -80)
        self.FIST_PUNCHING = (-220, -55)
        self.add(Game.allsprites)

    def update(self, mouse_buttons):
        # Move fist to mouse position
        self.rect.topleft = pg.mouse.get_pos()
        self.rect.move_ip(self.state)
        
        # Handle punches
        for event in mouse_buttons:
            if event.type == pg.MOUSEBUTTONDOWN:
                self.state = self.FIST_PUNCHING
                self.rect.move_ip((15, 25))
                self.punch()
            if event.type == pg.MOUSEBUTTONUP:
                self.state = self.FIST_RETRACTED
                self.rect.move_ip((-15, -25))

    def punch(self):
        punched = pg.sprite.spritecollide(
                      self,
                      Game.punchables,
                      False,
                      collided=pg.sprite.collide_rect_ratio(.9))
        
        if len(punched) == 0:
            self.whiff_sound.play()
        else:
            for each in punched:
                each.get_punched(self)
                self.punch_sound.play()


class Chimp(pg.sprite.Sprite):
    """Monkey (moves across the screen)
    Spins when punched.
    """

    def __init__(self, topleft=(10,90)):
        pg.sprite.Sprite.__init__(self)
        self.image, self.rect = Media.load_image("chimp.png", colorkey=-1, scale=4)
        self.image_original = self.image
        self.rect.topleft = topleft
        self.rotation = 0
        self.delta = 18
        self.move = self._walk
        self.add((Game.allsprites, Game.punchables))

    def update(self, *nevermint):
        """Walk or spin, depending on state"""
        self.move()

    def _walk(self):
        """Move monkey across screen (turning at ends)"""
        newpos = self.rect.move((self.delta, 0))
        if not Game.screen_rect.contains(newpos):
            self.delta = -self.delta
            newpos = self.rect.move((self.delta, 0))
            self.image = pg.transform.flip(self.image, True, False)

        self.rect = newpos

    def _spin(self):
        """Spin monkey image"""
        center = self.rect.center
        self.rotation += 12
        if self.rotation >= 360:
            self.rotation = 0
            self.move = self._walk
            self.image = self.image_original
        else:
            rotate = pg.transform.rotate
            self.image = rotate(self.image_original, self.rotation)

        self.rect = self.image.get_rect(center=center)

    def get_punched(self, obj):
        """Cause monkey to spin"""
        self.move = self._spin

        
class Game:

    def initialize():
        pg.init()
        Game.screen = pg.display.set_mode((1280, 480))
        Game.screen_rect = Game.screen.get_rect()
        pg.display.set_caption("Monkey Fever")
        pg.mouse.set_visible(False)

        Game.create_background()
        Game.set_background_text()
        Game.prepare_game_objects()

        return Game

    def create_background():
        Game.background = pg.Surface(Game.screen.get_size())
        Game.background = Game.background.convert()
        Game.background.fill((170, 238, 187))

    def set_background_text():
        if pg.font:
            font = pg.font.Font(None, 64)
            text = font.render("Pummel The Chimp, And Win $$$",
                               True,
                               (10, 10, 10))
            # Centered
            textpos = text.get_rect(
                    centerx=Game.background.get_width() / 2,
                    y=10)
            Game.background.blit(text, textpos)
        else:
            print("Warning, fonts disabled")

    def prepare_game_objects():
        Game.allsprites = pg.sprite.RenderPlain()
        Game.punchables = pg.sprite.Group()
        Game.chimp = Chimp()
        #Game.chimp2 = Chimp((90,10))
        Game.fist = Fist()
        Game.clock = pg.time.Clock()

    def execute():
        while True:
            Game.clock.tick(60)

            # Handle Input Events
            mouse_buttons = pg.event.get(
                                eventtype=(pg.MOUSEBUTTONDOWN,
                                           pg.MOUSEBUTTONUP))

            for event in pg.event.get():
                if event.type == pg.QUIT or (
                        event.type == pg.KEYDOWN and
                                event.key == pg.K_ESCAPE):
                    pg.quit()
                    sys.exit()

            Game.allsprites.update(mouse_buttons)

            # Draw Everything
            Game.screen.blit(Game.background, (0, 0))
            Game.allsprites.draw(Game.screen)
            pg.display.flip()


if __name__ == "__main__":
    Game.initialize().execute()
